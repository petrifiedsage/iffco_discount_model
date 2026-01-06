import pandas as pd
import numpy as np
import os
import glob
import joblib

# --- CONFIGURATION ---
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")
ANCHOR_DIR = os.path.join(DATA_DIR, "contracts_claims_extracted") 
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- DATE FIXER FUNCTION ---
def robust_date_parser(date_series):
    # 1. Try Standard Format (Day-First)
    parsed = pd.to_datetime(date_series, dayfirst=True, errors='coerce')
    
    # 2. If many are missing, try strict Year-Month-Day
    if parsed.isna().sum() > len(date_series) * 0.5:
        print("   ⚠️ Standard parsing failed. Retrying with specific formats...")
        for fmt in ['%d-%m-%y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y']:
            temp = pd.to_datetime(date_series, format=fmt, errors='coerce')
            if temp.notna().sum() > parsed.notna().sum():
                parsed = temp
                print(f"   ✅ Format matched: {fmt}")
                break
    return parsed

def create_history_file_v2():
    print("--- Generating Robust History File (V2) ---")
    
    full_df = pd.DataFrame()
    
    # 1. LOAD ANCHOR FILES
    print(f"   > Reading CSVs from {ANCHOR_DIR}...")
    files = glob.glob(os.path.join(ANCHOR_DIR, "*.csv"))
    
    if not files:
        print("❌ Error: No CSV files found.")
        return

    df_list = []
    for f in files:
        try:
            df = pd.read_csv(f, low_memory=False)
            
            # Map Columns
            rename_map = {
                'CUSTOMER_NAME': 'CUSTOMER', 'CLAIM_DATE': 'DATE', 
                'AMOUNT': 'SALES', 'PRODUCT_GROUP': 'PRODUCT_GROUP', 
                'Customer Name': 'CUSTOMER', 'Claim Date': 'DATE', 'Amount': 'SALES'
            }
            df = df.rename(columns=rename_map)
            
            if 'CUSTOMER' in df.columns and 'SALES' in df.columns and 'DATE' in df.columns:
                df = df[['CUSTOMER', 'PRODUCT_GROUP', 'DATE', 'SALES']]
                df_list.append(df)
        except: pass

    if df_list:
        full_df = pd.concat(df_list, ignore_index=True)
        print(f"   📊 Raw Rows Loaded: {len(full_df):,}")
    else:
        print("❌ No valid data loaded.")
        return

    # 2. ROBUST DATE PARSING (The Fix)
    print("   > Parsing Dates (Aggressive Mode)...")
    full_df['DATE'] = robust_date_parser(full_df['DATE'])
    
    # Check for Dropped Data
    missing_dates = full_df['DATE'].isna().sum()
    if missing_dates > 0:
        print(f"   ⚠️ Warning: {missing_dates:,} rows dropped due to invalid dates.")

    full_df = full_df.dropna(subset=['DATE', 'SALES'])
    
    # 3. YEARLY CHECK (Verify 2022 exists)
    full_df['Year'] = full_df['DATE'].dt.year
    year_counts = full_df['Year'].value_counts().sort_index()
    print("\n   📅 DATA FOUND BY YEAR:")
    print(year_counts)
    
    if 2022 not in year_counts:
        print("\n❌ CRITICAL: 2022 Data is STILL missing. Please check the raw CSV dates.")
    else:
        print("\n✅ 2022 Data Detected!")

    # 4. AGGREGATE & SAVE
    full_df['CUSTOMER'] = full_df['CUSTOMER'].astype(str).str.strip().str.title()
    full_df['PRODUCT_GROUP'] = full_df['PRODUCT_GROUP'].astype(str).str.strip().str.title()
    
    # Normalize to 1st of Month
    full_df['Plot_Date'] = full_df['DATE'].apply(lambda x: x.replace(day=1))
    
    history_monthly = full_df.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'Plot_Date'])['SALES'].sum().reset_index()
    
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    joblib.dump(history_monthly, os.path.join(MODELS_DIR, 'history_monthly.pkl'))
    
    print(f"\n✅ History Saved: {len(history_monthly):,} monthly records.")
    print("👉 Restart App to see 2022 data.")

if __name__ == "__main__":
    create_history_file_v2()