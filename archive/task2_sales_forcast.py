import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")
# TARGET: Only the extracted CSV folder
ANCHOR_DIR = os.path.join(DATA_DIR, "contracts_claims_extracted") 
MODELS_DIR = os.path.join(BASE_DIR, "models")

# --- ROBUST DATE PARSER ---
def smart_date_parser(date_series):
    # 1. Try DD-MM-YY (e.g. 06-02-22) - Most common in CSVs
    parsed = pd.to_datetime(date_series, dayfirst=True, errors='coerce')
    
    # 2. Try DD-MMM-YY (e.g. 11-DEC-24) - Fallback
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(date_series[mask], format='%d-%b-%y', errors='coerce')
        
    # 3. Last Resort
    mask = parsed.isna()
    if mask.any():
        parsed.loc[mask] = pd.to_datetime(date_series[mask], errors='coerce')
        
    return parsed

def clean_text(x):
    if pd.isna(x) or str(x).strip() == "": return "Unknown"
    return str(x).strip().title()

def train_task2_csv_only():
    print("--- Starting Task 2: V11 (Extracted CSVs Only) ---")
    
    full_df = pd.DataFrame()
    cust_lookup = {} 

    # =========================================================
    # 1. LOAD ANCHOR (EXTRACTED CSVs ONLY)
    # =========================================================
    print(f"\nStep 1: Loading CSVs from {ANCHOR_DIR}...")
    
    csv_files = glob.glob(os.path.join(ANCHOR_DIR, "*.csv"))
    
    if not csv_files:
        print("❌ Error: No .csv files found in the extracted folder!")
        return

    for f in csv_files:
        try:
            # low_memory=False is crucial for large CSVs to guess types correctly
            df = pd.read_csv(f, low_memory=False)
            
            # Standardize Headers
            df.columns = df.columns.str.strip().str.upper()
            
            # Flexible Mapping (Handles both raw extracted and cleaned names)
            rename_map = {
                'CUSTOMER_NAME': 'CUSTOMER', 
                'CLAIM_DATE': 'DATE', 
                'AMOUNT': 'SALES', 
                'PRODUCT_GROUP': 'PRODUCT_GROUP', 
                'GEOGRAPHICAL_CLUSTER': 'GEOGRAPHICAL_CLUSTER',
                # Variations found in some CSVs
                'CUSTOMER NAME': 'CUSTOMER',
                'CLAIM DATE': 'DATE',
                'AMOUNT': 'SALES'
            }
            df = df.rename(columns=rename_map)
            
            # Validate
            if 'DATE' not in df.columns or 'SALES' not in df.columns:
                print(f"     ⚠️ Skipping {os.path.basename(f)} - Missing DATE or SALES")
                continue
                
            # Build Customer Lookup (for Tactic files)
            if 'EBS_PARTY_SITE_ID' in df.columns and 'CUSTOMER' in df.columns:
                temp = df[['EBS_PARTY_SITE_ID', 'CUSTOMER']].dropna().drop_duplicates('EBS_PARTY_SITE_ID')
                cust_lookup.update(temp.set_index('EBS_PARTY_SITE_ID')['CUSTOMER'].to_dict())

            # Filter columns
            cols = ['CUSTOMER', 'PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER', 'DATE', 'SALES']
            
            # Add missing region if needed
            if 'GEOGRAPHICAL_CLUSTER' not in df.columns: 
                df['GEOGRAPHICAL_CLUSTER'] = "United Arab Emirates"
            
            df = df[[c for c in cols if c in df.columns]]
            df['SOURCE'] = 'ANCHOR'
            
            full_df = pd.concat([full_df, df], ignore_index=True)
            
        except Exception as e:
            print(f"   ! Error processing {os.path.basename(f)}: {e}")

    print(f"   ✅ Anchor Data Loaded: {len(full_df):,} rows")

    # =========================================================
    # 2. LOAD HISTORY (SIP & TACTIC)
    # =========================================================
    print("\nStep 2: Loading Historical Data (UAE Only)...")
    
    def find_file(name_part):
        # Look in main DATA_DIR for SIP/TACTIC files
        matches = glob.glob(os.path.join(DATA_DIR, f"*{name_part}*"))
        return matches[0] if matches else None

    # --- SIP ---
    sip_c = find_file("SIP_CLAIM_DIMNS")
    sip_p = find_file("SIP_PLAN_DIMNS")
    
    if sip_c and sip_p:
        try:
            sc = pd.read_csv(sip_c, low_memory=False)
            sp = pd.read_csv(sip_p, low_memory=False)
            
            # UAE Filter
            if 'SIP_LOCATION' in sp.columns:
                sp = sp[sp['SIP_LOCATION'].astype(str).str.contains('United Arab Emirates', case=False, na=False)]
            
            merged = pd.merge(sc, sp, on='TACTIC_NUMBER', how='inner')
            
            sip_df = pd.DataFrame()
            sip_df['CUSTOMER'] = merged.get('Customer_Name', merged.get('CUSTOMER_NAME'))
            sip_df['PRODUCT_GROUP'] = merged['PRODUCT_GROUP']
            sip_df['GEOGRAPHICAL_CLUSTER'] = merged.get('GEO_CLUST', 'United Arab Emirates')
            sip_df['DATE'] = merged['CLAIM_CREATION_DATE']
            sip_df['SALES'] = pd.to_numeric(merged['CLAIM_AMOUNT'], errors='coerce')
            sip_df['SOURCE'] = 'SIP'
            
            full_df = pd.concat([full_df, sip_df], ignore_index=True)
            print(f"   ✅ Added SIP History: {len(sip_df):,} rows")
        except: pass

    # --- TACTIC ---
    tac_c = find_file("TACTIC_CM_CLAIMS")
    tac_p = find_file("TACTIC_PLAN_DIMNS")
    
    if tac_c and tac_p:
        try:
            tc = pd.read_csv(tac_c, low_memory=False)
            tp = pd.read_csv(tac_p, low_memory=False)
            
            if 'TACTIC_LOCATION' in tp.columns:
                tp = tp[tp['TACTIC_LOCATION'].astype(str).str.contains('United Arab Emirates', case=False, na=False)]
            
            merged = pd.merge(tc, tp, on='TACTIC_NUMBER', how='inner')
            
            tac_df = pd.DataFrame()
            # Map ID to Name using Anchor Lookup
            tac_df['CUSTOMER'] = merged['EBS_PARTY_SITE_ID'].map(cust_lookup).fillna("Unknown Tactic Cust")
            tac_df['PRODUCT_GROUP'] = merged['PRODUCT_GROUP']
            tac_df['GEOGRAPHICAL_CLUSTER'] = merged.get('GEO_CLUST', 'United Arab Emirates')
            tac_df['DATE'] = merged['CLAIM_CREATION_DATE']
            tac_df['SALES'] = pd.to_numeric(merged['CLAIM_AMOUNT'], errors='coerce')
            tac_df['SOURCE'] = 'TACTIC'
            
            full_df = pd.concat([full_df, tac_df], ignore_index=True)
            print(f"   ✅ Added TACTIC History: {len(tac_df):,} rows")
        except: pass

    # =========================================================
    # 3. CLEANING & FEATURE ENGINEERING
    # =========================================================
    print(f"\nStep 3: Processing {len(full_df):,} Total Rows...")
    
    # 1. Parse Dates
    full_df['DATE'] = smart_date_parser(full_df['DATE'])
    
    # 2. Drop Invalid
    full_df = full_df.dropna(subset=['DATE', 'SALES'])
    full_df = full_df[full_df['SALES'] > 0]
    
    # 3. Clean Text
    for c in ['CUSTOMER', 'PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER']:
        full_df[c] = full_df[c].apply(clean_text)

    # 4. Save History File for App (Blue Line)
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    full_df['Plot_Date'] = full_df['DATE'].apply(lambda x: x.replace(day=1))
    history_monthly = full_df.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'Plot_Date'])['SALES'].sum().reset_index()
    joblib.dump(history_monthly, os.path.join(MODELS_DIR, 'history_monthly.pkl'))
    print("   ✅ History File Saved (for Blue Line in App).")

    # 5. Aggregate for Training
    full_df['Month'] = full_df['DATE'].dt.month
    full_df['Cust_Prod'] = full_df['CUSTOMER'] + "_" + full_df['PRODUCT_GROUP']
    
    monthly_data = full_df.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'Cust_Prod', 'GEOGRAPHICAL_CLUSTER', 'Month'])['SALES'].sum().reset_index()

    # =========================================================
    # 4. TRAINING (V9 RECENCY)
    # =========================================================
    print("\nStep 4: Training Model...")
    
    # Encodings
    cust_mean = monthly_data.groupby('CUSTOMER')['SALES'].mean().to_dict()
    prod_mean = monthly_data.groupby('PRODUCT_GROUP')['SALES'].mean().to_dict()
    geo_mean = monthly_data.groupby('GEOGRAPHICAL_CLUSTER')['SALES'].mean().to_dict()
    cust_prod_mean = monthly_data.groupby('Cust_Prod')['SALES'].mean().to_dict()

    # Recency (Last 12 Months)
    max_date = full_df['DATE'].max()
    cutoff_date = max_date - pd.DateOffset(months=12)
    recent_df = full_df[full_df['DATE'] >= cutoff_date]
    recent_monthly_avg = recent_df.groupby(['Cust_Prod', 'Month'])['SALES'].sum().groupby('Cust_Prod').mean().to_dict()
    
    # Map Features
    monthly_data['Cust_Avg'] = monthly_data['CUSTOMER'].map(cust_mean)
    monthly_data['Prod_Avg'] = monthly_data['PRODUCT_GROUP'].map(prod_mean)
    monthly_data['Geo_Avg'] = monthly_data['GEOGRAPHICAL_CLUSTER'].map(geo_mean)
    monthly_data['Cust_Prod_Avg'] = monthly_data['Cust_Prod'].map(cust_prod_mean)
    monthly_data['Recent_Avg'] = monthly_data['Cust_Prod'].map(recent_monthly_avg)
    
    # Fallbacks
    monthly_data['Recent_Avg'] = monthly_data['Recent_Avg'].fillna(monthly_data['Cust_Prod_Avg'])
    monthly_data['Cust_Prod_Avg'] = monthly_data['Cust_Prod_Avg'].fillna(monthly_data['Prod_Avg'])
    monthly_data['Is_Summer'] = monthly_data['Month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
    
    # Train
    features = ['Cust_Avg', 'Prod_Avg', 'Geo_Avg', 'Cust_Prod_Avg', 'Recent_Avg', 'Month', 'Is_Summer']
    X = monthly_data[features]
    y = np.log1p(monthly_data['SALES']) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, max_depth=12, random_state=42)
    model.fit(X_train, y_train)
    
    mae = mean_absolute_error(np.expm1(y_test), np.expm1(model.predict(X_test)))
    print(f"   ✅ Model Trained! MAE: {mae:,.2f} AED")
    
    # Save Model & Encoders
    encodings = {
        'cust_means': cust_mean, 'prod_means': prod_mean, 
        'geo_means': geo_mean, 'cust_prod_means': cust_prod_mean,
        'recent_means': recent_monthly_avg
    }
    
    joblib.dump(model, os.path.join(MODELS_DIR, 'task2_sales_forecasting_model_v4.pkl'))
    joblib.dump(encodings, os.path.join(MODELS_DIR, 'task2_encoders_v4.pkl'))
    
    # Save Metadata
    anchor_only = full_df[full_df['SOURCE'] == 'ANCHOR']
    metadata = {
        'products': sorted(anchor_only['PRODUCT_GROUP'].unique().tolist()),
        'regions': sorted(anchor_only['GEOGRAPHICAL_CLUSTER'].unique().tolist()),
        'customers': sorted(anchor_only['CUSTOMER'].unique().tolist())
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, 'task2_metadata_v4.pkl'))
    print("✅ All Files Saved. Ready for App!")

if __name__ == "__main__":
    train_task2_csv_only()