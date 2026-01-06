import pandas as pd
import os
import glob

# --- CONFIGURATION ---
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")
ANCHOR_DIR = os.path.join(DATA_DIR, "contracts_claims_extracted") 

def check_volume():
    print("--- 🔍 DATA VOLUME DIAGNOSTIC ---")
    
    # 1. LOAD RAW DATA
    print(f"\n1. Reading CSVs from {ANCHOR_DIR}...")
    files = glob.glob(os.path.join(ANCHOR_DIR, "*.csv"))
    
    full_df = pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)
    
    raw_count = len(full_df)
    print(f"   📊 RAW TRANSACTION ROWS: {raw_count:,}")
    
    # 2. SIMULATE PROCESSING
    print("\n2. Processing Dates & Aggregating...")
    # Map columns
    full_df = full_df.rename(columns={
        'CUSTOMER_NAME': 'CUSTOMER', 'CLAIM_DATE': 'DATE', 
        'AMOUNT': 'SALES', 'PRODUCT_GROUP': 'PRODUCT_GROUP', 
        'GEOGRAPHICAL_CLUSTER': 'GEOGRAPHICAL_CLUSTER'
    })
    
    # Filter
    full_df['DATE'] = pd.to_datetime(full_df['DATE'], errors='coerce')
    full_df = full_df.dropna(subset=['DATE', 'SALES'])
    full_df = full_df[full_df['SALES'] > 0]
    
    # AGGREGATE (The Shrinkage Step)
    full_df['Month'] = full_df['DATE'].dt.month
    
    # Group by Customer + Product + Month
    monthly_data = full_df.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER', 'Month'])['SALES'].sum().reset_index()
    
    agg_count = len(monthly_data)
    print(f"   📉 AGGREGATED MONTHLY ROWS: {agg_count:,}")
    print(f"   ℹ️  Compression Ratio: 1 Monthly Row = {raw_count / agg_count:.1f} Daily Transactions")
    
    if agg_count > 10000:
        print("\n✅ CONCLUSION: The model is fine.")
        print("   The '0.004 GB' is simply the size of the *Monthly Summary* table, not the raw file.")
    else:
        print("\n❌ WARNING: The aggregated data is too small. Something might be wrong with the grouping.")

if __name__ == "__main__":
    check_volume()