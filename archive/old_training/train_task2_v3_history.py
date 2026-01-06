import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# File Names (Update if yours differ)
FILES = {
    'sip_claim': 'TDA_FY 21_22_F1000.xlsx - SIP_CLAIM_DIMNS_F1000.csv',
    'sip_plan': 'TDA_FY 21_22_F1000.xlsx - SIP_PLAN_DIMNS_F1000.csv',
    'tactic_claim': 'TACTIC_21_22_F1000.xlsx - TACTIC_CM_CLAIMS_DIMNS_F1000.csv',
    'tactic_plan': 'TACTIC_21_22_F1000.xlsx - TACTIC_PLAN_DIMNS_F1000.csv',
    'current_2025': 'Contract_Claim_Dump-Jan25-Nov25_IDS_OU.xlsx' # The big 400k file
}

def clean_text(x):
    return str(x).strip().title()

def parse_date(date_value):
    """Robust date parser that handles multiple formats."""
    if pd.isna(date_value):
        return pd.NaT
    
    try:
        # First try inferring format
        return pd.to_datetime(date_value, infer_datetime_format=True)
    except:
        try:
            # Try dayfirst format (DD/MM/YYYY)
            return pd.to_datetime(date_value, dayfirst=True, format='mixed')
        except:
            try:
                # Try US format (MM/DD/YYYY)
                return pd.to_datetime(date_value, dayfirst=False, format='mixed')
            except:
                return pd.NaT

def train_model_v3():
    print("--- Starting Task 2 V3 (Historical Enrichment) ---")
    
    # ---------------------------------------------------------
    # 1. LOAD & PROCESS CURRENT DATA (2025)
    # ---------------------------------------------------------
    path_2025 = os.path.join(DATA_DIR, FILES['current_2025'])
    if not os.path.exists(path_2025):
        print(f"Error: 2025 Data not found at {path_2025}")
        return

    print("Step 1: Loading 2025 Data (The Anchor)...")
    try:
        all_sheets = pd.read_excel(path_2025, sheet_name=None)
        df_2025 = pd.concat(all_sheets.values(), ignore_index=True)
        
        # Build Customer Lookup (EBS_ID -> Name) for the historical data
        if 'EBS_PARTY_SITE_ID' in df_2025.columns and 'CUSTOMER_NAME' in df_2025.columns:
            lookup_df = df_2025[['EBS_PARTY_SITE_ID', 'CUSTOMER_NAME']].dropna().drop_duplicates('EBS_PARTY_SITE_ID')
            cust_lookup = lookup_df.set_index('EBS_PARTY_SITE_ID')['CUSTOMER_NAME'].to_dict()
            print(f"   > Built lookup for {len(cust_lookup)} customers.")
        else:
            cust_lookup = {}
            
        # Standardize 2025
        df_2025 = df_2025.rename(columns={'CUSTOMER_NAME': 'CUSTOMER', 'CLAIM_DATE': 'DATE', 'AMOUNT': 'SALES'})
        df_2025 = df_2025[['CUSTOMER', 'PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER', 'DATE', 'SALES']]
        df_2025['SOURCE'] = '2025_ACTUALS'
        
    except Exception as e:
        print(f"Error loading 2025 data: {e}")
        return

    # ---------------------------------------------------------
    # 2. PROCESS HISTORICAL DATA (2021-2022)
    # ---------------------------------------------------------
    print("Step 2: Processing 2021-2022 History...")
    history_rows = []
    
    # Load CSVs
    try:
        sip_claim = pd.read_csv(os.path.join(DATA_DIR, FILES['sip_claim']))
        sip_plan = pd.read_csv(os.path.join(DATA_DIR, FILES['sip_plan']))
        tactic_claim = pd.read_csv(os.path.join(DATA_DIR, FILES['tactic_claim']))
        tactic_plan = pd.read_csv(os.path.join(DATA_DIR, FILES['tactic_plan']))
        
        # A. Process SIP (Has Names)
        # Merge on TACTIC_NUMBER
        sip_merged = pd.merge(sip_claim, sip_plan[['TACTIC_NUMBER', 'PRODUCT_GROUP', 'GEO_CLUST']], 
                              on='TACTIC_NUMBER', how='inner')
        
        for _, row in sip_merged.iterrows():
            history_rows.append({
                'CUSTOMER': row.get('Customer_Name'), # Note capitalization in CSV
                'PRODUCT_GROUP': row.get('PRODUCT_GROUP'),
                'GEOGRAPHICAL_CLUSTER': row.get('GEO_CLUST'),
                'DATE': row.get('CLAIM_CREATION_DATE'),
                'SALES': row.get('CLAIM_AMOUNT'),
                'SOURCE': 'HISTORY_SIP'
            })
            
        # B. Process TACTIC (Has IDs)
        tactic_merged = pd.merge(tactic_claim, tactic_plan[['TACTIC_NUMBER', 'PRODUCT_GROUP', 'GEO_CLUST']], 
                                 on='TACTIC_NUMBER', how='inner')
        
        for _, row in tactic_merged.iterrows():
            # Try to find name from 2025 lookup
            ebs_id = row.get('EBS_PARTY_SITE_ID')
            cust_name = cust_lookup.get(ebs_id, "Unknown Historical")
            
            history_rows.append({
                'CUSTOMER': cust_name,
                'PRODUCT_GROUP': row.get('PRODUCT_GROUP'),
                'GEOGRAPHICAL_CLUSTER': row.get('GEO_CLUST'),
                'DATE': row.get('CLAIM_CREATION_DATE'),
                'SALES': row.get('CLAIM_AMOUNT'),
                'SOURCE': 'HISTORY_TACTIC'
            })
            
        df_history = pd.DataFrame(history_rows)
        print(f"   > Recovered {len(df_history)} valid historical records.")
        
    except Exception as e:
        print(f"Error processing history files: {e}")
        print("Continuing with only 2025 data...")
        df_history = pd.DataFrame()

    # ---------------------------------------------------------
    # 3. UNIFY & TRAIN
    # ---------------------------------------------------------
    print("Step 3: Unifying Datasets...")
    full_df = pd.concat([df_2025, df_history], ignore_index=True)
    
    # Cleaning - Use robust date parser
    full_df['DATE'] = full_df['DATE'].apply(parse_date)
    full_df = full_df.dropna(subset=['DATE', 'SALES'])
    
    # Text Standardization
    for col in ['CUSTOMER', 'PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER']:
        full_df[col] = full_df[col].apply(clean_text)

    # Aggregation (Monthly)
    full_df['Month'] = full_df['DATE'].dt.month
    
    monthly_data = full_df.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER', 'Month'])['SALES'].sum().reset_index()
    
    # Season Flags
    monthly_data['Is_Summer'] = monthly_data['Month'].apply(lambda x: 1 if x in [6, 7, 8] else 0)
    monthly_data['Is_Winter'] = monthly_data['Month'].apply(lambda x: 1 if x in [11, 12, 1, 2] else 0)

    print(f"Step 4: Training V3 Model on {len(monthly_data)} patterns...")
    
    # Encoding
    le_cust = LabelEncoder()
    le_prod = LabelEncoder()
    le_geo = LabelEncoder()

    monthly_data['Cust_Code'] = le_cust.fit_transform(monthly_data['CUSTOMER'])
    monthly_data['Prod_Code'] = le_prod.fit_transform(monthly_data['PRODUCT_GROUP'])
    monthly_data['Geo_Code'] = le_geo.fit_transform(monthly_data['GEOGRAPHICAL_CLUSTER'])

    # Train
    X = monthly_data[['Cust_Code', 'Prod_Code', 'Geo_Code', 'Month', 'Is_Summer', 'Is_Winter']]
    y = monthly_data['SALES']
    
    model = GradientBoostingRegressor(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Save V3
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    # We save as 'v2' name to overwrite the previous one so your App automatically picks it up!
    joblib.dump(model, os.path.join(MODELS_DIR, 'sales_forecasting_model_v3.pkl'))
    
    encoders = {'cust': le_cust, 'prod': le_prod, 'geo': le_geo}
    joblib.dump(encoders, os.path.join(MODELS_DIR, 'encoders_v3.pkl'))
    
    metadata = {
        'products': sorted(le_prod.classes_.tolist()),
        'regions': sorted(le_geo.classes_.tolist()),
        'customers': sorted(le_cust.classes_.tolist()) 
    }
    joblib.dump(metadata, os.path.join(MODELS_DIR, 'task2_metadata.pkl'))

    print("✅ V3 Model Trained & Saved (Overwrote V2 files for seamless App integration).")

if __name__ == "__main__":
    train_model_v3()