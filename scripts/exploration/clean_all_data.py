import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta

# --- CONFIGURATION ---
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
RAW_DATA_DIR = os.path.join(BASE_DIR, "data")
# Your extraction folder
EXTRACTED_DIR = os.path.join(RAW_DATA_DIR, "contracts_claims_extracted") 

# Output Directories
CLEAN_DIR = os.path.join(RAW_DATA_DIR, "cleaned")
TASK1_OUT = os.path.join(CLEAN_DIR, "task1_contracts")
TASK2_ANCHOR_OUT = os.path.join(CLEAN_DIR, "task2_anchor")
TASK2_HIST_OUT = os.path.join(CLEAN_DIR, "task2_history")

# Create folders
for d in [TASK1_OUT, TASK2_ANCHOR_OUT, TASK2_HIST_OUT]:
    if not os.path.exists(d): os.makedirs(d)

# --- UTILS ---
def clean_text(x):
    """Trims whitespace and converts to Title Case"""
    if pd.isna(x) or str(x).strip() == "":
        return "Unknown"
    return str(x).strip().title()

def robust_date_parser(date_series):
    """
    Handles mixed formats: 
    - 06-02-22 (DD-MM-YY)
    - 11-DEC-24 (DD-MMM-YY)
    - Excel Serials
    """
    # 1. First Pass: Try standard day-first (matches 06-02-22)
    parsed = pd.to_datetime(date_series, dayfirst=True, errors='coerce')
    
    # 2. Second Pass: Try DD-MMM-YY (matches 11-DEC-24) for failures
    mask_nan = parsed.isna()
    if mask_nan.any():
        parsed.loc[mask_nan] = pd.to_datetime(date_series[mask_nan], format='%d-%b-%y', errors='coerce')
    
    # 3. Third Pass: Try standard ISO just in case
    mask_nan = parsed.isna()
    if mask_nan.any():
        parsed.loc[mask_nan] = pd.to_datetime(date_series[mask_nan], errors='coerce')
        
    return parsed

# ==========================================
# 1. CLEAN TASK 1 (CONTRACTS)
# ==========================================
def clean_task1():
    print("\n--- 🧹 Cleaning Task 1 (Contracts) ---")
    files = glob.glob(os.path.join(RAW_DATA_DIR, "*Contract*.xls*"))
    
    all_contracts = []
    
    for f in files:
        try:
            # Read (handles .xls and .xlsx)
            if f.lower().endswith('.xls'):
                df = pd.read_excel(f, engine='xlrd')
            else:
                df = pd.read_excel(f)
            
            # Standardize Headers (Upper + Strip)
            df.columns = df.columns.str.strip().str.upper()
            
            # --- MAPPING (Based on your DIAGNOSTICS) ---
            mapping = {
                'CONTRACT#': 'CONTRACT_ID',
                'CREATED_DATE': 'DATE',
                'CUSTOMER': 'CUSTOMER',
                'GLOBAL_ACCOUNT': 'GLOBAL_ACCOUNT',
                'CHANNEL': 'CHANNEL',
                'TYPE': 'TYPE',
                'TACTIC_TYPE': 'TACTIC_TYPE',
                'PRODUCT_GROUP': 'PRODUCT_GROUP',
                'START_MONTH': 'START_MONTH',
                'END_MONTH': 'END_MONTH',
                'ACTUAL_REBATE': 'TARGET_VALUE',       # Primary Target
                'FIXED_REBATE_AMOUNT': 'FIXED_TARGET'  # Fallback Target
            }
            df = df.rename(columns=mapping)
            
            # Handle Target (Merge Percentage and Fixed Amount)
            if 'TARGET_VALUE' in df.columns:
                # If Actual Rebate is missing/0, try Fixed Amount
                df['TARGET_VALUE'] = pd.to_numeric(df['TARGET_VALUE'], errors='coerce').fillna(0)
                if 'FIXED_TARGET' in df.columns:
                    fixed_vals = pd.to_numeric(df['FIXED_TARGET'], errors='coerce').fillna(0)
                    # Simple logic: If percent is 0, take fixed.
                    # (Note: In reality, mixing % and Abs is tricky, but this ensures data isn't lost)
                    mask_zero = df['TARGET_VALUE'] == 0
                    df.loc[mask_zero, 'TARGET_VALUE'] = fixed_vals[mask_zero]
            
            # Keep only mapped columns that exist
            valid_cols = [c for c in mapping.values() if c in df.columns]
            df = df[valid_cols]
            
            all_contracts.append(df)
            
        except Exception as e:
            print(f"   ! Skipped {os.path.basename(f)}: {e}")

    if all_contracts:
        master_df = pd.concat(all_contracts, ignore_index=True)
        
        # Clean Text Columns
        text_cols = ['CUSTOMER', 'GLOBAL_ACCOUNT', 'CHANNEL', 'TYPE', 'TACTIC_TYPE', 'PRODUCT_GROUP']
        for c in text_cols:
            if c in master_df.columns:
                master_df[c] = master_df[c].apply(clean_text)
                
        # Save
        out_path = os.path.join(TASK1_OUT, "task1_contracts_clean.csv")
        master_df.to_csv(out_path, index=False)
        print(f"   ✅ Task 1 Cleaned: {len(master_df):,} rows saved.")
    else:
        print("   ❌ No Task 1 data found.")

# ==========================================
# 2. CLEAN TASK 2 (ANCHOR / DUMP FILES)
# ==========================================
def clean_task2_anchor():
    print("\n--- 🧹 Cleaning Task 2 (Anchor/Claims) ---")
    
    # We look in the Raw Data folder for those "Dump" files you showed
    # Or in the extracted folder if you moved them there. 
    # Based on your diagnostics, they seem to be in 'data' or 'extracted'.
    # We'll search recursively to be safe.
    
    search_pattern = os.path.join(RAW_DATA_DIR, "**", "*Claim_Dump*.xls*")
    files = glob.glob(search_pattern, recursive=True)
    
    # Also check extracted folder explicitly
    if not files:
        files = glob.glob(os.path.join(EXTRACTED_DIR, "*.csv")) # Fallback to CSVs if dumps aren't there

    if not files:
        print("   ❌ No Claim Dump files found!")
        return

    all_claims = []
    
    for f in files:
        try:
            print(f"   > Processing: {os.path.basename(f)}")
            # Read (Support Excel & CSV)
            if f.endswith('.xls'):
                df = pd.read_excel(f, engine='xlrd')
            elif f.endswith('.xlsx'):
                df = pd.read_excel(f)
            else:
                df = pd.read_csv(f, low_memory=False)
            
            df.columns = df.columns.str.strip().str.upper()
            
            # --- MAPPING ---
            # Based on your output: CLAIM_DATE, CUSTOMER_NAME, AMOUNT, PRODUCT_GROUP, GEOGRAPHICAL_CLUSTER
            mapping = {
                'CUSTOMER_NAME': 'CUSTOMER',
                'CLAIM_DATE': 'DATE',
                'AMOUNT': 'SALES',
                'PRODUCT_GROUP': 'PRODUCT_GROUP',
                'GEOGRAPHICAL_CLUSTER': 'GEOGRAPHICAL_CLUSTER'
            }
            df = df.rename(columns=mapping)
            
            # Check essential columns
            if 'DATE' not in df.columns or 'SALES' not in df.columns:
                continue
                
            # Date Parsing
            df['DATE'] = robust_date_parser(df['DATE'])
            
            # Drop Invalid
            df = df.dropna(subset=['DATE', 'SALES'])
            
            # Clean Text
            for c in ['CUSTOMER', 'PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER']:
                if c in df.columns: df[c] = df[c].apply(clean_text)
                
            # Keep only needed
            keep = ['CUSTOMER', 'DATE', 'SALES', 'PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER']
            df = df[[c for c in keep if c in df.columns]]
            
            all_claims.append(df)
            
        except Exception as e:
            print(f"   ! Error: {e}")

    if all_claims:
        master_df = pd.concat(all_claims, ignore_index=True)
        out_path = os.path.join(TASK2_ANCHOR_OUT, "task2_anchor_clean.csv")
        master_df.to_csv(out_path, index=False)
        print(f"   ✅ Task 2 Anchor Cleaned: {len(master_df):,} rows saved.")

# ==========================================
# 3. CLEAN TASK 2 HISTORY (SIP/TACTIC)
# ==========================================
def clean_task2_history():
    print("\n--- 🧹 Cleaning Task 2 (History SIP/TACTIC) ---")
    
    def find(pat):
        m = glob.glob(os.path.join(RAW_DATA_DIR, f"*{pat}*"))
        return m[0] if m else None

    # SIP
    sip_c = find("SIP_CLAIM_DIMNS")
    sip_p = find("SIP_PLAN_DIMNS")
    
    if sip_c and sip_p:
        try:
            sc = pd.read_csv(sip_c, low_memory=False)
            sp = pd.read_csv(sip_p, low_memory=False)
            
            # UAE Filter
            if 'SIP_LOCATION' in sp.columns:
                sp = sp[sp['SIP_LOCATION'].astype(str).str.contains('United Arab Emirates', case=False, na=False)]
            
            merged = pd.merge(sc, sp, on='TACTIC_NUMBER', how='inner')
            
            df = pd.DataFrame()
            df['CUSTOMER'] = merged['Customer_Name']
            df['PRODUCT_GROUP'] = merged['PRODUCT_GROUP']
            df['GEOGRAPHICAL_CLUSTER'] = merged.get('GEO_CLUST', 'United Arab Emirates')
            df['DATE'] = merged['CLAIM_CREATION_DATE']
            df['SALES'] = pd.to_numeric(merged['CLAIM_AMOUNT'], errors='coerce')
            
            # Clean
            df['DATE'] = robust_date_parser(df['DATE'])
            for c in ['CUSTOMER', 'PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER']:
                df[c] = df[c].apply(clean_text)
                
            df = df.dropna(subset=['DATE', 'SALES'])
            df.to_csv(os.path.join(TASK2_HIST_OUT, "sip_clean.csv"), index=False)
            print(f"   ✅ SIP History Cleaned: {len(df):,} rows.")
        except Exception as e: print(f"   ! SIP Error: {e}")

    # TACTIC
    tac_c = find("TACTIC_CM_CLAIMS")
    tac_p = find("TACTIC_PLAN_DIMNS")
    
    if tac_c and tac_p:
        try:
            tc = pd.read_csv(tac_c, low_memory=False)
            tp = pd.read_csv(tac_p, low_memory=False)
            
            if 'TACTIC_LOCATION' in tp.columns:
                tp = tp[tp['TACTIC_LOCATION'].astype(str).str.contains('United Arab Emirates', case=False, na=False)]
            
            merged = pd.merge(tc, tp, on='TACTIC_NUMBER', how='inner')
            
            df = pd.DataFrame()
            # Tactic usually lacks Customer Name in claim file, uses ID. 
            # We will save ID here and map it during training using Anchor data.
            df['EBS_ID'] = merged['EBS_PARTY_SITE_ID']
            df['PRODUCT_GROUP'] = merged['PRODUCT_GROUP']
            df['GEOGRAPHICAL_CLUSTER'] = merged.get('GEO_CLUST', 'United Arab Emirates')
            df['DATE'] = merged['CLAIM_CREATION_DATE']
            df['SALES'] = pd.to_numeric(merged['CLAIM_AMOUNT'], errors='coerce')
            
            df['DATE'] = robust_date_parser(df['DATE'])
            for c in ['PRODUCT_GROUP', 'GEOGRAPHICAL_CLUSTER']:
                df[c] = df[c].apply(clean_text)
                
            df = df.dropna(subset=['DATE', 'SALES'])
            df.to_csv(os.path.join(TASK2_HIST_OUT, "tactic_clean.csv"), index=False)
            print(f"   ✅ TACTIC History Cleaned: {len(df):,} rows.")
        except Exception as e: print(f"   ! TACTIC Error: {e}")

if __name__ == "__main__":
    clean_task1()
    clean_task2_anchor()
    clean_task2_history()
    print("\n✨ Data Cleaning Complete. Check 'data/cleaned' folder.")