import pandas as pd
import joblib
import os
import re

# --- CONFIGURATION ---
DATA_DIR = r"D:/BlueRose/iffco_discount_model/data"
OUTPUT_DIR = r"D:/BlueRose/iffco_discount_model/outputs"

def clean_customer_name(name):
    """
    Standardizes names to match Claims with Contracts.
    """
    if not isinstance(name, str):
        return "Unknown"
    
    name = str(name).upper().strip()
    name = name.replace("L.L.C", "").replace("LLC", "").replace(" FZCO", "")
    name = re.sub(r'\s*-[A-Z]$', '', name) # Remove -C, -F suffixes
    name = re.sub(r'\s*\(.*?\)', '', name) # Remove (H.O)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

def prepare_task3_data_v3():
    print("--- Preparing Data for Task 3 (V3: Full Audit) ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ---------------------------------------------------------
    # 1. PROCESS CONTRACTS
    # ---------------------------------------------------------
    contract_path = os.path.join(DATA_DIR, "Contract_Data_with_Product Group_IDS_OU_2025.xls")
    if not os.path.exists(contract_path): contract_path = contract_path.replace(".xls", ".xlsx")

    if os.path.exists(contract_path):
        print(f"\nReading Contracts from: {contract_path}")
        try:
            # Use pyexcel for .xls to avoid xlrd version conflicts
            if contract_path.endswith('.xls'):
                import pyexcel as pe
                all_sheets = {}
                sheet = pe.get_sheet(file_name=contract_path)
                data = sheet.get_array()
                all_sheets['Sheet1'] = pd.DataFrame(data[1:], columns=[str(h).strip() for h in data[0]])
            else:
                all_sheets = pd.read_excel(contract_path, sheet_name=None, engine='openpyxl')
            df_contracts = pd.concat(all_sheets.values(), ignore_index=True)
            
            print(f"📊 RAW CONTRACTS LOADED: {len(df_contracts):,} rows")

            # CLEANING
            df_contracts['CUSTOMER'] = df_contracts['CUSTOMER'].apply(clean_customer_name)
            df_contracts['PRODUCT_GROUP'] = df_contracts['PRODUCT_GROUP'].astype(str).str.strip().str.title()
            
            if 'CHANNEL' in df_contracts.columns:
                df_contracts['CHANNEL'] = df_contracts['CHANNEL'].astype(str).str.strip().str.upper()
            else:
                df_contracts['CHANNEL'] = "UNKNOWN"

            # Combine Actual and Fixed
            if 'FIXED_REBATE_AMOUNT' in df_contracts.columns:
                df_contracts['FIXED_REBATE_AMOUNT'] = pd.to_numeric(df_contracts['FIXED_REBATE_AMOUNT'], errors='coerce')
                df_contracts['ACTUAL_REBATE'] = pd.to_numeric(df_contracts['ACTUAL_REBATE'], errors='coerce')
                df_contracts['ACTUAL_REBATE'] = df_contracts['ACTUAL_REBATE'].fillna(df_contracts['FIXED_REBATE_AMOUNT'])
            else:
                df_contracts['ACTUAL_REBATE'] = pd.to_numeric(df_contracts['ACTUAL_REBATE'], errors='coerce')
            
            # AGGREGATE
            df_task1 = df_contracts.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'CHANNEL'])['ACTUAL_REBATE'].mean().reset_index()
            
            joblib.dump(df_task1, os.path.join(OUTPUT_DIR, "task1_discount_output.pkl"))
            print(f"✅ AGGREGATED CONTRACTS SAVED: {len(df_task1):,} unique strategies")
            
        except Exception as e:
            print(f"Error processing Contracts: {e}")
    else:
        print("❌ Contracts file not found.")

    # ---------------------------------------------------------
    # 2. PROCESS CLAIMS
    # ---------------------------------------------------------
    claim_path = os.path.join(DATA_DIR, "Contract_Claim_Dump-Jan25-Nov25_IDS_OU.xlsx")
    
    if os.path.exists(claim_path):
        print(f"\nReading Claims from: {claim_path}")
        try:
            # Load ALL sheets
            all_sheets = pd.read_excel(claim_path, sheet_name=None)
            df_claims = pd.concat(all_sheets.values(), ignore_index=True)
            
            print(f"📊 RAW CLAIMS LOADED: {len(df_claims):,} rows")
            
            df_claims.rename(columns={
                'CUSTOMER_NAME': 'CUSTOMER',
                'CUSTOMER_CHANNEL': 'CHANNEL',
                'PRODUCT_GROUP': 'PRODUCT_GROUP',
                'AMOUNT': 'SALES_VALUE'
            }, inplace=True)
            
            # CLEANING
            df_claims['CUSTOMER'] = df_claims['CUSTOMER'].apply(clean_customer_name)
            df_claims['PRODUCT_GROUP'] = df_claims['PRODUCT_GROUP'].astype(str).str.strip().str.title()
            df_claims['CHANNEL'] = df_claims['CHANNEL'].astype(str).str.strip().str.upper()
            
            # Ensure SALES_VALUE is numeric before aggregation
            df_claims['SALES_VALUE'] = pd.to_numeric(df_claims['SALES_VALUE'], errors='coerce')
            
            # AGGREGATE
            df_task2 = df_claims.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'CHANNEL'])['SALES_VALUE'].sum().reset_index()
            
            joblib.dump(df_task2, os.path.join(OUTPUT_DIR, "task2_sales_output.pkl"))
            print(f"✅ AGGREGATED CLAIMS SAVED: {len(df_task2):,} unique sales summaries")
            
        except Exception as e:
            print(f"Error processing Claims: {e}")
    else:
        print("❌ Claims file not found.")

if __name__ == "__main__":
    prepare_task3_data_v3()