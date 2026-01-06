import pandas as pd
import joblib
import os
import numpy as np

# --- CONFIGURATION ---
# Update these paths if they differ on your machine
DATA_DIR = r"D:/BlueRose/iffco_discount_model/data"
OUTPUT_DIR = r"D:/BlueRose/iffco_discount_model/outputs"

def prepare_task3_data():
    print("--- Preparing Data for Task 3 (Discount Efficiency) ---")
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # ---------------------------------------------------------
    # 1. PROCESS TASK 1 DATA (Discounts from Contract Data)
    # ---------------------------------------------------------
    contract_path = os.path.join(DATA_DIR, "Contract_Data_with_Product Group_IDS_OU_2025.xls")
    
    # Fallback if xlsx
    if not os.path.exists(contract_path):
        contract_path = contract_path.replace(".xls", ".xlsx")

    if os.path.exists(contract_path):
        print(f"Loading Contract Data: {contract_path}")
        try:
            # Use pyexcel for .xls to avoid xlrd version conflicts with pandas 2.2+
            if contract_path.endswith('.xls'):
                import pyexcel as pe
                sheet = pe.get_sheet(file_name=contract_path)
                data = sheet.get_array()
                df_contracts = pd.DataFrame(data[1:], columns=[str(h).strip() for h in data[0]])
            else:
                df_contracts = pd.read_excel(contract_path, engine='openpyxl')

            # CLEANING
            # Standardize names to match Task 2
            df_contracts['CUSTOMER'] = df_contracts['CUSTOMER'].astype(str).str.strip().str.title()
            df_contracts['PRODUCT_GROUP'] = df_contracts['PRODUCT_GROUP'].astype(str).str.strip().str.title()
            
            # Handle Channel (create if missing or standardise)
            if 'CHANNEL' in df_contracts.columns:
                df_contracts['CHANNEL'] = df_contracts['CHANNEL'].astype(str).str.strip().str.upper()
            else:
                df_contracts['CHANNEL'] = "UNKNOWN"

            # Combine Actual and Fixed Rebates
            if 'FIXED_REBATE_AMOUNT' in df_contracts.columns:
                df_contracts['FIXED_REBATE_AMOUNT'] = pd.to_numeric(df_contracts['FIXED_REBATE_AMOUNT'], errors='coerce')
                df_contracts['ACTUAL_REBATE'] = df_contracts['ACTUAL_REBATE'].fillna(df_contracts['FIXED_REBATE_AMOUNT'])
            # Ensure ACTUAL_REBATE is numeric
            df_contracts['ACTUAL_REBATE'] = pd.to_numeric(df_contracts['ACTUAL_REBATE'], errors='coerce')
            
            # Select relevant columns for the app
            cols = ['CUSTOMER', 'PRODUCT_GROUP', 'CHANNEL', 'ACTUAL_REBATE']
            # If you have 'RECOMMENDED_REBATE' from a previous run, include it. 
            # Otherwise we'll simulate it or use Actuals for now.
            if 'RECOMMENDED_REBATE' in df_contracts.columns:
                cols.append('RECOMMENDED_REBATE')
            
            df_task1 = df_contracts[cols].copy()
            
            # Save
            out_path1 = os.path.join(OUTPUT_DIR, "task1_discount_output.pkl")
            joblib.dump(df_task1, out_path1)
            print(f"✅ Saved: {out_path1}")
            
        except Exception as e:
            print(f"Error processing Task 1 data: {e}")
    else:
        print(f"❌ Error: Contract file not found at {contract_path}")

    # ---------------------------------------------------------
    # 2. PROCESS TASK 2 DATA (Sales from Claim Dump)
    # ---------------------------------------------------------
    claim_path = os.path.join(DATA_DIR, "Contract_Claim_Dump-Jan25-Nov25_IDS_OU.xlsx")
    
    if os.path.exists(claim_path):
        print(f"Loading Claim Data: {claim_path}")
        try:
            # Load all sheets
            all_sheets = pd.read_excel(claim_path, sheet_name=None)
            df_claims = pd.concat(all_sheets.values(), ignore_index=True)
            
            # RENAME columns to match Task 1
            # Claim Dump has 'CUSTOMER_NAME', 'CUSTOMER_CHANNEL', 'PRODUCT_GROUP', 'AMOUNT'
            df_claims.rename(columns={
                'CUSTOMER_NAME': 'CUSTOMER',
                'CUSTOMER_CHANNEL': 'CHANNEL',
                'PRODUCT_GROUP': 'PRODUCT_GROUP',
                'AMOUNT': 'SALES_VALUE'
            }, inplace=True)
            
            # CLEANING
            # Parse dates robustly (handles mixed formats from Excel)
            if 'CLAIM_DATE' in df_claims.columns:
                df_claims['CLAIM_DATE'] = pd.to_datetime(df_claims['CLAIM_DATE'], errors='coerce', dayfirst=True, format='mixed')
            
            df_claims['CUSTOMER'] = df_claims['CUSTOMER'].astype(str).str.strip().str.title()
            df_claims['PRODUCT_GROUP'] = df_claims['PRODUCT_GROUP'].astype(str).str.strip().str.title()
            df_claims['CHANNEL'] = df_claims['CHANNEL'].astype(str).str.strip().str.upper()
            
            # Ensure SALES_VALUE is numeric
            df_claims['SALES_VALUE'] = pd.to_numeric(df_claims['SALES_VALUE'], errors='coerce')
            
            # AGGREGATE Sales by Customer-Product-Channel
            df_task2 = df_claims.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'CHANNEL'])['SALES_VALUE'].sum().reset_index()
            
            # Save
            out_path2 = os.path.join(OUTPUT_DIR, "task2_sales_output.pkl")
            joblib.dump(df_task2, out_path2)
            print(f"✅ Saved: {out_path2}")
            
        except Exception as e:
            print(f"Error processing Task 2 data: {e}")
    else:
        print(f"❌ Error: Claim file not found at {claim_path}")

if __name__ == "__main__":
    prepare_task3_data()