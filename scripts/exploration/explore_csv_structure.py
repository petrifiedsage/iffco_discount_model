"""
Comprehensive Data Exploration of Extracted CSV Files
Understand the actual data structure before building the model
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DATA EXPLORATION: CSV FILE STRUCTURES")
print("="*80)

DATA_DIR = Path("data/csv")

# ============================================================================
# STEP 1: Explore Contract_Claim_Dump files (Main sales data)
# ============================================================================
print("\n[STEP 1] Contract_Claim_Dump Files (2022-2025)")
print("="*80)

claim_dump_files = sorted(list(DATA_DIR.glob("Contract_Claim_Dump*.csv")))[:4]  # Sample first 4
print(f"Found {len(list(DATA_DIR.glob('Contract_Claim_Dump*')))} Contract_Claim_Dump files")

for file in claim_dump_files:
    try:
        df = pd.read_csv(file, nrows=5)
        print(f"\n{file.name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        print(f"  Sample data:")
        print(f"    First row values: {df.iloc[0].to_dict() if len(df) > 0 else 'N/A'}")
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================================
# STEP 2: Explore TACTIC Files
# ============================================================================
print("\n[STEP 2] TACTIC Files (Claims, Plans, SKUs)")
print("="*80)

# Claims
print("\nTACTIC Claims Files:")
tactic_claims = sorted(list(DATA_DIR.glob("TACTIC_*_TACTIC_CM_CLAIMS_DIMNS.csv")))[:2]
for file in tactic_claims:
    try:
        df = pd.read_csv(file, nrows=5)
        print(f"\n{file.name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        if len(df) > 0:
            print(f"  First row: {df.iloc[0].to_dict()}")
    except Exception as e:
        print(f"  Error: {e}")

# SKU files
print("\n\nTACTIC SKU Files:")
tactic_skus = sorted(list(DATA_DIR.glob("TACTIC_*_TACTIC_SKU_DIMNS.csv")))[:2]
for file in tactic_skus:
    try:
        df = pd.read_csv(file, nrows=3)
        print(f"\n{file.name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        if len(df) > 0:
            print(f"  Sample: {df[['SKU_ID', 'PRODUCT']].iloc[0].to_dict() if 'SKU_ID' in df.columns else df.columns.tolist()}")
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================================
# STEP 3: Explore TDA/SIP Files
# ============================================================================
print("\n[STEP 3] TDA SIP Files (Claims, Plans, SKUs)")
print("="*80)

# SIP Claims
print("\nTDA SIP Claims Files:")
sip_claims = sorted(list(DATA_DIR.glob("TDA_FY*_SIP_CLAIM_DIMNS*.csv")))[:2]
for file in sip_claims:
    try:
        df = pd.read_csv(file, nrows=3)
        print(f"\n{file.name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        if len(df) > 0:
            print(f"  Key fields sample:")
            # Find key columns
            for col in df.columns:
                if any(keyword in col.upper() for keyword in ['PRODUCT', 'SKU', 'CUSTOMER', 'CLAIM', 'AMOUNT', 'DATE']):
                    print(f"    {col}: {df[col].iloc[0]}")
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================================
# STEP 4: Check Contract_Data with Product_Group
# ============================================================================
print("\n[STEP 4] Contract Data with Product Group")
print("="*80)

contract_files = list(DATA_DIR.glob("Contract_Data_with_Product*"))
for file in contract_files:
    try:
        df = pd.read_csv(file, nrows=3)
        print(f"\n{file.name}:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        if len(df) > 0:
            print(f"  First row keys:")
            for k, v in df.iloc[0].to_dict().items():
                print(f"    {k}: {v}")
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================================
# STEP 5: Deep Dive - Analyze ONE file completely
# ============================================================================
print("\n[STEP 5] Deep Dive into One Contract_Claim_Dump File")
print("="*80)

sample_file = Path("data/csv/Contract_Claim_Dump-Jan22-Dec22-UAE_Jan 2022.csv")
if sample_file.exists():
    try:
        df = pd.read_csv(sample_file)
        print(f"\n{sample_file.name}:")
        print(f"  Total rows: {len(df):,}")
        print(f"  Columns ({len(df.columns)}): {list(df.columns)}")
        
        print(f"\n  Data types:")
        for col in df.columns:
            print(f"    {col}: {df[col].dtype}")
        
        print(f"\n  Sample rows (first 3):")
        print(df.head(3).to_string())
        
        print(f"\n  Unique values in key columns:")
        for col in df.columns:
            if df[col].dtype == 'object':
                unique_count = df[col].nunique()
                if unique_count <= 20:
                    print(f"    {col} ({unique_count} unique):")
                    for val in df[col].unique()[:10]:
                        print(f"      - {val}")
            else:
                print(f"    {col}: {df[col].nunique()} unique values, range: {df[col].min():.2f} to {df[col].max():.2f}")
    except Exception as e:
        print(f"  Error: {e}")

# ============================================================================
# STEP 6: Check if there's a product mapping table
# ============================================================================
print("\n[STEP 6] Product Mapping/Reference Data")
print("="*80)

# Look for any reference files
all_files = list(DATA_DIR.glob("*.csv"))
reference_patterns = ['contract', 'product', 'sku', 'mapping', 'export']
potential_refs = [f for f in all_files if any(p in f.name.lower() for p in reference_patterns)]

print(f"\nPotential reference files:")
for f in potential_refs[:10]:
    try:
        df = pd.read_csv(f, nrows=5)
        print(f"  {f.name}: {df.shape[0]} rows, columns: {list(df.columns)[:5]}...")
    except:
        pass

# ============================================================================
# STEP 7: Summary Statistics
# ============================================================================
print("\n[STEP 7] SUMMARY STATISTICS")
print("="*80)

# Count file types
file_types = defaultdict(int)
for f in all_files:
    if "Contract_Claim" in f.name:
        file_types["Contract_Claim_Dump"] += 1
    elif "TACTIC_CM_CLAIMS" in f.name:
        file_types["TACTIC_Claims"] += 1
    elif "TACTIC_SKU" in f.name:
        file_types["TACTIC_SKU"] += 1
    elif "TACTIC_PLAN" in f.name:
        file_types["TACTIC_Plan"] += 1
    elif "SIP_CLAIM" in f.name:
        file_types["SIP_Claims"] += 1
    elif "SIP_SKU" in f.name:
        file_types["SIP_SKU"] += 1
    elif "SIP_PLAN" in f.name:
        file_types["SIP_Plan"] += 1
    elif "Contracts_export" in f.name:
        file_types["Contract_Export"] += 1
    else:
        file_types["Other"] += 1

print("\nFile Type Distribution:")
for ftype, count in sorted(file_types.items(), key=lambda x: x[1], reverse=True):
    print(f"  {ftype}: {count} files")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Identify which files contain actual sales/claim transactions")
print("2. Map customer → SKU/Product → Amount → Date")
print("3. Build correct training dataset")
print("4. Extract proper seasonality patterns")
