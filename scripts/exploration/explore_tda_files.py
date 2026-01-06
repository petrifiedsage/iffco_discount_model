"""
Explore TDA Dataset (Trade Display Allowance) - Dedicated Files
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

print("="*80)
print("TDA DATASET EXPLORATION")
print("="*80)

# List all TDA files
tda_files = sorted(list(DATA_DIR.glob("TDA*.xls*")))
print(f"\nFound {len(tda_files)} TDA files:")
for f in tda_files:
    print(f"  - {f.name}")

# Explore each TDA file
for tda_file in tda_files:
    print("\n" + "="*80)
    print(f"File: {tda_file.name}")
    print("="*80)
    
    try:
        df = pd.read_excel(tda_file, sheet_name=0)
        print(f"Shape: {df.shape}")
        print(f"\nColumns ({len(df.columns)}):")
        for col in df.columns:
            print(f"  - {col}")
        
        print(f"\nFirst 5 rows:")
        print(df.head(5).to_string())
        
        print(f"\n\nData types:")
        print(df.dtypes)
        
        # Summary statistics
        print(f"\n\nUnique values (key columns):")
        for col in df.columns[:8]:  # First 8 columns
            if df[col].dtype == 'object':
                print(f"  {col}: {df[col].nunique()} unique values")
                print(f"    Top values: {df[col].value_counts().head(3).to_dict()}")
            else:
                print(f"  {col}: {df[col].nunique()} unique values (numeric)")
        
    except Exception as e:
        print(f"Error reading file: {e}")
        # Try different sheet names
        try:
            xl_file = pd.ExcelFile(tda_file)
            print(f"\nSheet names: {xl_file.sheet_names}")
        except:
            pass

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"Total TDA files found: {len(tda_files)}")
print("These appear to be dedicated TDA datasets separate from Contract_Claim_Dump")
