"""
Explore Contract_Claim_Dump to find TDA indicators
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/csv")
claim_files = sorted(list(DATA_DIR.glob("Contract_Claim_Dump*.csv")))[:1]  # First file

print("Checking first Contract_Claim_Dump file for TDA indicators...\n")

for csv_file in claim_files:
    df = pd.read_csv(csv_file, low_memory=False)
    
    print(f"File: {csv_file.name}")
    print(f"Shape: {df.shape}\n")
    
    print("Columns:")
    print(df.columns.tolist())
    print("\n")
    
    # Check for TDA-related columns
    tda_cols = [col for col in df.columns if 'tda' in col.lower() or 'tactic' in col.lower() or 'status' in col.lower()]
    print(f"Potential TDA columns: {tda_cols}\n")
    
    if 'TACTIC_TYPE' in df.columns:
        print("TACTIC_TYPE values:")
        print(df['TACTIC_TYPE'].value_counts())
        print("\n")
    
    if 'STATUS' in df.columns:
        print("STATUS values:")
        print(df['STATUS'].value_counts())
        print("\n")
    
    if 'SUB_STATUS' in df.columns:
        print("SUB_STATUS values:")
        print(df['SUB_STATUS'].value_counts())
        print("\n")
    
    # Show sample data
    print("Sample rows:")
    print(df.head(3).to_string())
