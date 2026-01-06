"""
Explore SIP Dataset for TDA (Trade Display Allowance) Analysis
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/csv")

print("="*80)
print("SIP DATASET EXPLORATION FOR TDA ANALYSIS")
print("="*80)

# List all SIP files
sip_files = list(DATA_DIR.glob("SIP_*.csv"))
print(f"\nFound {len(sip_files)} SIP files:")
for f in sorted(sip_files)[:10]:
    print(f"  - {f.name}")

# Explore Claims
print("\n" + "="*80)
print("[1] SIP_CLAIM_DIMNS Files")
print("="*80)

claim_files = sorted(list(DATA_DIR.glob("SIP_CLAIM_DIMNS*.csv")))
if claim_files:
    df_claims = pd.read_csv(claim_files[0], low_memory=False)
    print(f"\nFile: {claim_files[0].name}")
    print(f"Shape: {df_claims.shape}")
    print(f"\nColumns: {df_claims.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df_claims.head(3).to_string())
    
    # Check for TDA-related columns
    print(f"\n\nData types:")
    print(df_claims.dtypes)

# Explore Plans
print("\n" + "="*80)
print("[2] SIP_PLAN_DIMNS Files")
print("="*80)

plan_files = sorted(list(DATA_DIR.glob("SIP_PLAN_DIMNS*.csv")))
if plan_files:
    df_plans = pd.read_csv(plan_files[0], low_memory=False)
    print(f"\nFile: {plan_files[0].name}")
    print(f"Shape: {df_plans.shape}")
    print(f"\nColumns: {df_plans.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df_plans.head(3).to_string())

# Explore SKUs
print("\n" + "="*80)
print("[3] SIP_SKU_DIMNS Files")
print("="*80)

sku_files = sorted(list(DATA_DIR.glob("SIP_SKU_DIMNS*.csv")))
if sku_files:
    df_skus = pd.read_csv(sku_files[0], low_memory=False)
    print(f"\nFile: {sku_files[0].name}")
    print(f"Shape: {df_skus.shape}")
    print(f"\nColumns: {df_skus.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df_skus.head(3).to_string())

# Summary
print("\n" + "="*80)
print("SUMMARY - SIP FILES OVERVIEW")
print("="*80)
print(f"Claim files: {len(claim_files)}")
print(f"Plan files: {len(plan_files)}")
print(f"SKU files: {len(sku_files)}")
