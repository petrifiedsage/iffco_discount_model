"""
Diagnostic script to identify product groups with 0 count issues
"""
from pathlib import Path
import pandas as pd

DATA_DIR = Path('data/csv')
CLAIM_GLOBS = [
    "TACTIC_*_TACTIC_CM_CLAIMS_DIMNS*.csv",
    "TACTIC_*_CLAIMS_DIMNS*.csv",
]
PLAN_GLOBS = [
    "TACTIC_*_TACTIC_PLAN_DIMNS*.csv",
]

# Load all claims with TACTIC_PLAN
claims_frames = []
for pattern in CLAIM_GLOBS:
    for fpath in sorted(DATA_DIR.glob(pattern)):
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        if 'TACTIC_PLAN' in df.columns:
            claims_frames.append(df[['TACTIC_PLAN']])

all_claims = pd.concat(claims_frames, ignore_index=True) if claims_frames else pd.DataFrame()
print(f"Total claims rows with TACTIC_PLAN: {len(all_claims):,}")
print(f"Unique TACTIC_PLAN in claims: {all_claims['TACTIC_PLAN'].nunique():,}")

# Load all plans
plan_frames = []
for pattern in PLAN_GLOBS:
    for fpath in sorted(DATA_DIR.glob(pattern)):
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        plan_frames.append(df)

all_plans = pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame()
print(f"Total plans rows: {len(all_plans):,}")
print(f"Unique TACTIC_PLAN in plans: {all_plans['TACTIC_PLAN'].nunique():,}")

# Find TACTIC_PLAN in claims but not in plans
claims_plans = set(all_claims['TACTIC_PLAN'].dropna().unique())
plans_plans = set(all_plans['TACTIC_PLAN'].dropna().unique())

in_claims_not_plans = claims_plans - plans_plans
in_plans_not_claims = plans_plans - claims_plans

print(f"\nTACTIC_PLAN in claims but NOT in plans: {len(in_claims_not_plans)}")
if in_claims_not_plans:
    print(f"  Sample: {list(in_claims_not_plans)[:5]}")

print(f"TACTIC_PLAN in plans but NOT in claims: {len(in_plans_not_claims)}")

# Check product groups in plans that have 0 claims
plans_with_pg = all_plans[all_plans['PRODUCT_GROUP'].notna()].copy()
print(f"\nPlans with PRODUCT_GROUP: {plans_with_pg['PRODUCT_GROUP'].nunique()} unique")
print(f"All PRODUCT_GROUP values in plans:")
for pg in sorted(plans_with_pg['PRODUCT_GROUP'].unique()):
    count_plans = len(plans_with_pg[plans_with_pg['PRODUCT_GROUP'] == pg])
    print(f"  {pg}: {count_plans}")

# Now load actual claims with full merge
claims_full = []
for pattern in CLAIM_GLOBS:
    for fpath in sorted(DATA_DIR.glob(pattern)):
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        if 'CLAIM_AMOUNT' in df.columns and 'CLAIM_STATUS' in df.columns and 'TACTIC_PLAN' in df.columns:
            df_sub = df[['TACTIC_PLAN', 'CLAIM_AMOUNT', 'CLAIM_STATUS']].copy()
            df_sub['CLAIM_AMOUNT'] = pd.to_numeric(df_sub['CLAIM_AMOUNT'], errors='coerce')
            claims_full.append(df_sub)

claims_df = pd.concat(claims_full, ignore_index=True) if claims_full else pd.DataFrame()
claims_df = claims_df.dropna(subset=['TACTIC_PLAN', 'CLAIM_AMOUNT'])
print(f"\nClaims after filtering: {len(claims_df):,} rows")

# Merge claims with plans to see what matches
merged = claims_df.merge(all_plans, on='TACTIC_PLAN', how='left')
print(f"After merge with plans: {len(merged):,} rows")
print(f"Rows with PRODUCT_GROUP after merge: {merged['PRODUCT_GROUP'].notna().sum():,}")

# Find product groups in plans but with 0 claims after merge
pg_in_plans = set(all_plans['PRODUCT_GROUP'].dropna().unique())
pg_after_merge = set(merged['PRODUCT_GROUP'].dropna().unique())
zero_count_pgs = pg_in_plans - pg_after_merge

print(f"\nPRODUCT_GROUP in plans but 0 claims after merge: {len(zero_count_pgs)}")
if zero_count_pgs:
    print("These product groups will show 0 count:")
    for pg in sorted(zero_count_pgs):
        print(f"  {pg}")
