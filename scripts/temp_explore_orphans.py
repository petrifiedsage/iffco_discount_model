"""
Explore orphan TACTIC_PLAN records in claims that don't match plans
"""
from pathlib import Path
import pandas as pd
from collections import Counter

DATA_DIR = Path('data/csv')
CLAIM_GLOBS = [
    "TACTIC_*_TACTIC_CM_CLAIMS_DIMNS*.csv",
    "TACTIC_*_CLAIMS_DIMNS*.csv",
]
PLAN_GLOBS = [
    "TACTIC_*_TACTIC_PLAN_DIMNS*.csv",
]

# Load all plans with TACTIC_PLAN
plan_frames = []
for pattern in PLAN_GLOBS:
    for fpath in sorted(DATA_DIR.glob(pattern)):
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        plan_frames.append(df)

all_plans = pd.concat(plan_frames, ignore_index=True) if plan_frames else pd.DataFrame()
plans_tactic_plans = set(all_plans['TACTIC_PLAN'].dropna().unique())

# Load all claims
claims_frames = []
for pattern in CLAIM_GLOBS:
    for fpath in sorted(DATA_DIR.glob(pattern)):
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        if 'CLAIM_AMOUNT' in df.columns and 'CLAIM_STATUS' in df.columns and 'TACTIC_PLAN' in df.columns:
            df_sub = df[['TACTIC_PLAN', 'CLAIM_AMOUNT', 'CLAIM_STATUS']].copy()
            if 'TACTIC_NAME' in df.columns:
                df_sub['TACTIC_NAME'] = df[['TACTIC_NAME']]
            if 'TACTIC_NUMBER' in df.columns:
                df_sub['TACTIC_NUMBER'] = df[['TACTIC_NUMBER']]
            df_sub['CLAIM_AMOUNT'] = pd.to_numeric(df_sub['CLAIM_AMOUNT'], errors='coerce')
            claims_frames.append(df_sub)

all_claims = pd.concat(claims_frames, ignore_index=True) if claims_frames else pd.DataFrame()
all_claims = all_claims.dropna(subset=['TACTIC_PLAN', 'CLAIM_AMOUNT'])

print("=" * 80)
print("ORPHAN CLAIMS ANALYSIS (TACTIC_PLAN in claims but NOT in plans)")
print("=" * 80)

claims_tactic_plans = set(all_claims['TACTIC_PLAN'].dropna().unique())
orphan_plans = claims_tactic_plans - plans_tactic_plans

print(f"\nOrphan TACTIC_PLAN count: {len(orphan_plans)}")
print(f"Claims rows with orphan TACTIC_PLAN: {all_claims[all_claims['TACTIC_PLAN'].isin(orphan_plans)].shape[0]:,}")

# Sample orphan plans
print("\nSample orphan TACTIC_PLAN values:")
for tp in list(orphan_plans)[:10]:
    claims_for_tp = all_claims[all_claims['TACTIC_PLAN'] == tp]
    count = len(claims_for_tp)
    amount = claims_for_tp['CLAIM_AMOUNT'].sum()
    tactic_name = claims_for_tp['TACTIC_NAME'].iloc[0] if 'TACTIC_NAME' in claims_for_tp.columns and len(claims_for_tp) > 0 else "N/A"
    print(f"  {tp}: {count} claims, ${amount:,.0f}, TACTIC_NAME={tactic_name}")

# Try to see if TACTIC_NAME contains useful info
if 'TACTIC_NAME' in all_claims.columns:
    orphan_claims = all_claims[all_claims['TACTIC_PLAN'].isin(orphan_plans)]
    print(f"\nUnique TACTIC_NAME values in orphan claims:")
    tactic_names = orphan_claims['TACTIC_NAME'].value_counts()
    for name, count in tactic_names.head(20).items():
        print(f"  {name}: {count}")

# Check if any orphan plans are in newer plan files
print("\n" + "=" * 80)
print("Checking which plan files contain orphan TACTIC_PLAN...")
for pattern in PLAN_GLOBS:
    for fpath in sorted(DATA_DIR.glob(pattern)):
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        orphan_in_file = set(df['TACTIC_PLAN'].dropna().unique()) & orphan_plans
        if orphan_in_file:
            print(f"  {fpath.name}: {len(orphan_in_file)} orphan plans found")
            # Check what columns are in this file
            product_cols = [c for c in df.columns if 'PRODUCT' in c.upper()]
            tactic_cols = [c for c in df.columns if 'TACTIC' in c.upper()]
            print(f"    Product cols: {product_cols}")
            print(f"    Tactic cols: {tactic_cols}")
            
            # Sample a few records
            sample_orphan = df[df['TACTIC_PLAN'].isin(list(orphan_in_file)[:3])]
            if len(sample_orphan) > 0:
                print(f"    Sample data:")
                if 'PRODUCT_GROUP' in sample_orphan.columns:
                    for _, row in sample_orphan.head(3).iterrows():
                        print(f"      TACTIC_PLAN={row['TACTIC_PLAN']}, PRODUCT_GROUP={row.get('PRODUCT_GROUP', 'N/A')}, TACTIC_TYPE={row.get('TACTIC_TYPE', 'N/A')}")

print("\n" + "=" * 80)
print("COVERAGE BY FILE YEAR")
print("=" * 80)

# Load each plan file and see which TACTIC_PLAN it covers
for pattern in PLAN_GLOBS:
    for fpath in sorted(DATA_DIR.glob(pattern)):
        df = pd.read_csv(fpath, low_memory=False)
        df.columns = df.columns.str.strip()
        unique_tp = df['TACTIC_PLAN'].nunique()
        orphan_count = len(set(df['TACTIC_PLAN'].dropna().unique()) & orphan_plans)
        print(f"{fpath.name}: {unique_tp} unique TACTIC_PLAN, {orphan_count} are orphans")
