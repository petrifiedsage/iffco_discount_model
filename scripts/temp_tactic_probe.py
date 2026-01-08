from pathlib import Path
import pandas as pd
from collections import Counter

DATA_DIR = Path('data/csv')
plan_files = sorted(DATA_DIR.glob('TACTIC_*_PLAN_DIMNS*.csv'))
claim_files = sorted(list(DATA_DIR.glob('TACTIC_*_CM_CLAIMS_DIMNS*.csv')) + list(DATA_DIR.glob('TACTIC_*_CLAIMS_DIMNS*.csv')))
print(f"plan files: {len(plan_files)}")
print(f"claim files: {len(claim_files)}")

status_counts = {}
total_rows = 0
missing_plan = 0
for f in claim_files:
    df = pd.read_csv(f, usecols=lambda c: c in {'CLAIM_STATUS','TACTIC_PLAN','CLAIM_AMOUNT'}, low_memory=False)
    total_rows += len(df)
    missing_plan += df['TACTIC_PLAN'].isna().sum() if 'TACTIC_PLAN' in df else 0
    if 'CLAIM_STATUS' in df:
        vals = df['CLAIM_STATUS'].astype(str).str.strip().str.upper()
        status_counts[f.name] = vals.value_counts().head(10).to_dict()

agg = Counter()
for d in status_counts.values():
    agg.update(d)
print(f"total claim rows (raw): {total_rows:,}")
print(f"rows missing TACTIC_PLAN: {missing_plan:,}")
print("top statuses across all claim files:")
for k,v in agg.most_common(15):
    print(f"  {k}: {v:,}")

pg_set = set()
for f in plan_files:
    df = pd.read_csv(f, usecols=lambda c: c in {'PRODUCT_GROUP','TACTIC_PLAN','TACTIC_TYPE'}, low_memory=False)
    pg_set.update(df['PRODUCT_GROUP'].dropna().astype(str).str.strip().unique())
print(f"unique PRODUCT_GROUP in plans: {len(pg_set)}")
