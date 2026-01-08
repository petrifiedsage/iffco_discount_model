import pandas as pd
import os

base_path = r'd:\BlueRose\iffco_discount_model\data\csv'

print('='*80)
print('JOIN RELATIONSHIP ANALYSIS')
print('='*80)

# 1. TACTIC Join Analysis
print('\n1. TACTIC FILES JOIN RELATIONSHIP')
print('-'*80)

tactic_plan = pd.read_csv(os.path.join(base_path, 'TACTIC_24_25_TACTIC_PLAN_DIMNS.csv'))
tactic_claims = pd.read_csv(os.path.join(base_path, 'TACTIC_24_25_TACTIC_CM_CLAIMS_DIMNS.csv'))

print(f'\nBefore Join:')
print(f'  TACTIC_PLAN rows: {len(tactic_plan):,}')
print(f'  TACTIC_CLAIMS rows: {len(tactic_claims):,}')
print(f'  Unique TACTIC_PLAN IDs in PLAN: {tactic_plan["TACTIC_PLAN"].nunique():,}')
print(f'  Unique TACTIC_PLAN IDs in CLAIMS: {tactic_claims["TACTIC_PLAN"].nunique():,}')

# Check overlap
plan_ids = set(tactic_plan['TACTIC_PLAN'].unique())
claim_ids = set(tactic_claims['TACTIC_PLAN'].unique())
overlap = plan_ids.intersection(claim_ids)

print(f'\nID Overlap:')
print(f'  TACTIC_PLAN IDs only in PLAN: {len(plan_ids - claim_ids):,}')
print(f'  TACTIC_PLAN IDs only in CLAIMS: {len(claim_ids - plan_ids):,}')
print(f'  TACTIC_PLAN IDs in both: {len(overlap):,}')
print(f'  Overlap %: {len(overlap)/len(plan_ids)*100:.1f}% of PLAN records')

# Join
merged = tactic_claims.merge(tactic_plan, on='TACTIC_PLAN', how='left', suffixes=('_claim', '_plan'))
merged_inner = tactic_claims.merge(tactic_plan, on='TACTIC_PLAN', how='inner', suffixes=('_claim', '_plan'))

print(f'\nAfter Join (left join on TACTIC_PLAN):')
print(f'  Merged rows: {len(merged):,}')
print(f'  Rows with TACTIC_TYPE: {merged["TACTIC_TYPE"].notna().sum():,} ({merged["TACTIC_TYPE"].notna().sum()/len(merged)*100:.1f}%)')
print(f'  Rows without TACTIC_TYPE: {merged["TACTIC_TYPE"].isna().sum():,}')

print(f'\nJoin Cardinality:')
print(f'  Avg PLAN rows per CLAIM: {len(merged_inner) / len(tactic_claims):.1f}')
print(f'  Explanation: {"1:Many (one claim → many plan rows)" if len(merged_inner) / len(tactic_claims) > 1.5 else "1:1"}')

# Check duplicates
claim_counts = merged.groupby('CLAIM_ID').size()
print(f'\nDuplicate Analysis:')
print(f'  Claims appearing once: {(claim_counts == 1).sum():,}')
print(f'  Claims appearing multiple times: {(claim_counts > 1).sum():,}')
if (claim_counts > 1).sum() > 0:
    print(f'  Max times a claim appears: {claim_counts.max()}')
    print(f'  Median times claims appear: {claim_counts.median():.0f}')

# 2. TDA Join Analysis
print('\n\n2. TDA SIP FILES JOIN RELATIONSHIP')
print('-'*80)

tda_plan = pd.read_csv(os.path.join(base_path, 'TDA_FY 24_25_SIP_PLAN_DIMNS1.csv'))
tda_claim = pd.read_csv(os.path.join(base_path, 'TDA_FY 24_25_SIP_CLAIM_DIMNS1.csv'))

print(f'\nBefore Join:')
print(f'  TDA_PLAN rows: {len(tda_plan):,}')
print(f'  TDA_CLAIM rows: {len(tda_claim):,}')
print(f'  Unique TDA_PLAN_ID in PLAN: {tda_plan["TDA_PLAN_ID"].nunique():,}')
print(f'  Unique TDA_PLAN_ID in CLAIM: {tda_claim["TDA_PLAN_ID"].nunique():,}')

# Check TDA_PLAN_ID + TDA_PLAN_ROW_ID composite key
tda_plan['composite_key'] = tda_plan['TDA_PLAN_ID'] + '_' + tda_plan['TDA_PLAN_ROW_ID']
tda_claim['composite_key'] = tda_claim['TDA_PLAN_ID'] + '_' + tda_claim['TDA_PLAN_ROW_ID']

print(f'  Unique composite keys in PLAN: {tda_plan["composite_key"].nunique():,}')
print(f'  Unique composite keys in CLAIM: {tda_claim["composite_key"].nunique():,}')

# Check overlap
plan_keys = set(tda_plan['composite_key'].unique())
claim_keys = set(tda_claim['composite_key'].unique())
overlap_tda = plan_keys.intersection(claim_keys)

print(f'\nKey Overlap:')
print(f'  Keys only in PLAN: {len(plan_keys - claim_keys):,}')
print(f'  Keys only in CLAIM: {len(claim_keys - plan_keys):,}')
print(f'  Keys in both: {len(overlap_tda):,}')
print(f'  Overlap %: {len(overlap_tda)/len(plan_keys)*100:.1f}% of PLAN records')

# Join
tda_merged = tda_claim.merge(
    tda_plan, 
    on=['TDA_PLAN_ID', 'TDA_PLAN_ROW_ID'], 
    how='left', 
    suffixes=('_claim', '_plan')
)

print(f'\nAfter Join (left join on TDA_PLAN_ID + TDA_PLAN_ROW_ID):')
print(f'  Merged rows: {len(tda_merged):,}')
print(f'  Rows with TACTIC_TYPE: {tda_merged["TACTIC_TYPE"].notna().sum():,} ({tda_merged["TACTIC_TYPE"].notna().sum()/len(tda_merged)*100:.1f}%)')

# 3. Recommendations
print('\n\n3. JOIN STRATEGY RECOMMENDATIONS')
print('-'*80)

print('\nTACTIC Files:')
print('  ✓ Join on: TACTIC_PLAN')
print('  ✓ Join type: LEFT JOIN (claims to plan)')
print('  ✓ Expected result: Multiple plan rows per claim (product variations)')
print('  ⚠ Action: Group by CLAIM_ID after join to avoid duplicates')
print('  ⚠ Alternative: Use first() aggregation or filter to single plan row')

print('\nTDA SIP Files:')
print('  ✓ Join on: TDA_PLAN_ID + TDA_PLAN_ROW_ID (composite key)')
print('  ✓ Join type: LEFT JOIN (claims to plan)')
print('  ✓ Expected result: Better 1:1 relationship')
print('  ✓ High match rate expected')

# 4. Sample joined data
print('\n\n4. SAMPLE JOINED DATA')
print('-'*80)

print('\nTACTIC Sample (first claim with all features):')
sample = merged[merged['TACTIC_TYPE'].notna()].iloc[0]
print(f'  CLAIM_ID: {sample["CLAIM_ID"]}')
print(f'  TACTIC_TYPE: {sample["TACTIC_TYPE"]}')
print(f'  PRODUCT_GROUP: {sample["PRODUCT_GROUP"]}')
print(f'  BRAND: {sample["BRAND"]}')
print(f'  CLAIM_STATUS: {sample["CLAIM_STATUS"]}')
print(f'  CLAIM_AMOUNT: {sample["CLAIM_AMOUNT"]}')
print(f'  MONTH: {sample["MONTH"]}')
print(f'  PERIOD: {sample["PERIOD"]}')

print('\n' + '='*80)
print('SUMMARY')
print('='*80)

print('\n1. TACTIC files have 1:Many relationship due to product variations')
print('2. TDA files should be joined on composite key for better 1:1 mapping')
print('3. Both sources viable for training with proper data handling')
print('4. Recommended: Aggregate plan features per claim or use dedupe strategy')

print('\n' + '='*80)
