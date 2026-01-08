import pandas as pd

base_path = r'd:\BlueRose\iffco_discount_model\data\csv'

tda_plan = pd.read_csv(base_path + r'\TDA_FY 24_25_SIP_PLAN_DIMNS1.csv')
tda_claim = pd.read_csv(base_path + r'\TDA_FY 24_25_SIP_CLAIM_DIMNS1.csv')

print('TDA_PLAN composite key analysis:')
print(f'Total rows: {len(tda_plan):,}')
print(f'Unique TDA_PLAN_ID: {tda_plan["TDA_PLAN_ID"].nunique():,}')
print(f'Unique TDA_PLAN_ROW_ID: {tda_plan["TDA_PLAN_ROW_ID"].nunique():,}')
composite_plan_keys = tda_plan.groupby(['TDA_PLAN_ID', 'TDA_PLAN_ROW_ID']).ngroups
print(f'Unique (TDA_PLAN_ID, TDA_PLAN_ROW_ID): {composite_plan_keys:,}')
print(f'\nRows per composite key:')
key_counts = tda_plan.groupby(['TDA_PLAN_ID', 'TDA_PLAN_ROW_ID']).size()
print(f'  Min: {key_counts.min()}')
print(f'  Max: {key_counts.max()}')
print(f'  Mean: {key_counts.mean():.1f}')
print(f'  Median: {key_counts.median():.0f}')

print('\n\nTDA_CLAIM composite key analysis:')
print(f'Total rows: {len(tda_claim):,}')
print(f'Unique TDA_PLAN_ID: {tda_claim["TDA_PLAN_ID"].nunique():,}')
print(f'Unique TDA_PLAN_ROW_ID: {tda_claim["TDA_PLAN_ROW_ID"].nunique():,}')
composite_claim_keys = tda_claim.groupby(['TDA_PLAN_ID', 'TDA_PLAN_ROW_ID']).ngroups
print(f'Unique (TDA_PLAN_ID, TDA_PLAN_ROW_ID): {composite_claim_keys:,}')
print(f'\nRows per composite key:')
key_counts_claim = tda_claim.groupby(['TDA_PLAN_ID', 'TDA_PLAN_ROW_ID']).size()
print(f'  Min: {key_counts_claim.min()}')
print(f'  Max: {key_counts_claim.max()}')
print(f'  Mean: {key_counts_claim.mean():.1f}')
print(f'  Median: {key_counts_claim.median():.0f}')

# Check TACTIC_NUMBER as unique key
print('\n\nAlternative key: TACTIC_NUMBER')
print(f'CLAIM unique TACTIC_NUMBER: {tda_claim["TACTIC_NUMBER"].nunique():,}')
print(f'PLAN unique TACTIC_NUMBER: {tda_plan["TACTIC_NUMBER"].nunique():,}')

# Check overlap on TACTIC_NUMBER
claim_tactics = set(tda_claim['TACTIC_NUMBER'].unique())
plan_tactics = set(tda_plan['TACTIC_NUMBER'].unique())
overlap_tactics = claim_tactics.intersection(plan_tactics)

print(f'\nTACTIC_NUMBER overlap:')
print(f'  In both CLAIM and PLAN: {len(overlap_tactics):,}')
print(f'  Only in CLAIM: {len(claim_tactics - plan_tactics):,}')
print(f'  Only in PLAN: {len(plan_tactics - claim_tactics):,}')

# Try joining on TACTIC_NUMBER
merged_tactic = tda_claim.merge(tda_plan, on='TACTIC_NUMBER', how='left', suffixes=('_claim', '_plan'))
print(f'\nJoin on TACTIC_NUMBER:')
print(f'  Result rows: {len(merged_tactic):,}')
print(f'  With TACTIC_TYPE: {merged_tactic["TACTIC_TYPE"].notna().sum():,}')
print(f'  Expansion factor: {len(merged_tactic) / len(tda_claim):.1f}x')

# Check CLAIM_ID uniqueness
print(f'\n\nCLAIM_ID analysis:')
print(f'  Total claims: {len(tda_claim):,}')
print(f'  Unique CLAIM_ID: {tda_claim["CLAIM_ID"].nunique():,}')
print(f'  Is CLAIM_ID unique? {tda_claim["CLAIM_ID"].nunique() == len(tda_claim)}')

print('\n\nRecommendation: Use TACTIC_NUMBER as join key (better cardinality)')
