import pandas as pd

csv_file = 'data/csv/Contract_Claim_Dump-Jan22-Dec22-UAE_Apr 2022.csv'
df = pd.read_csv(csv_file, low_memory=False)

print('=== CLAIM_DETAILS Analysis ===\n')
print('Top 20 CLAIM_DETAILS:')
print(df['CLAIM_DETAILS'].value_counts().head(20))

print('\n=== REASON Analysis ===\n')
print('Unique REASON values:')
print(df['REASON'].unique())

print('\n=== Check for Trade Development ===\n')
tda_check = df[df['CLAIM_DETAILS'].str.contains('Trade|TDA|Development', case=False, na=False)]
print(f'Rows with "Trade/TDA/Development": {len(tda_check)}')
if len(tda_check) > 0:
    print('Sample CLAIM_DETAILS:')
    print(tda_check['CLAIM_DETAILS'].unique()[:10])
