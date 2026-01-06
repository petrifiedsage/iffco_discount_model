import pandas as pd
from pathlib import Path

csv_file = Path('data/csv/Contract_Claim_Dump-Jan22-Dec22-UAE_Apr 2022.csv')
df = pd.read_csv(csv_file, low_memory=False)

print('=== TACTICS Column Analysis ===\n')
print('Unique TACTICS values:')
print(df['TACTICS'].value_counts().head(20))
print(f'\nTotal unique TACTICS: {df["TACTICS"].nunique()}')

print('\n=== Check for TDA in TACTICS ===\n')
tda_rows = df[df['TACTICS'].str.contains('TDA', case=False, na=False)]
print(f'Rows containing "TDA": {len(tda_rows)}')
if len(tda_rows) > 0:
    print('\nSample TDA tactics:')
    print(tda_rows['TACTICS'].unique()[:10])

print('\n=== All TACTICS (sample) ===')
tactics_list = df['TACTICS'].dropna().unique()
for tactic in tactics_list[:20]:
    print(f"  - {tactic}")
