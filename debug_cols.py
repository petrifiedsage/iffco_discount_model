import pandas as pd
from pathlib import Path

csv_file = Path('data/csv/Contract_Claim_Dump-Jan22-Dec22-UAE_Apr 2022.csv')
df = pd.read_csv(csv_file, low_memory=False)
df.columns = df.columns.str.strip()

approved = df[df['STATUS'] == 'Approved'].copy()
print(f'Approved rows: {len(approved)}')

# Check nulls
print(f'MONTH null: {approved["MONTH"].isna().sum()}')
print(f'AMOUNT null: {approved["AMOUNT"].isna().sum()}')
print(f'PRODUCT_GROUP null: {approved["PRODUCT_GROUP"].isna().sum()}')

# Check types
print(f'MONTH type: {approved["MONTH"].dtype}')
print(f'AMOUNT type: {approved["AMOUNT"].dtype}')

# Sample values
print(f'Sample MONTH values: {approved["MONTH"].head(10).tolist()}')
print(f'Sample AMOUNT values: {approved["AMOUNT"].head(10).tolist()}')

# Try numeric conversion
approved['MONTH_NUM'] = pd.to_numeric(approved['MONTH'], errors='coerce')
approved['AMOUNT_NUM'] = pd.to_numeric(approved['AMOUNT'], errors='coerce')
print(f'MONTH coerce nulls: {approved["MONTH_NUM"].isna().sum()}')
print(f'AMOUNT coerce nulls: {approved["AMOUNT_NUM"].isna().sum()}')
