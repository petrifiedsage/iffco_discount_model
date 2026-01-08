from pathlib import Path
import pandas as pd

f = list(Path('data/csv').glob('TACTIC_*_CM_CLAIMS*'))[0]
df = pd.read_csv(f, nrows=100, low_memory=False)
df.columns = df.columns.str.strip()
print("Columns in claims file:")
print(df.columns.tolist())
