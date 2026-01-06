import pandas as pd
from pathlib import Path

DATA_DIR = Path(r"d:\BlueRose\iffco_discount_model\data")

INPUTS = {
    "tactic": DATA_DIR / "TACTIC_21_22.xlsx",
    "tda": DATA_DIR / "TDA_FY 21_22.xlsx",
}

OUTPUT_DIR = DATA_DIR / "cleaned"
OUTPUT_DIR.mkdir(exist_ok=True)

DICT_PATH = DATA_DIR / "data_dictionary_all_sheets.csv"

def load_all_sheets(path: Path) -> dict[str, pd.DataFrame]:
    return pd.read_excel(path, sheet_name=None)

def summarize(df: pd.DataFrame, name: str):
    print(f"\n=== {name} ===")
    print(f"rows: {len(df)}, cols: {df.shape[1]}")
    print("dtypes:\n", df.dtypes)
    print("null %:\n", (df.isnull().mean() * 100).round(2))

def make_dictionary(df: pd.DataFrame, table: str, sheet: str) -> pd.DataFrame:
    return pd.DataFrame({
        "table": table,
        "sheet": sheet,
        "column": df.columns,
        "dtype": df.dtypes.astype(str),
        "null_pct": (df.isnull().mean() * 100).round(2).values
    })

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in df.select_dtypes(include="object").columns:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().sum() / len(df) > 0.5:
            df[col] = coerced
    df = df.drop_duplicates()
    return df

def main():
    dict_frames = []
    for table, path in INPUTS.items():
        sheets = load_all_sheets(path)
        for sheet_name, df in sheets.items():
            tag = f"{table}__{sheet_name}"
            summarize(df, tag)
            dict_frames.append(make_dictionary(df, table, sheet_name))
            cleaned = clean(df)
            summarize(cleaned, tag + "_clean")
            out_path = OUTPUT_DIR / f"{table}__{sheet_name}_clean.csv"
            cleaned.to_csv(out_path, index=False)
            print("saved:", out_path)
    data_dict = pd.concat(dict_frames, ignore_index=True)
    data_dict.to_csv(DICT_PATH, index=False)
    print("\nData dictionary saved:", DICT_PATH)

if __name__ == "__main__":
    main()