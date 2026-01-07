"""
Aggregate claim amounts by product group and calendar quarter (Q1-Q4).
Reads Contract_Claim_Dump*.csv files from data/csv, filters approved claims,
parses dates, and writes two CSVs:
- output/quarterly_claims_by_product.csv (all years combined, one row per product group with Q1-Q4 columns)
- output/quarterly_claims_by_product_year.csv (per-year breakdown with Q1-Q4 columns)
"""

from pathlib import Path
import re
import pandas as pd

DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

FILES_GLOB = "Contract_Claim_Dump*.csv"
REQUIRED_COLS = ["PRODUCT_GROUP", "AMOUNT", "STATUS", "CLAIM_DATE", "MONTH", "PERIOD"]


def parse_date(value: object) -> pd.Timestamp:
    """Parse claim date; returns NaT if not parseable."""
    if pd.isna(value):
        return pd.NaT
    text = str(value).strip()
    if not text:
        return pd.NaT
    # Try explicit formats first
    for fmt in ("%d-%m-%y", "%d-%m-%Y", "%d-%b-%y", "%d-%b-%Y", "%Y-%m-%d"):
        try:
            return pd.to_datetime(text, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(text, dayfirst=True)
    except Exception:
        return pd.NaT


def fallback_date(row: pd.Series) -> pd.Timestamp:
    """If DATE is NaT, build a date from PERIOD year + MONTH name."""
    if not pd.isna(row["DATE"]):
        return row["DATE"]
    month_name = str(row.get("MONTH", "")).strip()
    period_text = str(row.get("PERIOD", ""))
    match = re.search(r"(20\d{2})", period_text)
    if month_name and match:
        try:
            # Use the first day of the month for consistent quarter mapping
            return pd.to_datetime(f"1 {month_name} {match.group(1)}", dayfirst=True)
        except Exception:
            return pd.NaT
    return pd.NaT


def load_claims() -> pd.DataFrame:
    csv_files = sorted(DATA_DIR.glob(FILES_GLOB))
    if not csv_files:
        raise FileNotFoundError(f"No files found matching {FILES_GLOB} in {DATA_DIR}")

    frames = []
    for path in csv_files:
        df = pd.read_csv(path, low_memory=False)
        missing = [col for col in REQUIRED_COLS if col not in df.columns]
        if missing:
            print(f"[SKIP] {path.name}: missing columns {missing}")
            continue
        df = df[df["STATUS"] == "Approved"].copy()
        if df.empty:
            print(f"[SKIP] {path.name}: no approved rows")
            continue
        df["PRODUCT_GROUP"] = df["PRODUCT_GROUP"].astype(str).str.strip()
        df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce")
        df["DATE"] = df["CLAIM_DATE"].apply(parse_date)
        df["DATE"] = df.apply(fallback_date, axis=1)
        df = df.dropna(subset=["AMOUNT", "PRODUCT_GROUP", "DATE"])
        df = df[df["AMOUNT"] > 0]
        frames.append(df[["PRODUCT_GROUP", "AMOUNT", "DATE"]])
        print(f"[OK] {path.name}: {len(df):,} rows")

    if not frames:
        raise ValueError("No valid data loaded; all files missing or empty")
    return pd.concat(frames, ignore_index=True)


def aggregate_quarters(df: pd.DataFrame):
    df = df.copy()
    df["YEAR"] = df["DATE"].dt.year
    df["QUARTER"] = "Q" + ((df["DATE"].dt.month.sub(1) // 3) + 1).astype(str)

    # All years combined
    agg_all = df.groupby(["PRODUCT_GROUP", "QUARTER"], as_index=False)["AMOUNT"].sum()
    pivot_all = agg_all.pivot(index="PRODUCT_GROUP", columns="QUARTER", values="AMOUNT").fillna(0)
    pivot_all = pivot_all.reindex(columns=["Q1", "Q2", "Q3", "Q4"], fill_value=0)
    pivot_all.reset_index().to_csv(OUTPUT_DIR / "quarterly_claims_by_product.csv", index=False)

    # Per-year breakdown
    agg_year = df.groupby(["YEAR", "PRODUCT_GROUP", "QUARTER"], as_index=False)["AMOUNT"].sum()
    pivot_year = agg_year.pivot_table(
        index=["YEAR", "PRODUCT_GROUP"],
        columns="QUARTER",
        values="AMOUNT",
        fill_value=0,
    )
    pivot_year = pivot_year.reindex(columns=["Q1", "Q2", "Q3", "Q4"], fill_value=0)
    pivot_year.reset_index().to_csv(OUTPUT_DIR / "quarterly_claims_by_product_year.csv", index=False)

    print("\nSaved:")
    print(f"- {OUTPUT_DIR / 'quarterly_claims_by_product.csv'}")
    print(f"- {OUTPUT_DIR / 'quarterly_claims_by_product_year.csv'}")

    # Display a small preview
    preview = pivot_all.reset_index().head().to_string(index=False)
    print("\nSample (all years combined):")
    print(preview)


def main():
    print("Aggregating Contract_Claim_Dump data by product group and quarter...")
    df = load_claims()
    print(f"Total rows loaded: {len(df):,}")
    aggregate_quarters(df)


if __name__ == "__main__":
    main()
