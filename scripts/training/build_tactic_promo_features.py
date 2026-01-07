"""
Build promo intensity features from TACTIC claim/plan data.
Output: results/tactic_promo_intensity.csv with PRODUCT_GROUP x YEAR_MONTH aggregates.
"""

import pandas as pd
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data/csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

CLAIM_GLOB = "TACTIC_*_CLAIMS_DIMNS*.csv"
PLAN_GLOB = "TACTIC_*_TACTIC_PLAN_DIMNS*.csv"

MONTH_MAP = {
    'January':1,'February':2,'March':3,'April':4,'May':5,'June':6,
    'July':7,'August':8,'September':9,'October':10,'November':11,'December':12
}


def load_plan_lookup():
    plan_files = sorted(DATA_DIR.glob(PLAN_GLOB))
    frames = []
    for pf in plan_files:
        df = pd.read_csv(pf, low_memory=False)
        cols = [c for c in ["TACTIC_PLAN","PRODUCT_GROUP","COUNTRY","CHANNEL","TACTIC_TYPE"] if c in df.columns]
        frames.append(df[cols].drop_duplicates())
    return pd.concat(frames, ignore_index=True)


def parse_month_year(month_str, period_str):
    if pd.isna(month_str):
        return None
    m = MONTH_MAP.get(str(month_str).strip(), None)
    if m is None:
        return None
    # try to infer year from period like "FY 2023 - 2024"
    yr = None
    if isinstance(period_str, str) and "20" in period_str:
        parts = [p for p in period_str.replace("FY"," ").replace("-"," ").split() if p.isdigit()]
        if parts:
            yr = int(parts[-1])  # take ending year
    if yr is None:
        return None
    return pd.Period(f"{yr}-{m:02d}", freq="M")


def main():
    plan_lookup = load_plan_lookup()
    plan_lookup = plan_lookup.dropna(subset=["TACTIC_PLAN","PRODUCT_GROUP"])

    claim_files = sorted(DATA_DIR.glob(CLAIM_GLOB))
    if not claim_files:
        raise FileNotFoundError("No tactic claim files found")

    frames = []
    for cf in claim_files:
        df = pd.read_csv(cf, low_memory=False)
        # Filter approved claims only
        if "CLAIM_STATUS" in df.columns:
            df = df[df["CLAIM_STATUS"].astype(str).str.strip().eq("Approved")]
        # Map plan to product group
        df = df.merge(plan_lookup, on="TACTIC_PLAN", how="left")
        df = df.dropna(subset=["PRODUCT_GROUP"])
        # Parse month/year
        df["YEAR_MONTH"] = df.apply(lambda r: parse_month_year(r.get("MONTH"), r.get("PERIOD")), axis=1)
        df = df.dropna(subset=["YEAR_MONTH"])
        # Amount
        amt_col = "CLAIM_AMOUNT" if "CLAIM_AMOUNT" in df.columns else None
        if amt_col:
            df["CLAIM_AMOUNT"] = pd.to_numeric(df[amt_col], errors="coerce")
        else:
            df["CLAIM_AMOUNT"] = 0.0
        frames.append(df[["PRODUCT_GROUP","YEAR_MONTH","CLAIM_AMOUNT","COUNTRY","CHANNEL","TACTIC_TYPE"]])

    all_claims = pd.concat(frames, ignore_index=True)
    all_claims["MONTH_START"] = all_claims["YEAR_MONTH"].dt.to_timestamp()

    agg = (
        all_claims.groupby(["PRODUCT_GROUP","YEAR_MONTH"], as_index=False)
        .agg(
            promo_amount=("CLAIM_AMOUNT","sum"),
            promo_count=("CLAIM_AMOUNT","count"),
        )
        .sort_values(["PRODUCT_GROUP","YEAR_MONTH"])
    )

    out_path = RESULTS_DIR / "tactic_promo_intensity.csv"
    agg.to_csv(out_path, index=False)
    print(f"Saved promo intensity to {out_path} (rows={len(agg)})")


if __name__ == "__main__":
    main()
