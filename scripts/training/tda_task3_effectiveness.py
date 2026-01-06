"""
Task 3: TDA Effectiveness on Product Groups
- Uses TDA SIP plan files (TDA_FY *.xls*) to mark TDA-active months by product group
- Uses Contract_Claim_Dump CSVs for sales baseline (Approved claims only)
- Computes lift vs. non-TDA baseline at product-group level
Outputs: results/tda_task3_summary.csv, activity mix, and monthly joined view
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")
CLAIM_DIR = DATA_DIR / "csv"
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------

def parse_date(val):
    """Robust date parser for mixed formats."""
    if pd.isna(val):
        return pd.NaT
    if isinstance(val, (pd.Timestamp, datetime)):
        return pd.to_datetime(val)
    val = str(val).strip()
    for fmt in ["%d-%b-%y", "%d-%b-%Y", "%d-%m-%y", "%d-%m-%Y", "%Y-%m-%d"]:
        try:
            return pd.to_datetime(val, format=fmt, dayfirst=True)
        except Exception:
            continue
    try:
        return pd.to_datetime(val, dayfirst=True, errors="coerce")
    except Exception:
        return pd.NaT

def expand_months(start_ts, end_ts):
    """Return list of Month periods between start and end inclusive."""
    if pd.isna(start_ts) or pd.isna(end_ts):
        return []
    if end_ts < start_ts:
        start_ts, end_ts = end_ts, start_ts
    start_p = start_ts.to_period("M")
    end_p = end_ts.to_period("M")
    return list(pd.period_range(start=start_p, end=end_p, freq="M"))

# --------------------------------------------------------------------------------------
# Load TDA SIP plan data
# --------------------------------------------------------------------------------------

def load_tda_months():
    tda_files = sorted(list(DATA_DIR.glob("TDA_FY *.xls*")))
    if not tda_files:
        raise FileNotFoundError("No TDA_FY *.xls* files found under data/")

    frames = []
    for fpath in tda_files:
        df = pd.read_excel(
            fpath,
            sheet_name=0,
            usecols=lambda c: c in {
                "TDA_PLAN_ID",
                "PRODUCT_GROUP",
                "BRAND",
                "SIP_LOCATION",
                "CHANNEL",
                "TACTIC_TYPE",
                "DETAILS_ACTIVITY_TYPE",
                "DETAILS_STATUS",
                "TACTIC_START_DATE",
                "TACTIC_END_DATE",
            },
        )
        df["DETAILS_STATUS"] = df["DETAILS_STATUS"].astype(str).str.strip()
        df = df[df["DETAILS_STATUS"].eq("Approved")]
        df = df.dropna(subset=["PRODUCT_GROUP", "TACTIC_START_DATE", "TACTIC_END_DATE"])
        df["TACTIC_START_DATE"] = df["TACTIC_START_DATE"].apply(parse_date)
        df["TACTIC_END_DATE"] = df["TACTIC_END_DATE"].apply(parse_date)
        df["FISCAL_YEAR"] = fpath.stem.split()[1] if len(fpath.stem.split()) > 1 else "NA"
        df["MONTHS"] = df.apply(lambda r: expand_months(r["TACTIC_START_DATE"], r["TACTIC_END_DATE"]), axis=1)
        df = df.explode("MONTHS")
        df = df.dropna(subset=["MONTHS"])
        df["MONTH_START"] = df["MONTHS"].dt.to_timestamp()
        frames.append(df[
            [
                "TDA_PLAN_ID",
                "PRODUCT_GROUP",
                "BRAND",
                "SIP_LOCATION",
                "CHANNEL",
                "TACTIC_TYPE",
                "DETAILS_ACTIVITY_TYPE",
                "FISCAL_YEAR",
                "MONTHS",
                "MONTH_START",
            ]
        ])

    tda_months = pd.concat(frames, ignore_index=True)
    tda_months = tda_months.rename(columns={"MONTHS": "YEAR_MONTH"})
    return tda_months

# --------------------------------------------------------------------------------------
# Load sales data (Contract_Claim_Dump)
# --------------------------------------------------------------------------------------

def load_sales_monthly():
    claim_files = sorted(list(CLAIM_DIR.glob("Contract_Claim_Dump*.csv")))
    if not claim_files:
        raise FileNotFoundError("No Contract_Claim_Dump*.csv files found under data/csv/")

    frames = []
    for csv_file in claim_files:
        df = pd.read_csv(csv_file, low_memory=False)
        required = {"CUSTOMER_NAME", "PRODUCT_GROUP", "AMOUNT", "CLAIM_DATE", "MONTH", "STATUS"}
        if not required.issubset(set(df.columns)):
            continue
        df = df[df["STATUS"].str.strip().eq("Approved")]
        if df.empty:
            continue
        df_clean = pd.DataFrame()
        df_clean["PRODUCT_GROUP"] = df["PRODUCT_GROUP"].astype(str).str.strip()
        df_clean["SALES"] = pd.to_numeric(df["AMOUNT"], errors="coerce")
        df_clean["DATE"] = df["CLAIM_DATE"].apply(parse_date)
        df_clean = df_clean.dropna(subset=["SALES", "DATE", "PRODUCT_GROUP"])
        df_clean["YEAR_MONTH"] = df_clean["DATE"].dt.to_period("M")
        frames.append(df_clean[["PRODUCT_GROUP", "YEAR_MONTH", "SALES"]])

    sales = pd.concat(frames, ignore_index=True)
    sales = sales.groupby(["PRODUCT_GROUP", "YEAR_MONTH"], as_index=False)["SALES"].sum()
    sales["MONTH_START"] = sales["YEAR_MONTH"].dt.to_timestamp()
    return sales

# --------------------------------------------------------------------------------------
# Baseline and lift computation
# --------------------------------------------------------------------------------------

def add_baseline(sales_with_flags):
    def compute_group(g):
        g = g.sort_values("MONTH_START").copy()
        history = []
        baselines = []
        for _, row in g.iterrows():
            baseline = np.mean(history[-3:]) if history else np.nan
            baselines.append(baseline)
            if not row["IS_TDA_MONTH"]:
                history.append(row["SALES"])
        g["BASELINE"] = baselines
        non_tda_mean = g.loc[~g["IS_TDA_MONTH"], "SALES"].mean()
        g["BASELINE"] = g["BASELINE"].fillna(non_tda_mean)
        return g

    groups = []
    for _, g in sales_with_flags.groupby("PRODUCT_GROUP"):
        groups.append(compute_group(g))
    enhanced = pd.concat(groups, ignore_index=True)
    enhanced["BASELINE"] = enhanced["BASELINE"].replace(0, np.nan)
    enhanced["LIFT_ABS"] = enhanced["SALES"] - enhanced["BASELINE"]
    enhanced["LIFT_PCT"] = enhanced["LIFT_ABS"] / enhanced["BASELINE"]
    return enhanced

# --------------------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("TASK 3: TDA EFFECTIVENESS ON PRODUCT GROUPS")
    print("=" * 80)

    tda_months = load_tda_months()
    print(f"Loaded TDA months: {len(tda_months):,} rows, {tda_months['PRODUCT_GROUP'].nunique()} products")

    sales = load_sales_monthly()
    print(f"Loaded sales: {len(sales):,} rows, {sales['PRODUCT_GROUP'].nunique()} products")

    # Restrict to overlapping timeframe
    max_sales_month = sales["MONTH_START"].max()
    min_sales_month = sales["MONTH_START"].min()
    tda_months = tda_months[(tda_months["MONTH_START"] >= min_sales_month) & (tda_months["MONTH_START"] <= max_sales_month)]

    # Flags per product-month
    tda_flags = (
        tda_months.groupby(["PRODUCT_GROUP", "YEAR_MONTH"], as_index=False)
        .agg({"TDA_PLAN_ID": "nunique"})
        .rename(columns={"TDA_PLAN_ID": "TDA_PLANS"})
    )

    merged = sales.merge(tda_flags, on=["PRODUCT_GROUP", "YEAR_MONTH"], how="left")
    merged["IS_TDA_MONTH"] = merged["TDA_PLANS"].notna()

    merged = add_baseline(merged)

    # Summaries
    tda_only = merged[merged["IS_TDA_MONTH"] & merged["BASELINE"].notna()].copy()
    summary = (
        tda_only.groupby("PRODUCT_GROUP")
        .agg(
            tda_months=("YEAR_MONTH", "nunique"),
            tda_plans=("TDA_PLANS", "sum"),
            sales_tda=("SALES", "sum"),
            baseline_mean=("BASELINE", "mean"),
            lift_pct_mean=("LIFT_PCT", "mean"),
            lift_pct_median=("LIFT_PCT", "median"),
            lift_abs_total=("LIFT_ABS", "sum"),
        )
        .reset_index()
        .sort_values("lift_pct_mean", ascending=False)
    )

    # Activity mix
    activity_mix = (
        tda_months.groupby(["PRODUCT_GROUP", "DETAILS_ACTIVITY_TYPE"], as_index=False)
        .size()
        .rename(columns={"size": "ACTIVITY_COUNT"})
        .sort_values(["PRODUCT_GROUP", "ACTIVITY_COUNT"], ascending=[True, False])
    )

    # Save outputs
    summary_path = RESULTS_DIR / "tda_task3_summary.csv"
    activity_path = RESULTS_DIR / "tda_task3_activity_mix.csv"
    monthly_path = RESULTS_DIR / "tda_task3_monthly_join.csv"

    summary.to_csv(summary_path, index=False)
    activity_mix.to_csv(activity_path, index=False)
    merged.to_csv(monthly_path, index=False)

    print("\nOutputs saved:")
    print(f"- Summary: {summary_path}")
    print(f"- Activity mix: {activity_path}")
    print(f"- Monthly joined: {monthly_path}")

    print("\nTop 10 products by average lift% (TDA months):")
    display_cols = [
        "PRODUCT_GROUP",
        "tda_months",
        "tda_plans",
        "sales_tda",
        "baseline_mean",
        "lift_pct_mean",
        "lift_pct_median",
    ]
    print(summary[display_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
