# task2_sales_forcast_v7_tweedie_part1.py
"""
Part 1 — Data preparation, promo seasonality, feature engineering, segmentation.

Outputs:
 - models/monthly_processed.parquet        (main prepared dataset)
 - models/sales_segments_lookup.parquet    (customer-product -> segment mapping)
 - models/promo_seasonality.parquet        (promo seasonality table)
 - models/feature_metadata.json            (list of features and metadata)
"""

import os
import glob
import json
from typing import List, Dict
import numpy as np
import pandas as pd
import joblib

# ----------------------------
# CONFIG
# ----------------------------
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLAIMS_DIR = os.path.join(DATA_DIR, "contracts_claims_extracted")

os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------
# HELPERS (robust column detection)
# ----------------------------
def detect_column(df: pd.DataFrame, candidates: List[str]):
    cols = {c.strip().upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in cols:
            return cols[cand.upper()]
    return None

def extract_claim_amount(df: pd.DataFrame):
    """Rename whichever monetary column exists to CLAIM_AMOUNT"""
    CLAIM_COL_CANDIDATES = [
        "CLAIM_AMOUNT", "CONVERTED_AMOUNT", "AMENDED_DEAL_AMOUNT",
        "ORIGINAL_DEAL_AMOUNT", "ADJUSTMENT_VALUE_AMOUNT", "AMOUNT", "VALUE", "NET_AMOUNT"
    ]
    cols_upper = {c.strip().upper(): c for c in df.columns}
    for cand in CLAIM_COL_CANDIDATES:
        if cand in cols_upper:
            return df.rename(columns={cols_upper[cand]: "CLAIM_AMOUNT"})
    raise KeyError("No claim amount column found among candidates: " + ", ".join(CLAIM_COL_CANDIDATES))


# ----------------------------
# 1) LOAD CLAIM CSV FILES (UAE claims)
# ----------------------------
print("📂 Loading monthly UAE claim files...")
csv_files = glob.glob(os.path.join(CLAIMS_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError(f"No claim CSV files found under {CLAIMS_DIR}")

df_parts = []
for f in csv_files:
    try:
        tmp = pd.read_csv(f, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        tmp["__SRC_FILE__"] = os.path.basename(f)
        df_parts.append(tmp)
    except Exception as e:
        print(f"⚠️ Skipping file {f} due to read error: {e}")
if not df_parts:
    raise RuntimeError("No CSVs could be read from claims directory")

claims_df = pd.concat(df_parts, ignore_index=True)
print(f"📊 Raw rows loaded: {len(claims_df):,}")

# ----------------------------
# 1a) Detect & rename required columns in claims_df
# ----------------------------
REQUIRED_CANDIDATES = {
    "CLAIM_DATE": ["CLAIM_DATE", "CLAIM_CREATION_DATE", "CLAIM_CREATION", "DATE", "TRANS_DATE"],
    "CUSTOMER": ["CUSTOMER_NAME", "CUSTOMER", "CUSTOMER_CODE", "ACCOUNT_NAME"],
    "PRODUCT_GROUP": ["PRODUCT_GROUP", "PRODUCT", "PRODUCT_GEN_GROUP", "PRODUCT_GENERIC_GROUP", "PRODUCT_SUB_GROUP", "CATEGORY", "BRAND"],
    "AMOUNT": ["AMOUNT", "CLAIM_AMOUNT", "CONVERTED_AMOUNT", "ORIGINAL_DEAL_AMOUNT", "VALUE", "NET_AMOUNT"],
    "REGION": ["GEOGRAPHICAL_CLUSTER", "REGION", "LOCATION", "COUNTRY", "SITE"]
}

rename_map = {}
for target, cand_list in REQUIRED_CANDIDATES.items():
    found = detect_column(claims_df, cand_list)
    if found is None:
        raise KeyError(f"Required column for '{target}' not found. Candidates: {cand_list}")
    rename_map[found] = target

claims_df = claims_df.rename(columns=rename_map)
claims_df = claims_df[["CLAIM_DATE", "CUSTOMER", "PRODUCT_GROUP", "AMOUNT", "REGION"]].copy()

# parse dates, numeric amounts, and filter
claims_df["CLAIM_DATE"] = pd.to_datetime(claims_df["CLAIM_DATE"], errors="coerce", dayfirst=True)
claims_df = claims_df[claims_df["CLAIM_DATE"].notna()].copy()
claims_df["AMOUNT"] = pd.to_numeric(claims_df["AMOUNT"], errors="coerce").fillna(0)
claims_df = claims_df[claims_df["AMOUNT"] > 0].copy()

claims_df["YEAR"] = claims_df["CLAIM_DATE"].dt.year
claims_df["MONTH"] = claims_df["CLAIM_DATE"].dt.month
claims_df["PLOT_DATE"] = claims_df["CLAIM_DATE"].dt.to_period("M").dt.to_timestamp()

print(f"✅ Clean rows: {len(claims_df):,}")

# ----------------------------
# 2) Aggregate monthly claims
# ----------------------------
monthly = (
    claims_df
    .groupby(["CUSTOMER", "PRODUCT_GROUP", "REGION", "YEAR", "MONTH", "PLOT_DATE"], dropna=False)
    .agg(SALES=("AMOUNT", "sum"), CLAIM_COUNT=("AMOUNT", "count"))
    .reset_index()
)
print(f"📈 Aggregated monthly rows: {len(monthly):,}")

# ----------------------------
# 3) Promo seasonality: PLAN <-> CLAIM mapping (multi-year)
# ----------------------------
print("📈 Computing promo seasonality using PLAN ↔ CLAIM mapping (multi-year)...")

# discover files (support versioned filenames like *_21_22, *_22_23)
TDA_PLAN_FILES   = glob.glob(os.path.join(DATA_DIR, "TDA*PLAN*DIMNS*.csv")) + glob.glob(os.path.join(DATA_DIR, "SIP_PLAN_DIMNS*.csv"))
TDA_CLAIM_FILES  = glob.glob(os.path.join(DATA_DIR, "TDA*CLAIM*DIMNS*.csv")) + glob.glob(os.path.join(DATA_DIR, "SIP_CLAIM_DIMNS*.csv"))
TACTIC_PLAN_FILES   = glob.glob(os.path.join(DATA_DIR, "TACTIC*PLAN*DIMNS*.csv"))
TACTIC_CLAIM_FILES  = glob.glob(os.path.join(DATA_DIR, "TACTIC*CLAIM*DIMNS*.csv")) + glob.glob(os.path.join(DATA_DIR, "TACTIC_CM_CLAIMS_DIMNS*.csv"))

def safe_concat_csv(file_list: List[str]) -> pd.DataFrame:
    parts = []
    for f in file_list:
        try:
            tmp = pd.read_csv(f, low_memory=False)
            tmp.columns = tmp.columns.str.strip()
            tmp["__SRC_FILE__"] = os.path.basename(f)
            parts.append(tmp)
        except Exception as e:
            print(f"⚠️ Could not read promo file {f}: {e}")
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

tda_plan = safe_concat_csv(TDA_PLAN_FILES)
tda_claim = safe_concat_csv(TDA_CLAIM_FILES)
tactic_plan = safe_concat_csv(TACTIC_PLAN_FILES)
tactic_claim = safe_concat_csv(TACTIC_CLAIM_FILES)

if tda_plan.empty and tactic_plan.empty:
    print("⚠️ No plan files found for TDA/TACTIC. Promo seasonality will be neutral.")
if tda_claim.empty and tactic_claim.empty:
    print("⚠️ No claim files found for TDA/TACTIC. Promo seasonality will be neutral.")

# normalize uppercase col names for plan/claim tables
for df_ in [tda_plan, tda_claim, tactic_plan, tactic_claim]:
    if not df_.empty:
        df_.columns = df_.columns.str.strip().str.upper()

def prepare_plan_claim(plan_df: pd.DataFrame, claim_df: pd.DataFrame,
                       plan_id_candidates: List[str], claim_plan_id_candidates: List[str]) -> pd.DataFrame:
    """Return merged table with columns PRODUCT_GROUP, MONTH, CLAIM_AMOUNT"""
    if plan_df.empty or claim_df.empty:
        return pd.DataFrame(columns=["PRODUCT_GROUP", "MONTH", "CLAIM_AMOUNT"])
    # detect plan id in plan_df and product column
    plan_id_col = detect_column(plan_df, plan_id_candidates)
    prod_col = detect_column(plan_df, ["PRODUCT_GROUP", "PRODUCT", "PRODUCT_GEN_GROUP", "BRAND", "SKU"])
    if plan_id_col is None or prod_col is None:
        raise KeyError("Plan file missing plan id or product group column")
    plan_small = plan_df[[plan_id_col, prod_col]].drop_duplicates().rename(columns={plan_id_col: "PLAN_ID", prod_col: "PRODUCT_GROUP"})
    # detect plan id in claim_df
    claim_plan_col = detect_column(claim_df, claim_plan_id_candidates)
    if claim_plan_col is None:
        raise KeyError("Claim file missing plan id column")
    # standardize monetary col to CLAIM_AMOUNT
    claim_df_std = extract_claim_amount(claim_df)
    month_col = detect_column(claim_df_std, ["MONTH", "PERIOD", "DEAL_MONTH", "CLAIM_MONTH"])
    if month_col is None:
        raise KeyError("Claim file missing month/period column")
    claim_small = claim_df_std[[claim_plan_col, month_col, "CLAIM_AMOUNT"]].rename(columns={claim_plan_col: "PLAN_ID", month_col: "MONTH"})
    merged = claim_small.merge(plan_small, on="PLAN_ID", how="left")
    return merged[["PRODUCT_GROUP", "MONTH", "CLAIM_AMOUNT"]]

# prepare merges (be tolerant of different plan id column names)
tda_merged = prepare_plan_claim(
    tda_plan, tda_claim,
    plan_id_candidates=["TDA_PLAN_ID", "TDA_PLAN", "TDA_PLAN_ROW_ID", "PLAN_ID"],
    claim_plan_id_candidates=["TDA_PLAN_ID", "TDA_PLAN", "PLAN_ID"]
)

tactic_merged = prepare_plan_claim(
    tactic_plan, tactic_claim,
    plan_id_candidates=["TACTIC_PLAN", "TACTIC_PLAN_ID", "PLAN_ID"],
    claim_plan_id_candidates=["TACTIC_PLAN", "TACTIC_PLAN_ID", "PLAN_ID"]
)

promo_raw = pd.concat([tda_merged, tactic_merged], ignore_index=True)
promo_raw = promo_raw.dropna(subset=["PRODUCT_GROUP"])
promo_raw = promo_raw[pd.to_numeric(promo_raw["CLAIM_AMOUNT"], errors="coerce").fillna(0) > 0]

if promo_raw.empty:
    print("⚠️ No promo claim rows after merging PLAN↔CLAIM. Promo seasonality will be neutral.")
    promo_seasonality = pd.DataFrame(columns=["PRODUCT_GROUP", "MONTH", "PROMO_SEASONALITY_INDEX", "PROMO_FREQ", "PROMO_SUM"])
else:
    # normalize month -> int
    promo_raw["MONTH"] = promo_raw["MONTH"].astype(str).str.strip()
    def month_to_int(x):
        x = str(x)
        if x.isdigit():
            try:
                return int(float(x))
            except Exception:
                return np.nan
        key = x[:3].upper()
        mapping = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
        return mapping.get(key, np.nan)
    promo_raw["MONTH"] = promo_raw["MONTH"].apply(month_to_int)
    promo_raw = promo_raw[promo_raw["MONTH"].notna()].copy()
    promo_raw["MONTH"] = promo_raw["MONTH"].astype(int)

    promo_stats = (
        promo_raw
        .groupby(["PRODUCT_GROUP", "MONTH"])["CLAIM_AMOUNT"]
        .agg(PROMO_MEAN="mean", PROMO_SUM="sum", PROMO_FREQ="count")
        .reset_index()
    )

    # compute normalized seasonality index per product
    promo_stats["PROMO_SEASONALITY_INDEX"] = (
        promo_stats["PROMO_MEAN"] /
        promo_stats.groupby("PRODUCT_GROUP")["PROMO_MEAN"].transform("mean")
    )

    # clip extreme values to avoid exploding promo indices (optional)
    promo_stats["PROMO_SEASONALITY_INDEX"] = promo_stats["PROMO_SEASONALITY_INDEX"].clip(lower=0.25, upper=5.0)

    promo_seasonality = promo_stats[["PRODUCT_GROUP", "MONTH", "PROMO_SEASONALITY_INDEX", "PROMO_FREQ", "PROMO_SUM"]].copy()

# persist promo table for inspection
promo_seasonality.to_parquet(os.path.join(MODELS_DIR, "promo_seasonality.parquet"), index=False)
print(f"📦 Promo seasonality saved to models/promo_seasonality.parquet (rows: {len(promo_seasonality):,})")

# ----------------------------
# 4) Merge promo features into monthly
# ----------------------------
monthly = monthly.merge(promo_seasonality, on=["PRODUCT_GROUP", "MONTH"], how="left")
monthly["PROMO_SEASONALITY_INDEX"] = monthly["PROMO_SEASONALITY_INDEX"].fillna(1.0)
monthly["PROMO_FREQ"] = pd.to_numeric(monthly.get("PROMO_FREQ", 0)).fillna(0).astype(int)
monthly["PROMO_SUM"] = pd.to_numeric(monthly.get("PROMO_SUM", 0)).fillna(0.0)

# ----------------------------
# 4.5) Add time-lag / trend helper (non-leaking)
# ----------------------------
def add_time_lag_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["CUSTOMER", "PRODUCT_GROUP", "PLOT_DATE"]).reset_index(drop=True)
    grp = df.groupby(["CUSTOMER", "PRODUCT_GROUP"], sort=False)

    # lags
    for lag in (1, 3, 6, 12):
        df[f"SALES_LAG_{lag}"] = grp["SALES"].shift(lag)

    # rolling means (uses past only)
    for w in (3, 6, 12):
        df[f"SALES_ROLL_MEAN_{w}"] = grp["SALES"].transform(lambda s: s.shift(1).rolling(window=w, min_periods=1).mean())

    df["SALES_ROLL_STD_3"] = grp["SALES"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).std())

    # YoY
    df["SALES_LAG_12"] = grp["SALES"].shift(12)
    df["YOY_ABS_12"] = df["SALES"] - df["SALES_LAG_12"]
    df["YOY_PCT_12"] = (df["SALES"] / (df["SALES_LAG_12"] + 1e-9)) - 1.0

    # rolling slope: small helper
    def rolling_slope(series, window):
        shifted = series.shift(1)
        def slope(arr):
            arr = np.asarray(arr)
            if arr.size < 2:
                return 0.0
            x = np.arange(arr.size, dtype=float)
            try:
                return float(np.polyfit(x, arr, 1)[0])
            except Exception:
                return 0.0
        return shifted.rolling(window=window, min_periods=1).apply(slope, raw=True)
    for w in (3, 6):
        df[f"TREND_SLOPE_{w}"] = grp["SALES"].transform(lambda s: rolling_slope(s, w))

    # percent change deltas
    df["PCT_CHANGE_LAG1"] = (df["SALES"] / (df["SALES_LAG_1"] + 1e-9)) - 1.0
    df["PCT_CHANGE_LAG3"] = (df["SALES"] / (df["SALES_LAG_3"] + 1e-9)) - 1.0

    # clean numeric new cols
    new_numeric_cols = [c for c in df.columns if any(pref in c for pref in ("SALES_LAG_", "SALES_ROLL_", "YOY_", "TREND_SLOPE_", "PCT_CHANGE_"))]
    for c in new_numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return df

monthly = add_time_lag_trend_features(monthly)

# ----------------------------
# 5) Additional feature engineering & interactions
# ----------------------------
print("🧠 Engineering features...")

# base aggregates
cust_avg = monthly.groupby("CUSTOMER")["SALES"].mean().to_dict()
prod_avg = monthly.groupby("PRODUCT_GROUP")["SALES"].mean().to_dict()
geo_avg  = monthly.groupby("REGION")["SALES"].mean().to_dict()

cust_prod_avg = monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"].mean().to_dict()

# sort then create RECENT_AVG (rolling 3 months including current)
monthly = monthly.sort_values(["CUSTOMER", "PRODUCT_GROUP", "PLOT_DATE"])
monthly["RECENT_AVG"] = monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"].rolling(3, min_periods=1).mean().reset_index(level=[0,1], drop=True)

# Ensure lag features exist (already added earlier) — fillna safely
for lag in (1,2,3,6,12):
    col = f"SALES_LAG_{lag}"
    if col not in monthly.columns:
        monthly[col] = 0.0
monthly["SALES_ROLL_MEAN_3"] = monthly.get("SALES_ROLL_MEAN_3", 0.0)
monthly["SALES_ROLL_STD_3"] = monthly.get("SALES_ROLL_STD_3", 0.0)

# mapping
monthly["Cust_Avg"] = monthly["CUSTOMER"].map(cust_avg).fillna(0.0)
monthly["Prod_Avg"] = monthly["PRODUCT_GROUP"].map(prod_avg).fillna(0.0)
monthly["Geo_Avg"]  = monthly["REGION"].map(geo_avg).fillna(0.0)

monthly["Cust_Prod_Avg"] = monthly.apply(lambda r: cust_prod_avg.get((r["CUSTOMER"], r["PRODUCT_GROUP"]), r["Prod_Avg"]), axis=1)

# promo-derived ratios
monthly["PROMO_SUM_OVER_PROD"] = monthly["PROMO_SUM"] / (monthly["Prod_Avg"] + 1e-6)
monthly["CUST_PROD_OVER_PROD"] = monthly["Cust_Prod_Avg"] / (monthly["Prod_Avg"] + 1e-6)
monthly["RECENT_X_PROMO"] = monthly["RECENT_AVG"] * monthly["PROMO_SEASONALITY_INDEX"]

monthly["Is_Summer"] = monthly["MONTH"].isin([6,7,8]).astype(int)

# ensure no NaNs in numeric features
numeric_cols = monthly.select_dtypes(include=[np.number]).columns
monthly[numeric_cols] = monthly[numeric_cols].fillna(0.0)

# ----------------------------
# 6) SEGMENTATION (HIGH / MEDIUM / LOW)
# ----------------------------
def define_segments(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_df.copy()
    # compute pair level stats
    pair = df.groupby(["CUSTOMER", "PRODUCT_GROUP"], dropna=False).agg(
        n_months = ("SALES", "size"),
        total_amount = ("SALES", "sum")
    ).reset_index()
    # thresholds
    p80 = np.percentile(pair["total_amount"].values, 80) if len(pair)>0 else 0.0
    p50 = np.percentile(pair["total_amount"].values, 50) if len(pair)>0 else 0.0

    def _rule(row):
        if (row["n_months"] >= 12) or (row["total_amount"] >= p80):
            return "high"
        if (row["n_months"] >= 6) or (row["total_amount"] >= p50):
            return "medium"
        return "low"

    pair["SEGMENT"] = pair.apply(_rule, axis=1)
    out = df.merge(pair[["CUSTOMER", "PRODUCT_GROUP", "SEGMENT"]], on=["CUSTOMER", "PRODUCT_GROUP"], how="left")
    out["SEGMENT"] = out["SEGMENT"].fillna("low")
    return out, pair

monthly, pair_stats = define_segments(monthly)
print("🔖 Segmentation done. Segment counts:")
print(monthly["SEGMENT"].value_counts())

# persist pair_stats (lookup) for runtime routing
pair_stats.to_parquet(os.path.join(MODELS_DIR, "sales_segments_lookup.parquet"), index=False)
print(f"📦 Saved segment lookup to models/sales_segments_lookup.parquet (rows: {len(pair_stats):,})")

# ----------------------------
# 7) Save processed monthly dataset and simple metadata
# ----------------------------
save_path = os.path.join(MODELS_DIR, "monthly_processed.parquet")
monthly.to_parquet(save_path, index=False)
print(f"📦 Saved processed monthly dataset to {save_path} (rows: {len(monthly):,})")

# also save feature metadata for next step (Part 2 training expects FEATURES list)
FEATURES = [
    "Cust_Avg","Prod_Avg","Geo_Avg","Cust_Prod_Avg","RECENT_AVG",
    "SALES_LAG_1","SALES_LAG_2","SALES_LAG_3",
    "SALES_ROLL_MEAN_3","SALES_ROLL_STD_3",
    "MONTH","Is_Summer","PROMO_SEASONALITY_INDEX",
    "PROMO_FREQ","PROMO_SUM","PROMO_SUM_OVER_PROD","CUST_PROD_OVER_PROD","RECENT_X_PROMO",
    # keep TREND features & YOY for potential use
    "YOY_ABS_12","YOY_PCT_12","TREND_SLOPE_3","TREND_SLOPE_6","PCT_CHANGE_LAG1","PCT_CHANGE_LAG3"
]

meta = {
    "n_rows": len(monthly),
    "n_customers": int(monthly["CUSTOMER"].nunique()),
    "n_products": int(monthly["PRODUCT_GROUP"].nunique()),
    "features": FEATURES
}
with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "w") as fh:
    json.dump(meta, fh, indent=2)

print("✅ Part 1 complete — prepared dataset + promo seasonality + segmentation are saved to models/.")
print("Next: run Part 2 (training) which will load models/monthly_processed.parquet and train per-segment models.")
