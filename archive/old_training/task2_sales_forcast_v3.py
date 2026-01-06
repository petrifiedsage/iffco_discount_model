import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Optional LightGBM; if unavailable, we fall back to only the sklearn model
try:
    import lightgbm as lgb
except ImportError:  # pragma: no cover - optional dependency
    lgb = None

# ----------------------------------
# CONFIG
# ----------------------------------
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
CLAIMS_DIR = os.path.join(DATA_DIR, "contracts_claims_extracted")

os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------------
# HELPERS
# ----------------------------------
def extract_claim_amount(df):
    """
    Detect and standardize claim amount column from ERP exports
    """
    CLAIM_COL_CANDIDATES = [
        "CLAIM_AMOUNT",
        "CONVERTED_AMOUNT",
        "AMENDED_DEAL_AMOUNT",
        "ORIGINAL_DEAL_AMOUNT",
        "ADJUSTMENT_VALUE_AMOUNT",
        "AMOUNT"
    ]
    cols = [c.upper() for c in df.columns]
    for col in CLAIM_COL_CANDIDATES:
        if col in cols:
            # rename the actual column name (respecting the original casing)
            actual = [c for c in df.columns if c.strip().upper() == col][0]
            return df.rename(columns={actual: "CLAIM_AMOUNT"})
    raise KeyError("❌ No claim amount column found in claim CSV")


def detect_column(df, candidates):
    """Return the actual column name from df if any candidate matches (case-insensitive)."""
    cols = {c.strip().upper(): c for c in df.columns}
    for cand in candidates:
        if cand.upper() in cols:
            return cols[cand.upper()]
    return None


def compute_metrics(y_true_log, y_pred_log, label):
    """Compute common regression metrics on log-scale preds."""
    y_true = np.expm1(y_true_log)
    y_pred = np.expm1(y_pred_log)
    eps = 1.0  # floor to avoid exploding percentages on tiny actuals

    metrics = {
        "log_mae": mean_absolute_error(y_true_log, y_pred_log),
        "mae_aed": mean_absolute_error(y_true, y_pred),
        "smape": np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)),
        "wape": np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps),
        "r2": r2_score(y_true, y_pred),
    }

    print(f"\n📏 Metrics for {label}:")
    print(f"   • Log-MAE: {metrics['log_mae']:.4f}")
    print(f"   • MAE (AED): {metrics['mae_aed']:,.0f}")
    print(f"   • sMAPE: {metrics['smape']*100:.2f}%")
    print(f"   • WAPE: {metrics['wape']*100:.2f}%")
    print(f"   • R² (AED scale): {metrics['r2']:.4f}")

    return metrics


# ----------------------------------
# 1. LOAD ALL UAE CLAIM CSV FILES
# ----------------------------------
print("📂 Loading monthly UAE claim files...")

csv_files = glob.glob(os.path.join(CLAIMS_DIR, "*.csv"))
if not csv_files:
    raise FileNotFoundError("❌ No claim CSV files found")

df_list = []
for f in csv_files:
    temp = pd.read_csv(f, low_memory=False)
    temp.columns = temp.columns.str.strip()
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)
print(f"📊 Raw rows loaded: {len(df):,}")

# Required columns (must exist in claim CSVs collected earlier)
# Accept alternative column names by detection
col_map = {c.strip().upper(): c for c in df.columns}

REQUIRED_CANDIDATES = {
    "CLAIM_DATE": ["CLAIM_DATE", "CLAIM_CREATION_DATE", "CLAIM_CREATION", "DATE"],
    "CUSTOMER": ["CUSTOMER_NAME", "CUSTOMER", "CUSTOMER"],
    "PRODUCT_GROUP": ["PRODUCT_GROUP", "PRODUCT", "PRODUCT_GEN_GROUP", "PRODUCT_GENERIC_GROUP", "PRODUCT_SUB_GROUP"],
    "AMOUNT": ["AMOUNT", "CLAIM_AMOUNT", "CONVERTED_AMOUNT", "ORIGINAL_DEAL_AMOUNT"],
    "REGION": ["GEOGRAPHICAL_CLUSTER", "REGION", "LOCATION", "COUNTRY"]
}

# detect and rename
rename_map = {}
for target, candidates in REQUIRED_CANDIDATES.items():
    found = detect_column(df, candidates)
    if found is None:
        raise KeyError(f"❌ Required column for '{target}' not found among candidates: {candidates}")
    rename_map[found] = target

df = df.rename(columns=rename_map)

# Keep only required columns (drop others to save memory)
df = df[["CLAIM_DATE", "CUSTOMER", "PRODUCT_GROUP", "AMOUNT", "REGION"]].copy()

df["CLAIM_DATE"] = pd.to_datetime(df["CLAIM_DATE"], errors="coerce", dayfirst=True)
df = df[df["CLAIM_DATE"].notna()]
df = df[pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0) > 0]

df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce").fillna(0)
df["YEAR"] = df["CLAIM_DATE"].dt.year
df["MONTH"] = df["CLAIM_DATE"].dt.month
df["PLOT_DATE"] = df["CLAIM_DATE"].dt.to_period("M").dt.to_timestamp()

print(f"✅ Clean rows: {len(df):,}")

# ----------------------------------
# 2. AGGREGATE MONTHLY CLAIMS
# ----------------------------------
monthly = (
    df.groupby(["CUSTOMER", "PRODUCT_GROUP", "REGION", "YEAR", "MONTH", "PLOT_DATE"])
      .agg(
          SALES=("AMOUNT", "sum"),
          CLAIM_COUNT=("AMOUNT", "count")
      )
      .reset_index()
)

# ----------------------------------
# PROMO SEASONALITY (PLAN ↔ CLAIM JOIN) - MULTI-YEAR SUPPORT
# ----------------------------------
print("📈 Computing promo seasonality using PLAN ↔ CLAIM mapping (multi-year)...")

# discover plan/claim files for TDA & TACTIC (supports versioned filenames)
TDA_PLAN_FILES   = glob.glob(os.path.join(DATA_DIR, "SIP_PLAN_DIMNS*.csv"))
TDA_CLAIM_FILES  = glob.glob(os.path.join(DATA_DIR, "SIP_CLAIM_DIMNS*.csv"))
TACTIC_PLAN_FILES  = glob.glob(os.path.join(DATA_DIR, "TACTIC_PLAN_DIMNS*.csv"))
TACTIC_CLAIM_FILES = glob.glob(os.path.join(DATA_DIR, "TACTIC_CM_CLAIMS_DIMNS*.csv"))

def safe_concat_csv(files):
    if not files:
        return pd.DataFrame()
    parts = []
    for f in files:
        tmp = pd.read_csv(f, low_memory=False)
        tmp.columns = tmp.columns.str.strip()
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True)

tda_plan = safe_concat_csv(TDA_PLAN_FILES)
tda_claim = safe_concat_csv(TDA_CLAIM_FILES)
tactic_plan = safe_concat_csv(TACTIC_PLAN_FILES)
tactic_claim = safe_concat_csv(TACTIC_CLAIM_FILES)

# If any of these are empty, continue but warn (we can still run model without promo signals)
if tda_plan.empty and tactic_plan.empty:
    print("⚠️ Warning: No PLAN files found for TDA/TACTIC. Promo seasonality will be neutral.")
if tda_claim.empty and tactic_claim.empty:
    print("⚠️ Warning: No CLAIM files found for TDA/TACTIC. Promo seasonality will be neutral.")

# normalize column names to upper for plan/claim tables
for df_ in [tda_plan, tda_claim, tactic_plan, tactic_claim]:
    if not df_.empty:
        df_.columns = df_.columns.str.strip().str.upper()

# Helper to map plan_id and product column names robustly
def prepare_plan_claim(plan_df, claim_df, plan_id_candidates, claim_plan_id_candidates):
    if plan_df.empty or claim_df.empty:
        return pd.DataFrame(columns=["PRODUCT_GROUP", "MONTH", "CLAIM_AMOUNT"])
    # detect plan id col in plan_df
    plan_id_col = detect_column(plan_df, plan_id_candidates)
    prod_col = detect_column(plan_df, ["PRODUCT_GROUP", "PRODUCT", "PRODUCT_GEN_GROUP", "PRODUCT_GENERIC_GROUP", "PRODUCT_SUB_GROUP", "BRAND"])
    if plan_id_col is None or prod_col is None:
        raise KeyError("❌ Could not find plan id or product group column in plan file.")
    plan_df = plan_df[[plan_id_col, prod_col]].drop_duplicates().rename(columns={plan_id_col: "PLAN_ID", prod_col: "PRODUCT_GROUP"})

    # detect plan id col in claim_df
    claim_plan_col = detect_column(claim_df, claim_plan_id_candidates)
    if claim_plan_col is None:
        raise KeyError("❌ Could not find plan id column in claim file.")
    claim_df = extract_claim_amount(claim_df)  # standardize monetary column
    month_col = detect_column(claim_df, ["MONTH", "PERIOD", "DEAL_MONTH"])
    if month_col is None:
        raise KeyError("❌ Could not find MONTH/PERIOD column in claim file.")
    # keep only relevant columns and rename
    claim_df = claim_df[[claim_plan_col, month_col, "CLAIM_AMOUNT"]].rename(columns={claim_plan_col: "PLAN_ID", month_col: "MONTH"})
    # join
    merged = claim_df.merge(plan_df, on="PLAN_ID", how="left")
    return merged[["PRODUCT_GROUP", "MONTH", "CLAIM_AMOUNT"]]

# prepare TDA merged
tda_merged = prepare_plan_claim(
    tda_plan, tda_claim,
    plan_id_candidates=["TDA_PLAN_ID", "TDA_PLAN_ROW_ID", "TDA_PLAN", "TDA_PLANNUMBER", "TDA_PLAN_NO"],
    claim_plan_id_candidates=["TDA_PLAN_ID", "TDA_PLAN_ROW_ID", "TDA_PLAN", "TDA_PLANNUMBER", "TDA_PLAN_NO"]
)

# prepare TACTIC merged
tactic_merged = prepare_plan_claim(
    tactic_plan, tactic_claim,
    plan_id_candidates=["TACTIC_PLAN", "TACTIC_PLAN_ID", "TACTIC", "TACTIC_PLAN_NUMBER"],
    claim_plan_id_candidates=["TACTIC_PLAN", "TACTIC_PLAN_ID", "TACTIC", "TACTIC_PLAN_NUMBER"]
)

# combine
promo_raw = pd.concat([tda_merged, tactic_merged], ignore_index=True)
promo_raw = promo_raw.dropna(subset=["PRODUCT_GROUP"])
promo_raw = promo_raw[pd.to_numeric(promo_raw["CLAIM_AMOUNT"], errors="coerce").fillna(0) > 0]

if promo_raw.empty:
    print("⚠️ No promo claim rows after merging PLAN↔CLAIM. Using neutral promo signals.")
    promo_seasonality = pd.DataFrame(columns=["PRODUCT_GROUP", "MONTH", "PROMO_SEASONALITY_INDEX", "PROMO_FREQ", "PROMO_SUM"])
else:
    # normalize MONTH in promo_raw (handle strings like 'Jan', 'JAN', numbers)
    promo_raw["MONTH"] = promo_raw["MONTH"].astype(str).str.strip()
    def month_to_int(x):
        x = str(x)
        if x.isdigit():
            return int(x)
        m = x[:3].upper()
        mapping = {"JAN":1,"FEB":2,"MAR":3,"APR":4,"MAY":5,"JUN":6,"JUL":7,"AUG":8,"SEP":9,"OCT":10,"NOV":11,"DEC":12}
        return mapping.get(m, np.nan)
    promo_raw["MONTH"] = promo_raw["MONTH"].apply(month_to_int)
    promo_raw = promo_raw[promo_raw["MONTH"].notna()]
    promo_raw["MONTH"] = promo_raw["MONTH"].astype(int)

    # compute promo stats: mean, sum, freq
    promo_stats = (
        promo_raw
        .groupby(["PRODUCT_GROUP", "MONTH"])["CLAIM_AMOUNT"]
        .agg(PROMO_MEAN="mean", PROMO_SUM="sum", PROMO_FREQ="count")
        .reset_index()
    )

    # normalized seasonality index
    promo_stats["PROMO_SEASONALITY_INDEX"] = (
        promo_stats["PROMO_MEAN"] /
        promo_stats.groupby("PRODUCT_GROUP")["PROMO_MEAN"].transform("mean")
    )

    promo_seasonality = promo_stats[["PRODUCT_GROUP", "MONTH", "PROMO_SEASONALITY_INDEX", "PROMO_FREQ", "PROMO_SUM"]].copy()

# Ensure month dtype numeric
if not promo_seasonality.empty:
    promo_seasonality["MONTH"] = promo_seasonality["MONTH"].astype("int32")

# ----------------------------------
# 4. MERGE PROMO FEATURES INTO CLAIMS
# ----------------------------------
# Merge seasonality, freq, sum
monthly = monthly.merge(
    promo_seasonality,
    on=["PRODUCT_GROUP", "MONTH"],
    how="left"
)

# fill defaults: neutral index, zero freq & sum
monthly["PROMO_SEASONALITY_INDEX"] = monthly["PROMO_SEASONALITY_INDEX"].fillna(1.0)
monthly["PROMO_FREQ"] = pd.to_numeric(monthly.get("PROMO_FREQ", 0)).fillna(0).astype(int)
monthly["PROMO_SUM"] = pd.to_numeric(monthly.get("PROMO_SUM", 0)).fillna(0.0)

# ----------------------------------
# 5. FEATURE ENGINEERING
# ----------------------------------
print("🧠 Engineering features...")

cust_avg = monthly.groupby("CUSTOMER")["SALES"].mean().to_dict()
prod_avg = monthly.groupby("PRODUCT_GROUP")["SALES"].mean().to_dict()
geo_avg  = monthly.groupby("REGION")["SALES"].mean().to_dict()

cust_prod_avg = (
    monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"]
    .mean()
    .to_dict()
)

monthly = monthly.sort_values("PLOT_DATE")

monthly["RECENT_AVG"] = (
    monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"]
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

# --- ADD: LAGS, rolling stats, and derived promo ratios (high-impact) ---
monthly = monthly.sort_values(["CUSTOMER", "PRODUCT_GROUP", "PLOT_DATE"])

# Lags
monthly["SALES_LAG_1"] = monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"].shift(1).fillna(0)
monthly["SALES_LAG_2"] = monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"].shift(2).fillna(0)
monthly["SALES_LAG_3"] = monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"].shift(3).fillna(0)

# Rolling statistics (use prior months only)
monthly["SALES_ROLL_MEAN_3"] = monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"]\
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean()).fillna(0)
monthly["SALES_ROLL_STD_3"] = monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"]\
    .transform(lambda x: x.shift(1).rolling(3, min_periods=1).std()).fillna(0)

# Promo-derived relative features
monthly["PROMO_SUM_OVER_PROD"] = monthly["PROMO_SUM"] / (monthly["Prod_Avg"] + 1e-6)
monthly["CUST_PROD_OVER_PROD"] = monthly["Cust_Prod_Avg"] / (monthly["Prod_Avg"] + 1e-6)

# Interaction (you already had RECENT_X_PROMO, recompute to ensure order)
monthly["RECENT_X_PROMO"] = monthly["RECENT_AVG"] * monthly["PROMO_SEASONALITY_INDEX"]

# Fill NaNs (after creating derived features)
monthly = monthly.fillna(0)


monthly["Cust_Avg"] = monthly["CUSTOMER"].map(cust_avg)
monthly["Prod_Avg"] = monthly["PRODUCT_GROUP"].map(prod_avg)
monthly["Geo_Avg"]  = monthly["REGION"].map(geo_avg)

monthly["Cust_Prod_Avg"] = monthly.apply(
    lambda r: cust_prod_avg.get((r["CUSTOMER"], r["PRODUCT_GROUP"]), r["Prod_Avg"]),
    axis=1
)

# Interaction feature to help amplify promo impact
monthly["RECENT_X_PROMO"] = monthly["RECENT_AVG"] * monthly["PROMO_SEASONALITY_INDEX"]

monthly["Is_Summer"] = monthly["MONTH"].isin([6, 7, 8]).astype(int)
monthly = monthly.fillna(0)

# ----------------------------------
# 6. TRAIN / TEST SPLIT
# ----------------------------------
train_df = monthly[monthly["YEAR"] <= 2023]
test_df  = monthly[monthly["YEAR"] >= 2024]

# Optional downsample to speed up prototyping on large datasets
"""MAX_TRAIN_ROWS = 80000
if len(train_df) > MAX_TRAIN_ROWS:
    train_df = train_df.sample(n=MAX_TRAIN_ROWS, random_state=42)
"""

FEATURES = [
    "Cust_Avg",
    "Prod_Avg",
    "Geo_Avg",
    "Cust_Prod_Avg",
    "RECENT_AVG",
    "SALES_LAG_1",
    "SALES_LAG_2",
    "SALES_LAG_3",
    "SALES_ROLL_MEAN_3",
    "SALES_ROLL_STD_3",
    "MONTH",
    "Is_Summer",
    "PROMO_SEASONALITY_INDEX",
    "PROMO_FREQ",
    "PROMO_SUM",
    "PROMO_SUM_OVER_PROD",
    "CUST_PROD_OVER_PROD",
    "RECENT_X_PROMO"
]


X_train = train_df[FEATURES]
y_train = np.log1p(train_df["SALES"])

X_test = test_df[FEATURES]
y_test = np.log1p(test_df["SALES"])

print(f"Train rows: {len(train_df):,}")
print(f"Test rows : {len(test_df):,}")

# ----------------------------------
# 7. TRAIN MODEL
# ----------------------------------
print("🤖 Training Histogram Gradient Boosting model (fast baseline)...")

hgb_model = HistGradientBoostingRegressor(
    loss="poisson",  # sales are positive
    learning_rate=0.08,
    max_depth=8,
    max_iter=250,
    l2_regularization=0.0,
    random_state=42
)

hgb_model.fit(X_train, y_train)
hgb_preds = hgb_model.predict(X_test)
compute_metrics(y_test, hgb_preds, label="HistGradientBoosting (sklearn)")

# Optional LightGBM model with early stopping
if lgb is not None:
    print("\n🤖 Training LightGBM model (if this hangs, ensure lightgbm is installed)...")
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_valid = lgb.Dataset(X_test, label=y_test, reference=lgb_train)

    lgb_params = {
        "objective": "regression",
        "metric": "l1",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "max_depth": -1,
        "min_data_in_leaf": 30,
        "feature_fraction": 0.9,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "verbose": -1,
    }

    lgb_model = lgb.train(
        lgb_params,
        lgb_train,
        num_boost_round=600,
        valid_sets=[lgb_train, lgb_valid],
        valid_names=["train", "valid"],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)],
    )

    lgb_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
    compute_metrics(y_test, lgb_preds, label="LightGBM")

    # Save LightGBM model separately
    lgb_model.save_model(os.path.join(MODELS_DIR, "task2_sales_forecasting_lgbm.txt"))
else:
    print("\n⚠️ LightGBM not installed; skipped LightGBM training. Install via: pip install lightgbm")

# ----------------------------------
# 8. SAVE MODEL & ARTIFACTS
# ----------------------------------
joblib.dump(hgb_model, os.path.join(MODELS_DIR, "task2_sales_forecasting_model_v6.pkl"))

# Save encoders including promo stats (dicts keyed by (PRODUCT_GROUP, MONTH))
promo_encoders = {}
if not promo_seasonality.empty:
    promo_encoders = {
        "seasonality": promo_seasonality.set_index(["PRODUCT_GROUP", "MONTH"])["PROMO_SEASONALITY_INDEX"].to_dict(),
        "freq": promo_seasonality.set_index(["PRODUCT_GROUP", "MONTH"])["PROMO_FREQ"].to_dict(),
        "sum": promo_seasonality.set_index(["PRODUCT_GROUP", "MONTH"])["PROMO_SUM"].to_dict()
    }

encoders = {
    "cust_means": cust_avg,
    "prod_means": prod_avg,
    "geo_means": geo_avg,
    "cust_prod_means": cust_prod_avg,
    "recent_means": (
        monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["RECENT_AVG"]
        .last()
        .to_dict()
    ),
    "promo_seasonality": promo_encoders  # contains seasonality/freq/sum dicts
}

joblib.dump(encoders, os.path.join(MODELS_DIR, "task2_encoders_v6.pkl"))

metadata = {
    "customers": sorted(monthly["CUSTOMER"].unique()),
    "products": sorted(monthly["PRODUCT_GROUP"].unique()),
    "regions": sorted(monthly["REGION"].unique())
}

joblib.dump(metadata, os.path.join(MODELS_DIR, "task2_metadata_v6.pkl"))

history_df = monthly[["CUSTOMER", "PRODUCT_GROUP", "PLOT_DATE", "SALES"]].copy()
history_df = history_df.rename(columns={"PLOT_DATE": "Plot_Date"})

joblib.dump(history_df, os.path.join(MODELS_DIR, "history_monthly.pkl"))

print("🏁 Model 2 (Promo-Aware v6) training complete.")
