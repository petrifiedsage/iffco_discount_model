"""task2_sales_forcast_v7_tweedie_part2.py"""
# task2_sales_forcast_v7_tweedie_part2.py
"""
Part 2 — Per-segment training + evaluation + saving artifacts.

Requires outputs from Part 1:
  - models/monthly_processed.parquet
  - models/feature_metadata.json

Trains:
  - HIGH   -> LightGBM (tweedie on RAW SALES) with early stopping
  - MEDIUM -> HistGradientBoostingRegressor on log1p(SALES)
  - LOW    -> Baseline: customer-product pair historical mean (fallback)

Saves:
  - models/task2_model_high_lgbm_tweedie.txt  (LightGBM text model)
  - models/task2_model_high_lgbm.pkl          (joblib dump of Booster if installed)
  - models/task2_model_medium_hgb.pkl
  - models/task2_model_low_baseline.pkl
  - models/task2_encoders_v8.pkl              (cust/prod means, promo encoders, recent)
  - models/task2_metrics_by_segment.json
  - models/task2_overall_metrics.json
  - results/prediction_vs_actual_{segment}.csv
"""

import os
import json
import joblib
import warnings
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# optional lgb
try:
    import lightgbm as lgb
except Exception:
    lgb = None

# ----------------------
# CONFIG
# ----------------------
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Files produced by Part 1
PROCESSED_PATH = os.path.join(MODELS_DIR, "monthly_processed.parquet")
META_PATH = os.path.join(MODELS_DIR, "feature_metadata.json")

if not os.path.exists(PROCESSED_PATH):
    raise FileNotFoundError(f"Processed dataset not found: {PROCESSED_PATH}. Run Part 1 first.")
if not os.path.exists(META_PATH):
    raise FileNotFoundError(f"Feature metadata not found: {META_PATH}. Run Part 1 first.")

# hyperparams (tweakable)
MAX_TRAIN_ROWS_PER_SEGMENT = 120_000   # cap for quick iteration, set None to disable
LGB_EARLY_STOP = 50
LGB_NUM_BOOST_ROUND = 2000

# ----------------------
# UTILITIES / METRICS
# ----------------------
def compute_metrics_from_logs(y_true_log, y_pred_log) -> Dict[str, float]:
    """Given arrays on log1p scale, return metrics computed on original AED scale and log scale."""
    y_true = np.expm1(np.array(y_true_log))
    y_pred = np.expm1(np.array(y_pred_log))
    eps = 1.0
    log_mae = mean_absolute_error(y_true_log, y_pred_log)
    mae = mean_absolute_error(y_true, y_pred)
    smape = float(np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred) + eps)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + eps))
    r2 = float(r2_score(y_true, y_pred))
    return {
        "log_mae": float(log_mae),
        "mae_aed": float(mae),
        "smape": float(smape),
        "wape": float(wape),
        "r2": float(r2)
    }

def save_json(path, obj):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2)

# ----------------------
# LOAD DATA + METADATA
# ----------------------
print("📥 Loading processed dataset and metadata...")
monthly = pd.read_parquet(PROCESSED_PATH)
with open(META_PATH) as fh:
    meta = json.load(fh)

FEATURES = meta.get("features", [
    "Cust_Avg","Prod_Avg","Geo_Avg","Cust_Prod_Avg","RECENT_AVG",
    "SALES_LAG_1","SALES_LAG_2","SALES_LAG_3",
    "SALES_ROLL_MEAN_3","SALES_ROLL_STD_3",
    "MONTH","Is_Summer","PROMO_SEASONALITY_INDEX",
    "PROMO_FREQ","PROMO_SUM","PROMO_SUM_OVER_PROD","CUST_PROD_OVER_PROD","RECENT_X_PROMO"
])

print(f"Loaded rows: {len(monthly):,}, segments: {monthly['SEGMENT'].nunique()}")

# train/test split by YEAR (same as Part1)
train_df = monthly[monthly["YEAR"] <= 2023].copy()
test_df  = monthly[monthly["YEAR"] >= 2024].copy()

print(f"Train rows total: {len(train_df):,}; Test rows total: {len(test_df):,}")

# ----------------------
# Prepare encoders to save
# ----------------------
encoders = {
    "cust_means": monthly.groupby("CUSTOMER")["SALES"].mean().to_dict(),
    "prod_means": monthly.groupby("PRODUCT_GROUP")["SALES"].mean().to_dict(),
    "geo_means": monthly.groupby("REGION")["SALES"].mean().to_dict(),
    "cust_prod_means": monthly.groupby(["CUSTOMER","PRODUCT_GROUP"])["SALES"].mean().to_dict(),
    "recent_means": monthly.groupby(["CUSTOMER","PRODUCT_GROUP"])["RECENT_AVG"].last().to_dict(),
    # promo encoders (if available)
}

# extract promo encoders if present
if {"PRODUCT_GROUP","MONTH","PROMO_SEASONALITY_INDEX"}.issubset(monthly.columns):
    promo_df = monthly[["PRODUCT_GROUP","MONTH","PROMO_SEASONALITY_INDEX","PROMO_FREQ","PROMO_SUM"]].drop_duplicates()
    promo_enc = {
        "seasonality": promo_df.set_index(["PRODUCT_GROUP","MONTH"])["PROMO_SEASONALITY_INDEX"].to_dict(),
        "freq": promo_df.set_index(["PRODUCT_GROUP","MONTH"])["PROMO_FREQ"].to_dict(),
        "sum": promo_df.set_index(["PRODUCT_GROUP","MONTH"])["PROMO_SUM"].to_dict()
    }
    encoders["promo_seasonality"] = promo_enc

# ----------------------
# Segment-wise training function
# ----------------------
def train_segment(segment_name: str, train_df_seg: pd.DataFrame, test_df_seg: pd.DataFrame, features: list):
    """
    Train appropriate model per segment and evaluate.
    Returns dictionary with model info, metrics, and predictions on test set.
    """
    result = {"segment": segment_name, "n_train": len(train_df_seg), "n_test": len(test_df_seg)}
    if len(train_df_seg) == 0:
        print(f"[WARN] No training rows for segment {segment_name}. Skipping.")
        return result

    # cap training rows (optional)
    if MAX_TRAIN_ROWS_PER_SEGMENT and len(train_df_seg) > MAX_TRAIN_ROWS_PER_SEGMENT:
        print(f"⚠️ Downsampling train set for segment {segment_name} from {len(train_df_seg):,} to {MAX_TRAIN_ROWS_PER_SEGMENT:,}")
        train_df_seg = train_df_seg.sample(n=MAX_TRAIN_ROWS_PER_SEGMENT, random_state=42)

    X_train = train_df_seg[features].copy()
    X_test  = test_df_seg[features].copy()
    # Fill NA and sanitize numeric columns
    X_train = X_train.fillna(0.0)
    X_test  = X_test.fillna(0.0)

    # targets
    y_train_raw = train_df_seg["SALES"].values
    y_test_raw  = test_df_seg["SALES"].values

    # choose model per segment
    if segment_name.lower() == "high" and lgb is not None:
        # LightGBM Tweedie on RAW SALES
        print(f"\n➡️ Training LightGBM Tweedie for HIGH segment (n_train={len(X_train):,})")
        lgb_train = lgb.Dataset(X_train, label=y_train_raw)
        lgb_valid = lgb.Dataset(X_test, label=y_test_raw, reference=lgb_train)
        params = {
            "objective": "tweedie",
            "metric": "mae",
            "learning_rate": 0.05,
            "num_leaves": 64,
            "min_data_in_leaf": 30,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "tweedie_variance_power": 1.3,
            "verbosity": -1,
        }
        # use callbacks (works across versions)
        callbacks = []
        if hasattr(lgb, "early_stopping"):
            callbacks.append(lgb.early_stopping(LGB_EARLY_STOP))
        if hasattr(lgb, "log_evaluation"):
            callbacks.append(lgb.log_evaluation(100))

        lgb_model = lgb.train(
            params,
            lgb_train,
            num_boost_round=LGB_NUM_BOOST_ROUND,
            valid_sets=[lgb_train, lgb_valid],
            valid_names=["train", "valid"],
            callbacks=callbacks
        )

        preds_raw = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration or None)
        # compute metrics by converting to log1p and reusing existing metric function
        metrics = compute_metrics_from_logs(np.log1p(y_test_raw), np.log1p(preds_raw))

        # save model artifacts
        txt_path = os.path.join(MODELS_DIR, f"task2_model_high_lgbm_tweedie.txt")
        lgb_model.save_model(txt_path)
        # joblib dump lgb booster (pickle-friendly)
        try:
            joblib.dump(lgb_model, os.path.join(MODELS_DIR, f"task2_model_high_lgbm_tweedie.pkl"))
        except Exception:
            # some lgb versions contain lambdas, but save text model above is the stable artifact
            pass

        result.update({
            "model_type": "lightgbm_tweedie",
            "model_path_txt": txt_path,
            "metrics": metrics,
            "preds_raw": preds_raw
        })
        return result

    # MEDIUM -> HGB on log target
    if segment_name.lower() == "medium":
        print(f"\n➡️ Training HistGradientBoosting (log-target) for MEDIUM segment (n_train={len(X_train):,})")
        y_train_log = np.log1p(y_train_raw)
        y_test_log = np.log1p(y_test_raw)
        hgb = HistGradientBoostingRegressor(
            learning_rate=0.08,
            max_depth=8,
            max_iter=500,
            random_state=42
        )
        hgb.fit(X_train, y_train_log)
        preds_log = hgb.predict(X_test)
        metrics = compute_metrics_from_logs(y_test_log, preds_log)

        joblib.dump(hgb, os.path.join(MODELS_DIR, "task2_model_medium_hgb.pkl"))

        result.update({
            "model_type": "hgb_log",
            "model_path": os.path.join(MODELS_DIR, "task2_model_medium_hgb.pkl"),
            "metrics": metrics,
            "preds_log": preds_log
        })
        return result

    # LOW -> Baseline: pair-level mean fallback, if not present use product mean, else global mean
    print(f"\n➡️ Creating baseline for LOW segment (n_train={len(X_train):,})")
    pair_means = train_df_seg.groupby(["CUSTOMER","PRODUCT_GROUP"])["SALES"].mean()
    prod_means = train_df_seg.groupby("PRODUCT_GROUP")["SALES"].mean()
    global_mean = train_df_seg["SALES"].mean()

    # Compose predictions: try pair mean -> product mean -> global mean
    preds = []
    for idx, row in test_df_seg.iterrows():
        key = (row["CUSTOMER"], row["PRODUCT_GROUP"])
        if key in pair_means:
            preds.append(pair_means[key])
        elif row["PRODUCT_GROUP"] in prod_means:
            preds.append(prod_means[row["PRODUCT_GROUP"]])
        else:
            preds.append(global_mean)
    preds = np.array(preds)
    # convert to log-scale for metrics
    metrics = compute_metrics_from_logs(np.log1p(y_test_raw), np.log1p(preds))

    # save baseline model artifact (just encoders)
    baseline_art = {
        "pair_means": pair_means.to_dict(),
        "prod_means": prod_means.to_dict(),
        "global_mean": float(global_mean)
    }
    joblib.dump(baseline_art, os.path.join(MODELS_DIR, "task2_model_low_baseline.pkl"))

    result.update({
        "model_type": "baseline_pair_mean",
        "model_path": os.path.join(MODELS_DIR, "task2_model_low_baseline.pkl"),
        "metrics": metrics,
        "preds_raw": preds
    })
    return result

# ----------------------
# Train for each segment
# ----------------------
segments = monthly["SEGMENT"].unique().tolist()
print("Segments found:", segments)

metrics_by_segment = {}
predictions_summary = []

for seg in segments:
    seg_train = train_df[train_df["SEGMENT"] == seg].copy()
    seg_test  = test_df[test_df["SEGMENT"] == seg].copy()

    seg_result = train_segment(seg, seg_train, seg_test, FEATURES)

    metrics_by_segment[seg] = {
        "n_train": seg_result.get("n_train"),
        "n_test": seg_result.get("n_test"),
        "model_type": seg_result.get("model_type"),
        "metrics": seg_result.get("metrics")
    }

    # assemble a CSV for predictions vs actuals for this segment
    if seg_result.get("preds_raw") is not None:
        preds = seg_result["preds_raw"]
        # ensure length matches test set
        if len(preds) == len(seg_test):
            out_df = seg_test[["CUSTOMER","PRODUCT_GROUP","PLOT_DATE","SALES"]].copy()
            out_df["PREDICTION"] = preds
            out_df.to_csv(os.path.join(RESULTS_DIR, f"prediction_vs_actual_{seg}.csv"), index=False)
        else:
            # some models store preds as log or other keys; attempt to align
            if seg_result.get("preds_log") is not None and len(seg_result["preds_log"]) == len(seg_test):
                out_df = seg_test[["CUSTOMER","PRODUCT_GROUP","PLOT_DATE","SALES"]].copy()
                out_df["PREDICTION"] = np.expm1(seg_result["preds_log"])
                out_df.to_csv(os.path.join(RESULTS_DIR, f"prediction_vs_actual_{seg}.csv"), index=False)
    elif seg_result.get("preds_log") is not None:
        preds = np.expm1(seg_result["preds_log"])
        out_df = seg_test[["CUSTOMER","PRODUCT_GROUP","PLOT_DATE","SALES"]].copy()
        out_df["PREDICTION"] = preds
        out_df.to_csv(os.path.join(RESULTS_DIR, f"prediction_vs_actual_{seg}.csv"), index=False)

# ----------------------
# Overall evaluation (merge segment predictions)
# ----------------------
print("\n📊 Aggregating overall metrics across segments...")
all_preds = []
all_actuals = []
for seg in segments:
    path = os.path.join(RESULTS_DIR, f"prediction_vs_actual_{seg}.csv")
    if os.path.exists(path):
        dfp = pd.read_csv(path)
        all_preds.extend(dfp["PREDICTION"].values.tolist())
        all_actuals.extend(dfp["SALES"].values.tolist())

if len(all_actuals) > 0:
    overall_metrics = compute_metrics_from_logs(np.log1p(np.array(all_actuals)), np.log1p(np.array(all_preds)))
else:
    overall_metrics = {}

# ----------------------
# Save artifacts: encoders & metrics
# ----------------------
encoders_path = os.path.join(MODELS_DIR, "task2_encoders_v8.pkl")
joblib.dump(encoders, encoders_path)
print(f"Saved encoders -> {encoders_path}")

metrics_path = os.path.join(MODELS_DIR, "task2_metrics_by_segment.json")
save_json(metrics_path, metrics_by_segment)
print(f"Saved metrics by segment -> {metrics_path}")

overall_path = os.path.join(MODELS_DIR, "task2_overall_metrics.json")
save_json(overall_path, overall_metrics)
print(f"Saved overall metrics -> {overall_path}")

print("\n✅ Part 2 complete — per-segment models trained & saved. Check results/ for prediction CSVs and models/ for artifacts.")
