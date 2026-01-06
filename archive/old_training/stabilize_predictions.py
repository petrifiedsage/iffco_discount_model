# stabilize_predictions.py
import os
import joblib
import glob
import math
from collections import defaultdict
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm

MODELS_DIR = r"D:/BlueRose/iffco_discount_model/models"
RESULTS_DIR = r"D:/BlueRose/iffco_discount_model/results"   # change if yours differs
HISTORY_PATH = os.path.join(MODELS_DIR, "history_monthly.pkl")  # produced by Part1/Part2

# =========================
# PARAMETERS (tweak these)
# =========================
SEGMENT_PARAMS = {
    "high":   {"alpha": 0.70, "clip_min": 0.5, "clip_max": 2.5, "rolling": True},
    "medium": {"alpha": 0.80, "clip_min": 0.6, "clip_max": 1.8, "rolling": True},
    "low":    {"alpha": 0.50, "clip_min": 0.3, "clip_max": 3.0, "rolling": False},
}

# small floor to avoid divide-by-zero for clipping
EPS_FLOOR = 1.0

# =========================
# HELPERS
# =========================
def load_history(history_path=HISTORY_PATH):
    """Load saved history_monthly.pkl (joblib) or csv fallback."""
    if os.path.exists(history_path):
        try:
            hist = joblib.load(history_path)
            # If saved as DataFrame, fine; if saved as dict adapt accordingly
            if isinstance(hist, pd.DataFrame):
                hdf = hist.copy()
            else:
                # if history saved as tuple/dict, try common keys
                raise ValueError("history_monthly.pkl content not DataFrame")
        except Exception:
            # try csv fallback
            fallback = os.path.join(MODELS_DIR, "history_monthly.csv")
            if os.path.exists(fallback):
                hdf = pd.read_csv(fallback, parse_dates=["Plot_Date"])
            else:
                raise
    else:
        raise FileNotFoundError(f"History not found at {history_path}")
    # normalize
    hdf = hdf.rename(columns={c: c.strip() for c in hdf.columns})
    if "Plot_Date" not in hdf.columns:
        # try other names
        for cand in ["PLOT_DATE", "Date", "PlotDate"]:
            if cand in hdf.columns:
                hdf = hdf.rename(columns={cand: "Plot_Date"})
                break
    hdf["Plot_Date"] = pd.to_datetime(hdf["Plot_Date"])
    hdf = hdf.sort_values(["CUSTOMER", "PRODUCT_GROUP", "Plot_Date"])
    return hdf

def build_history_lookup(history_df):
    """
    Build a dict: lookup[(cust_upper, prod_upper)] = pd.Series indexed by Plot_Date -> SALES
    Using upper-case keys for robust matching.
    """
    groups = {}
    for (cust, prod), g in history_df.groupby(["CUSTOMER", "PRODUCT_GROUP"]):
        key = (str(cust).strip().upper(), str(prod).strip().upper())
        s = g.set_index("Plot_Date")["SALES"].sort_index()
        groups[key] = s
    return groups

def compute_recent_stats_for_pair(series, pred_dates):
    """
    series: pandas Series indexed by Plot_Date -> sales (historical)
    pred_dates: iterable of pd.Timestamp (prediction months)
    returns lists: last_month_sales_list, recent_avg_list
    For each pred_date, we compute stats from months strictly before pred_date.
    """
    last_sales = []
    recent_avg = []
    if series is None or series.empty:
        # return zeros
        return [0.0]*len(pred_dates), [0.0]*len(pred_dates)

    # ensure sorted index
    idx = series.index
    vals = series.values
    for d in pred_dates:
        # want last month that is < d
        mask = idx < pd.Timestamp(d)
        if not mask.any():
            last_sales.append(0.0)
            recent_avg.append(0.0)
            continue
        filtered = series[mask]
        # last month's sales:
        last_val = float(filtered.iloc[-1])
        last_sales.append(last_val)
        # recent avg over last 3 months (if available)
        recent = filtered.tail(3)
        recent_avg.append(float(recent.mean()))
    return last_sales, recent_avg

# =========================
# CORE: Stabilize one DataFrame of predictions
# =========================
def stabilize_predictions_df(pred_df,
                             history_lookup,
                             segment_name="high",
                             model_col="Predicted",
                             date_col="Date",
                             cust_col="CUSTOMER",
                             prod_col="PRODUCT_GROUP",
                             save_path=None,
                             params=None):
    """
    pred_df: DataFrame with at least columns [date_col, cust_col, prod_col, model_col]
    history_lookup: dict from (CUP, PUP) -> Series of Plot_Date -> SALES
    params: dict with keys 'alpha', 'clip_min', 'clip_max', 'rolling'
    Returns stabilized DataFrame with new columns:
      - RECENT_AVG, LAST_MONTH_SALES, STABILIZED_PRED, CLIPPED
    """
    if params is None:
        params = SEGMENT_PARAMS.get(segment_name, {"alpha": 0.7, "clip_min": 0.5, "clip_max": 2.5, "rolling": True})

    alpha = float(params.get("alpha", 0.7))
    clip_min = float(params.get("clip_min", 0.5))
    clip_max = float(params.get("clip_max", 2.5))
    apply_rolling = bool(params.get("rolling", True))

    df = pred_df.copy()
    # normalize names
    df[date_col] = pd.to_datetime(df[date_col])
    df["__CUP__"] = df[cust_col].astype(str).str.strip().str.upper()
    df["__PUP__"] = df[prod_col].astype(str).str.strip().str.upper()

    # Compute product-level average fallback (from history) to use when last_month_sales==0
    prod_mean_lookup = {}
    for (c,p), s in history_lookup.items():
        if p not in prod_mean_lookup:
            prod_mean_lookup[p] = []
        prod_mean_lookup[p].append(s.mean())
    prod_mean = {p: (np.mean(lst) if len(lst)>0 else 0.0) for p,lst in prod_mean_lookup.items() for p in [p]}

    # Prepare containers
    RECENT = np.zeros(len(df), dtype=float)
    LAST = np.zeros(len(df), dtype=float)

    # iterate by group (faster than per-row lookups)
    grouped = df.groupby(["__CUP__", "__PUP__"])
    for (cup, pup), group in grouped:
        key = (cup, pup)
        pred_dates = list(group[date_col])
        hist_series = history_lookup.get(key, pd.Series(dtype=float))
        last_sales_list, recent_avg_list = compute_recent_stats_for_pair(hist_series, pred_dates)
        RECENT[group.index] = recent_avg_list
        LAST[group.index] = last_sales_list

    df["RECENT_AVG"] = RECENT
    df["LAST_MONTH_SALES"] = LAST

    # Fallbacks: if both recent and last are zero, use product mean or EPS_FLOOR
    df["FALLBACK_BASE"] = df.apply(
        lambda r: max(r["RECENT_AVG"], prod_mean.get(r["__PUP__"], 0.0), EPS_FLOOR), axis=1
    )

    # Compute blended (model + recent)
    df[model_col] = pd.to_numeric(df[model_col], errors="coerce").fillna(0.0)
    df["STABILIZED_PRED"] = alpha * df[model_col] + (1.0 - alpha) * df["RECENT_AVG"]

    # Clip range based on last month sales (or fallback)
    def apply_clip(row):
        baseline = row["LAST_MONTH_SALES"] if row["LAST_MONTH_SALES"] > 0 else row["FALLBACK_BASE"]
        min_v = clip_min * baseline
        max_v = clip_max * baseline
        return float(np.clip(row["STABILIZED_PRED"], min_v, max_v))

    df["STABILIZED_PRED_CLIPPED"] = df.apply(apply_clip, axis=1)

    # Optional rolling median smoothing per pair (on clipped stabilized preds)
    if apply_rolling:
        df = df.sort_values([cust_col, prod_col, date_col])
        smoothed = []
        for (cup, pup), g in df.groupby(["__CUP__", "__PUP__"], sort=False):
            vals = g["STABILIZED_PRED_CLIPPED"].values
            if len(vals) <= 2:
                smoothed.extend(vals)
            else:
                # rolling median window=3
                med = pd.Series(vals).rolling(window=3, min_periods=1, center=False).median().tolist()
                smoothed.extend(med)
        df["STABILIZED_FINAL"] = smoothed
    else:
        df["STABILIZED_FINAL"] = df["STABILIZED_PRED_CLIPPED"]

    # Expose some debug columns
    df["BLEND_ALPHA"] = alpha
    df["CLIP_MIN"] = clip_min
    df["CLIP_MAX"] = clip_max

    # tidy
    out_cols = list(pred_df.columns) + ["RECENT_AVG", "LAST_MONTH_SALES", "FALLBACK_BASE",
                                        "STABILIZED_PRED", "STABILIZED_PRED_CLIPPED", "STABILIZED_FINAL",
                                        "BLEND_ALPHA", "CLIP_MIN", "CLIP_MAX"]
    out = df[out_cols].copy()

    if save_path:
        out.to_csv(save_path, index=False)
    return out

# =========================
# BATCH: apply to all per-segment prediction files
# =========================
def run_batch_on_results(results_dir=RESULTS_DIR,
                         models_dir=MODELS_DIR,
                         pattern="prediction_*.csv",
                         output_dir=None):
    if output_dir is None:
        output_dir = os.path.join(results_dir, "stabilized")
    os.makedirs(output_dir, exist_ok=True)

    # load history
    history_df = load_history(os.path.join(models_dir, "history_monthly.pkl"))
    history_lookup = build_history_lookup(history_df)

    files = glob.glob(os.path.join(results_dir, pattern))
    if not files:
        # fallback single common file name
        fallback = os.path.join(results_dir, "prediction_vs_actual.csv")
        if os.path.exists(fallback):
            files = [fallback]
    if not files:
        raise FileNotFoundError(f"No prediction files found in {results_dir} matching {pattern}")

    for f in files:
        print("Processing:", f)
        df = pd.read_csv(f, parse_dates=True)
        # normalize col names for robustness
        cols = {c.strip().upper(): c for c in df.columns}
        # find necessary columns
        date_col = None
        for cand in ("DATE", "PLOT_DATE", "PRED_DATE"):
            if cand in cols:
                date_col = cols[cand]
                break
        if date_col is None:
            # try to infer by dtype
            for c in df.columns:
                if np.issubdtype(df[c].dtype, np.datetime64) or "DATE" in c.upper():
                    date_col = c
                    break
        if date_col is None:
            raise KeyError("Could not find date column in predictions file")

        cust_col = cols.get("CUSTOMER", cols.get("CUSTOMER_NAME", "CUSTOMER"))
        prod_col = cols.get("PRODUCT_GROUP", cols.get("PRODUCT", "PRODUCT_GROUP"))
        pred_col = cols.get("PREDICTED", cols.get("PREDICTION", cols.get("PREDICTED_AMOUNT", "Predicted")))

        # infer segment name from filename
        base = os.path.basename(f).lower()
        seg = "high"
        if "medium" in base:
            seg = "medium"
        elif "low" in base:
            seg = "low"

        out_path = os.path.join(output_dir, os.path.basename(f).replace(".csv", ".stabilized.csv"))
        stabilized = stabilize_predictions_df(
            df,
            history_lookup,
            segment_name=seg,
            model_col=pred_col,
            date_col=date_col,
            cust_col=cust_col,
            prod_col=prod_col,
            save_path=out_path,
            params=SEGMENT_PARAMS.get(seg)
        )
        print(" -> saved stabilized:", out_path)

# =========================
# Example: run it
# =========================
if __name__ == "__main__":
    # Adjust directories above if needed, then run:
    run_batch_on_results()
