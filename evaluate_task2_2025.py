"""
Evaluate Task 2 model on 2025 holdout: train window 2022-2024, predict 2025.
Outputs:
- output/task2_2025_pred_vs_actual.csv (row-level monthly per customer-product)
- output/task2_2025_summary_by_product.csv (aggregated per product group)
Prints overall metrics (R2, MAE, RMSE, Median APE).
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = Path("models/task2_reboot_model.pkl")
METADATA_PATH = Path("models/task2_reboot_metadata.pkl")
SEASONALITY_PATH = Path("models/product_seasonality_correct.csv")
PROMO_PATH = Path("results/tactic_promo_intensity.csv")

REQUIRED_COLS = ['CUSTOMER_NAME', 'PRODUCT_GROUP', 'AMOUNT', 'CLAIM_DATE', 'MONTH', 'STATUS']

def parse_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    text = str(date_str).strip()
    for fmt in ['%d-%m-%y', '%d-%m-%Y', '%d-%b-%y', '%d-%b-%Y', '%Y-%m-%d']:
        try:
            return pd.to_datetime(text, format=fmt)
        except Exception:
            continue
    try:
        return pd.to_datetime(text, dayfirst=True)
    except Exception:
        return pd.NaT


def load_claims():
    files = sorted(DATA_DIR.glob("Contract_Claim_Dump*.csv"))
    frames = []
    for path in files:
        df = pd.read_csv(path, low_memory=False)
        if not all(col in df.columns for col in REQUIRED_COLS):
            print(f"[SKIP] {path.name}: missing required columns")
            continue
        df = df[df['STATUS'] == 'Approved'].copy()
        if df.empty:
            print(f"[SKIP] {path.name}: no approved rows")
            continue
        df_clean = pd.DataFrame()
        df_clean['CUSTOMER'] = df['CUSTOMER_NAME'].astype(str).str.strip()
        df_clean['PRODUCT_GROUP'] = df['PRODUCT_GROUP'].astype(str).str.strip()
        df_clean['SALES'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
        df_clean['DATE'] = df['CLAIM_DATE'].apply(parse_date)
        df_clean['MONTH_NAME'] = df['MONTH'].astype(str).str.strip()
        df_clean = df_clean.dropna(subset=['CUSTOMER', 'PRODUCT_GROUP', 'SALES', 'DATE'])
        df_clean = df_clean[df_clean['SALES'] > 0]
        frames.append(df_clean)
        print(f"[OK] {path.name}: {len(df_clean):,} rows")
    if not frames:
        raise SystemExit("No valid claim data loaded")
    df_all = pd.concat(frames, ignore_index=True)
    df_all['YEAR_MONTH'] = df_all['DATE'].dt.to_period('M')
    return df_all


def add_promo(df):
    if not PROMO_PATH.exists():
        df['PROMO_AMOUNT'] = 0.0
        df['PROMO_COUNT'] = 0.0
        df['PROMO_LOG'] = 0.0
        df['PROMO_COUNT_LOG'] = 0.0
        df['PROMO_3M_AVG'] = 0.0
        return df
    promo = pd.read_csv(PROMO_PATH)
    if 'YEAR_MONTH' in promo.columns:
        promo['YEAR_MONTH'] = pd.PeriodIndex(promo['YEAR_MONTH'], freq='M')
    promo = promo.rename(columns={'promo_amount': 'PROMO_AMOUNT', 'promo_count': 'PROMO_COUNT'})
    df = df.merge(promo[['PRODUCT_GROUP', 'YEAR_MONTH', 'PROMO_AMOUNT', 'PROMO_COUNT']],
                  on=['PRODUCT_GROUP', 'YEAR_MONTH'], how='left')
    df['PROMO_AMOUNT'] = df['PROMO_AMOUNT'].fillna(0.0)
    df['PROMO_COUNT'] = df['PROMO_COUNT'].fillna(0.0)
    df['PROMO_LOG'] = np.log1p(df['PROMO_AMOUNT'])
    df['PROMO_COUNT_LOG'] = np.log1p(df['PROMO_COUNT'])
    df['PROMO_3M_AVG'] = df.groupby('PRODUCT_GROUP')['PROMO_AMOUNT'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean()
    ).fillna(0.0)
    return df


def build_features(df_monthly):
    df = df_monthly.copy()
    df = add_promo(df)

    for lag in [1, 3, 6, 12]:
        df[f'LAG_{lag}'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].shift(lag)

    df['LAG_3_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform(
        lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    df['LAG_6_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform(
        lambda x: x.shift(1).rolling(6, min_periods=1).mean())

    df['MONTH'] = df['MONTH_START'].dt.month
    df['QUARTER'] = df['MONTH_START'].dt.quarter
    df['YEAR'] = df['MONTH_START'].dt.year
    df['IS_SUMMER'] = df['MONTH'].isin([6, 7, 8]).astype(int)

    df['CUSTOMER_AVG'] = df.groupby('CUSTOMER')['SALES'].transform('mean')
    df['PRODUCT_AVG'] = df.groupby('PRODUCT_GROUP')['SALES'].transform('mean')
    df['CUSTOMER_PRODUCT_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform('mean')

    df['TREND_3M'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform(
        lambda x: (x.shift(1) - x.shift(4)) / 4).fillna(0)

    df = df[df['LAG_1'].notna()].copy()
    return df


def merge_seasonality(df):
    season_df = pd.read_csv(SEASONALITY_PATH)
    df = df.merge(season_df[['PRODUCT_GROUP', 'MONTH', 'SEASONALITY_INDEX', 'CONFIDENCE']],
                  on=['PRODUCT_GROUP', 'MONTH'], how='left')
    df['SEASONALITY_INDEX'] = df['SEASONALITY_INDEX'].fillna(1.0)
    df['CONFIDENCE'] = df['CONFIDENCE'].fillna(0.5)
    return df


def main():
    print("Loading data...")
    df_raw = load_claims()
    print(f"Total raw rows: {len(df_raw):,}")

    # Monthly aggregation
    df_monthly = df_raw.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'YEAR_MONTH']).agg({
        'SALES': 'sum',
        'MONTH_NAME': 'first'
    }).reset_index()
    df_monthly['MONTH_START'] = df_monthly['YEAR_MONTH'].dt.to_timestamp()
    df_monthly = df_monthly.sort_values(['CUSTOMER', 'PRODUCT_GROUP', 'MONTH_START'])
    print(f"Monthly rows: {len(df_monthly):,}")

    # Features
    df_feat = build_features(df_monthly)
    df_feat = merge_seasonality(df_feat)

    # Keep only 2025 for evaluation
    df_2025 = df_feat[df_feat['YEAR'] == 2025].copy()
    if df_2025.empty:
        raise SystemExit("No 2025 rows found after feature building")
    print(f"2025 evaluation rows: {len(df_2025):,}")

    # Load model + metadata
    model = joblib.load(MODEL_PATH)
    metadata = joblib.load(METADATA_PATH)
    feature_cols = metadata['feature_cols']

    # Align features
    X_test = df_2025[feature_cols].copy()
    y_true = df_2025['SALES']

    # Predict
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)

    # Metrics
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ape = np.abs((y_true - y_pred) / (y_true + 1)) * 100
    med_ape = float(np.median(ape))

    print("\nOverall metrics (2025 holdout):")
    print(f"  R2   : {r2:.4f}")
    print(f"  MAE  : ${mae:,.0f}")
    print(f"  RMSE : ${rmse:,.0f}")
    print(f"  Median APE: {med_ape:.1f}%")

    out_rows = df_2025[['CUSTOMER', 'PRODUCT_GROUP', 'YEAR_MONTH', 'MONTH_START']].copy()
    out_rows['ACTUAL'] = y_true.values
    out_rows['PREDICTED'] = y_pred
    out_rows['APE'] = ape.values

    pred_out = OUTPUT_DIR / "task2_2025_pred_vs_actual.csv"
    out_rows.to_csv(pred_out, index=False)

    summary = (
        out_rows
        .groupby('PRODUCT_GROUP')[['ACTUAL', 'PREDICTED']]
        .sum()
        .reset_index()
    )
    summary['APE'] = np.abs((summary['ACTUAL'] - summary['PREDICTED']) / (summary['ACTUAL'] + 1)) * 100
    summary_out = OUTPUT_DIR / "task2_2025_summary_by_product.csv"
    summary.to_csv(summary_out, index=False)

    print("\nSaved outputs:")
    print(f"- {pred_out}")
    print(f"- {summary_out}")

    # Show top 5 by absolute error
    top_err = out_rows.assign(ERR=np.abs(out_rows['ACTUAL'] - out_rows['PREDICTED']))
    print("\nLargest absolute errors (top 5):")
    print(top_err.nlargest(5, 'ERR')[['CUSTOMER', 'PRODUCT_GROUP', 'YEAR_MONTH', 'ACTUAL', 'PREDICTED', 'APE']])


if __name__ == "__main__":
    main()
