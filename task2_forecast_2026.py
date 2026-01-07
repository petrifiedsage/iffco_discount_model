"""
Forecast 2026 claims by product group (Q1-Q4) using the corrected Task 2 model.
- Loads model, metadata, states, and seasonality.
- Forecasts Jan–Dec 2026 for every customer-product state.
- Saves:
    output/task2_forecast_2026_monthly.csv
    output/task2_forecast_2026_quarterly.csv
"""

from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "task2_reboot_model.pkl"
METADATA_PATH = MODEL_DIR / "task2_reboot_metadata.pkl"
STATES_PATH = MODEL_DIR / "task2_reboot_states.pkl"
SEASONALITY_PATH = MODEL_DIR / "product_seasonality_correct.csv"

START_YEAR = 2026
START_MONTH = 1
MONTHS_AHEAD = 12

print("Loading artifacts...")
model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
states = joblib.load(STATES_PATH)
df_seasonality = pd.read_csv(SEASONALITY_PATH)
print("✓ Model + metadata + states loaded")
print(f"  Feature count: {len(metadata['feature_cols'])}")
print(f"  Customer-product states: {len(states):,}")

feature_cols = metadata['feature_cols']


def forecast_customer_product(customer: str, product: str, state: dict) -> list:
    recent_history = np.array(state.get('recent_history', []), dtype=float)
    if recent_history.size == 0:
        return []

    # Lags
    current_lag_1 = recent_history[-1]
    current_lag_3 = recent_history[-3:].mean() if recent_history.size >= 3 else recent_history.mean()
    current_lag_6 = recent_history[-6:].mean() if recent_history.size >= 6 else recent_history.mean()
    current_lag_12 = recent_history[-12:].mean() if recent_history.size >= 12 else recent_history.mean()

    baseline = recent_history[-6:].mean() if recent_history.size >= 6 else recent_history.mean()

    # Seasonality slice for product
    product_seasonality = df_seasonality[df_seasonality['PRODUCT_GROUP'] == product]

    forecasts = []
    for step in range(MONTHS_AHEAD):
        month = ((START_MONTH + step - 1) % 12) + 1
        year = START_YEAR + ((START_MONTH + step - 1) // 12)
        quarter = ((month - 1) // 3) + 1
        is_summer = 1 if month in [6, 7, 8] else 0

        season_data = product_seasonality[product_seasonality['MONTH'] == month]
        seasonality_index = season_data['SEASONALITY_INDEX'].values[0] if len(season_data) else 1.0
        season_confidence = season_data['CONFIDENCE'].values[0] if len(season_data) else 0.5

        trend_3m = (current_lag_1 - current_lag_3) / 3 if current_lag_3 > 0 else 0

        features = {
            'LAG_1': current_lag_1,
            'LAG_3': current_lag_3,
            'LAG_6': current_lag_6,
            'LAG_12': current_lag_12,
            'LAG_3_AVG': current_lag_3,
            'LAG_6_AVG': current_lag_6,
            'CUSTOMER_AVG': state.get('customer_avg', current_lag_6),
            'PRODUCT_AVG': state.get('product_avg', state.get('customer_avg', current_lag_6)),
            'CUSTOMER_PRODUCT_AVG': state.get('customer_avg', current_lag_6),
            'TREND_3M': trend_3m,
            'PROMO_AMOUNT': 0.0,
            'PROMO_COUNT': 0.0,
            'PROMO_LOG': 0.0,
            'PROMO_COUNT_LOG': 0.0,
            'PROMO_3M_AVG': 0.0,
            'MONTH': month,
            'QUARTER': quarter,
            'YEAR': year,
            'IS_SUMMER': is_summer,
            'SEASONALITY_INDEX': seasonality_index,
        }

        X = pd.DataFrame([features])
        for col in feature_cols:
            if col not in X.columns:
                X[col] = 0.0
        X = X[feature_cols]

        y_pred_log = model.predict(X)[0]
        y_pred = np.expm1(y_pred_log)

        if season_confidence > 0.7:
            final = (y_pred * seasonality_index * 0.7) + (baseline * seasonality_index * 0.3)
        else:
            final = y_pred * (0.5 + 0.5 * seasonality_index)

        final = max(0.0, final)

        forecasts.append({
            'YEAR': year,
            'MONTH': month,
            'QUARTER': f"Q{quarter}",
            'PRODUCT_GROUP': product,
            'CUSTOMER': customer,
            'FORECAST': final,
            'SEASONALITY': seasonality_index,
        })

        current_lag_12 = current_lag_6
        current_lag_6 = current_lag_3
        current_lag_3 = (current_lag_3 * 2 + final) / 3
        current_lag_1 = final

    return forecasts


print("Generating forecasts for 2026 (12 months)...")
all_forecasts = []
for (customer, product), state in states.items():
    all_forecasts.extend(forecast_customer_product(customer, product, state))

if not all_forecasts:
    raise SystemExit("No forecasts generated. Check states and inputs.")

forecast_df = pd.DataFrame(all_forecasts)
forecast_df = forecast_df[forecast_df['YEAR'] == START_YEAR]
print(f"Total forecast rows: {len(forecast_df):,}")

monthly_out = OUTPUT_DIR / "task2_forecast_2026_monthly.csv"
forecast_df.to_csv(monthly_out, index=False)

quarterly = (
    forecast_df
    .groupby(['PRODUCT_GROUP', 'QUARTER'], as_index=False)['FORECAST']
    .sum()
    .pivot(index='PRODUCT_GROUP', columns='QUARTER', values='FORECAST')
    .reindex(columns=['Q1', 'Q2', 'Q3', 'Q4'], fill_value=0)
    .reset_index()
)

quarterly_out = OUTPUT_DIR / "task2_forecast_2026_quarterly.csv"
quarterly.to_csv(quarterly_out, index=False)

print("Saved:")
print(f"- {monthly_out}")
print(f"- {quarterly_out}")

print("\nSample (quarterly, first 5 rows):")
print(quarterly.head().to_string(index=False))

print("\nDone.")
