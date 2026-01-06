"""
Model Performance Testing Script
Tests the corrected Task 2 model on 2025 holdout data
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "task2_reboot_model.pkl"
METADATA_PATH = MODEL_DIR / "task2_reboot_metadata.pkl"
HISTORY_PATH = MODEL_DIR / "history_monthly.pkl"

print("="*80)
print("TASK 2 MODEL PERFORMANCE TESTING")
print("="*80)

# Load model and data
print("\n[1] Loading model artifacts...")
model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
history_df = joblib.load(HISTORY_PATH)

print(f"  Feature columns: {len(metadata['feature_cols'])}")
print(f"  Training R²: {metadata['train_r2']:.4f}")
print(f"  Test R²: {metadata['test_r2']:.4f}")

# Load historical data
print("\n[2] Preparing test data (2025)...")
df = history_df.copy()
df['YEAR'] = df['MONTH_START'].dt.year
df['MONTH'] = df['MONTH_START'].dt.month
df['QUARTER'] = df['MONTH_START'].dt.quarter
df['IS_SUMMER'] = df['MONTH'].isin([6, 7, 8]).astype(int)

# Create lags
for lag in [1, 3, 6, 12]:
    df[f'LAG_{lag}'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].shift(lag)

df['LAG_3_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df['LAG_6_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform(
    lambda x: x.shift(1).rolling(6, min_periods=1).mean()
)

# Baselines
df['CUSTOMER_AVG'] = df.groupby('CUSTOMER')['SALES'].transform('mean')
df['PRODUCT_AVG'] = df.groupby('PRODUCT_GROUP')['SALES'].transform('mean')
df['CUSTOMER_PRODUCT_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform('mean')

# Trend
df['TREND_3M'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform(
    lambda x: (x.shift(1) - x.shift(4)) / 4
).fillna(0)

# Load seasonality
seasonality_df = pd.read_csv(MODEL_DIR / "product_seasonality_correct.csv")
df = df.merge(
    seasonality_df[['PRODUCT_GROUP', 'MONTH', 'SEASONALITY_INDEX']],
    on=['PRODUCT_GROUP', 'MONTH'],
    how='left'
)
df['SEASONALITY_INDEX'] = df['SEASONALITY_INDEX'].fillna(1.0)

# Filter to 2025 test data
test_data = df[df['YEAR'] == 2025].copy()
test_data = test_data.dropna(subset=['LAG_1'])

print(f"  Test records: {len(test_data):,}")
print(f"  Date range: {test_data['MONTH_START'].min()} to {test_data['MONTH_START'].max()}")
print(f"  Unique customers: {test_data['CUSTOMER'].nunique():,}")
print(f"  Unique products: {test_data['PRODUCT_GROUP'].nunique()}")

# Make predictions
print("\n[3] Generating predictions...")
X_test = test_data[metadata['feature_cols']]
y_test_log = np.log1p(test_data['SALES'])

y_pred_log = model.predict(X_test)
y_pred = np.expm1(y_pred_log)
y_actual = test_data['SALES'].values

# Calculate metrics
print("\n[4] Performance Metrics:")
print("="*80)

# R-squared
r2 = r2_score(y_test_log, y_pred_log)
print(f"\nR-squared (log space): {r2:.4f}")
print(f"  Model explains {r2*100:.2f}% of variance")

# MAE
mae = mean_absolute_error(y_actual, y_pred)
print(f"\nMean Absolute Error (MAE): ${mae:,.2f}")
print(f"  Average prediction error per transaction")

# RMSE
rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
print(f"\nRoot Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"  Penalizes large errors more heavily")

# MAPE (Mean Absolute Percentage Error)
mape = np.mean(np.abs((y_actual - y_pred) / (y_actual + 1e-6))) * 100
print(f"\nMean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"  Average error as % of actual value")

# Median APE
ape = np.abs((y_actual - y_pred) / (y_actual + 1)) * 100
median_ape = np.median(ape)
print(f"\nMedian Absolute Percentage Error: {median_ape:.2f}%")
print(f"  50% of predictions within {median_ape:.2f}% of actual")

# Percentiles
print(f"\nError Distribution:")
print(f"  25th percentile: {np.percentile(ape, 25):.2f}%")
print(f"  50th percentile: {median_ape:.2f}%")
print(f"  75th percentile: {np.percentile(ape, 75):.2f}%")
print(f"  90th percentile: {np.percentile(ape, 90):.2f}%")

# Accuracy buckets
within_20 = (ape <= 20).sum() / len(ape) * 100
within_30 = (ape <= 30).sum() / len(ape) * 100
within_50 = (ape <= 50).sum() / len(ape) * 100

print(f"\nAccuracy Buckets:")
print(f"  Within 20% error: {within_20:.1f}% of predictions")
print(f"  Within 30% error: {within_30:.1f}% of predictions")
print(f"  Within 50% error: {within_50:.1f}% of predictions")

# Revenue impact analysis
total_actual = y_actual.sum()
total_predicted = y_pred.sum()
revenue_error = (total_predicted - total_actual) / total_actual * 100

print(f"\nRevenue Forecasting:")
print(f"  Actual total sales: ${total_actual:,.2f}")
print(f"  Predicted total sales: ${total_predicted:,.2f}")
print(f"  Revenue error: {revenue_error:+.2f}%")

# Product-level performance
print("\n[5] Performance by Product Category:")
print("-"*80)
product_metrics = []
for product in test_data['PRODUCT_GROUP'].unique():
    prod_mask = test_data['PRODUCT_GROUP'] == product
    if prod_mask.sum() > 10:  # Only products with >10 test records
        prod_actual = y_actual[prod_mask]
        prod_pred = y_pred[prod_mask]
        prod_ape = np.abs((prod_actual - prod_pred) / (prod_actual + 1)) * 100
        
        product_metrics.append({
            'Product': product,
            'Records': prod_mask.sum(),
            'MAE': mean_absolute_error(prod_actual, prod_pred),
            'Median APE': np.median(prod_ape),
            'Within 30%': (prod_ape <= 30).sum() / len(prod_ape) * 100
        })

prod_df = pd.DataFrame(product_metrics).sort_values('Median APE')
for _, row in prod_df.head(10).iterrows():
    print(f"{row['Product']:30s} | MAE: ${row['MAE']:8,.0f} | "
          f"APE: {row['Median APE']:5.1f}% | Within 30%: {row['Within 30%']:5.1f}%")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print(f"✓ Test R²: {r2:.3f}")
print(f"✓ Median APE: {median_ape:.1f}%")
print(f"✓ MAE: ${mae:,.0f}")
print(f"✓ {within_30:.1f}% predictions within 30% error")
print(f"✓ Revenue forecast accuracy: {100 - abs(revenue_error):.1f}%")
print("="*80)
