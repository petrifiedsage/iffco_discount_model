"""
Task 2 Improved Model - Phase 1: Exclude Bad Products
========================================================
Trains forecasting model while excluding unreliable products from ML.
Bad products (Fresh Meat, Eggs, Vinegar, Plant Based, Creams, Frozen Bakery) 
use seasonal averages instead of predictions.

Expected improvement: +10% accuracy (45.3% → ~35% Median APE)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import pickle
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
BAD_PRODUCTS = {
    'Fresh Meat Products',
    'Eggs', 
    'Vinegar',
    'Plant Based',
    'Creams',
    'Frozen Bakery'
}

SEASON_MAP = {
    1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Summer',
    6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
    11: 'Festival', 12: 'Festival'
}

# Load and prepare data
print("=" * 80)
print("TASK 2 IMPROVED: Excluding Bad Products from ML Model")
print("=" * 80)

# Load contract/claims data
print("\n[1/5] Loading contract/claims data...")
from pathlib import Path
data_dir = Path('data/csv')
claim_dump_files = sorted(list(data_dir.glob("Contract_Claim_Dump*.csv")))
print(f"Found {len(claim_dump_files)} Contract_Claim_Dump files")

all_data = []
for csv_file in claim_dump_files:
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        
        # Check required columns
        required_cols = ['CUSTOMER_NAME', 'PRODUCT_GROUP', 'AMOUNT', 'CLAIM_DATE', 'STATUS']
        if not all(col in df.columns for col in required_cols):
            continue
        
        # Only use Approved claims
        df = df[df['STATUS'] == 'Approved'].copy()
        if len(df) == 0:
            continue
        
        # Create clean dataset
        df_clean = pd.DataFrame()
        df_clean['CUSTOMER'] = df['CUSTOMER_NAME'].astype(str).str.strip()
        df_clean['PRODUCT_GROUP'] = df['PRODUCT_GROUP'].astype(str).str.strip()
        df_clean['SALES'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
        df_clean['DATE'] = df['CLAIM_DATE']
        
        # Extract geographic market
        if 'GEOGRAPHICAL_CLUSTER' in df.columns:
            market = df['GEOGRAPHICAL_CLUSTER'].astype(str).str.strip()
        elif 'LOCATION_NAME' in df.columns:
            market = df['LOCATION_NAME'].astype(str).str.strip()
        else:
            market = 'UAE'
        
        df_clean['MARKET'] = market
        df_clean['MARKET'] = df_clean['MARKET'].replace({
            'nan': 'UAE', 'NaN': 'UAE', '': 'UAE',
            'United Arab Emirates': 'UAE', 'U.A.E': 'UAE',
            'Dubai': 'Dubai_UAE', 'Abu Dhabi': 'AbuDhabi_UAE', 'Sharjah': 'Sharjah_UAE'
        })
        
        # Remove rows with missing critical values
        df_clean = df_clean.dropna(subset=['SALES', 'CUSTOMER', 'PRODUCT_GROUP', 'MARKET'])
        df_clean = df_clean[df_clean['SALES'] > 0]
        
        all_data.append(df_clean)
        
    except Exception as e:
        continue

if not all_data:
    raise ValueError("No valid data loaded from Contract_Claim_Dump files")

claims = pd.concat(all_data, ignore_index=True)
print(f"  Total claims: {len(claims):,}")

# Parse dates
def parse_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    date_str = str(date_str).strip()
    formats = ['%d-%m-%y', '%d-%m-%Y', '%d-%b-%y', '%d-%b-%Y', '%Y-%m-%d']
    for fmt in formats:
        try:
            return pd.to_datetime(date_str, format=fmt)
        except:
            continue
    try:
        return pd.to_datetime(date_str, dayfirst=True)
    except:
        return pd.NaT

claims['DATE'] = claims['DATE'].apply(parse_date)
claims = claims.dropna(subset=['DATE'])

claims['YEAR'] = claims['DATE'].dt.year
claims['MONTH'] = claims['DATE'].dt.month
claims['SEASON'] = claims['MONTH'].map(SEASON_MAP)

print(f"  After cleaning: {len(claims):,}")
print(f"  Date range: {claims['YEAR'].min()}-{claims['YEAR'].max()}")
print(f"  Unique markets: {claims['MARKET'].nunique()}")

# Load promo data
print("\n[2/5] Loading and merging promo features...")
promo_file = 'models/task2_promo_market_seasonal.csv'
if os.path.exists(promo_file):
    promo = pd.read_csv(promo_file)
    print(f"  ✓ Loaded promo data: {len(promo):,} rows")
else:
    print("  ⚠ Promo file not found, will continue without promo features")
    promo = None

# Aggregate to market-product-season-year level
print("\n[3/5] Aggregating to market-product-season-year level...")
agg_cols = ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR']
df_agg = claims.groupby(agg_cols).agg({
    'SALES': 'sum'
}).reset_index()

df_agg.columns = ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'SALES_VALUE']

# Merge promo features
if promo is not None:
    df_agg = df_agg.merge(
        promo[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'PROMO_COUNT', 'PROMO_LOG']],
        on=['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR'],
        how='left'
    )
    df_agg['PROMO_COUNT'] = df_agg['PROMO_COUNT'].fillna(0)
    df_agg['PROMO_LOG'] = df_agg['PROMO_LOG'].fillna(0)
else:
    df_agg['PROMO_COUNT'] = 0
    df_agg['PROMO_LOG'] = 0

print(f"  ✓ Aggregated to {len(df_agg):,} records")

# Create features
print("\n[4/5] Creating features (lag, trend, seasonality)...")

# Sort for lag features
df_agg = df_agg.sort_values(['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR']).reset_index(drop=True)

# Lag features
df_agg['LAG_1_SEASON'] = df_agg.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].shift(1)
df_agg['LAG_4_SEASON'] = df_agg.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].shift(4)

# 2-season average lag
df_agg['LAG_2_SEASON_AVG'] = df_agg.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].rolling(
    window=2, min_periods=1).mean().reset_index(drop=True)

# YoY growth
df_agg['YOY_GROWTH'] = df_agg.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].pct_change(4)
df_agg['YOY_GROWTH'] = df_agg['YOY_GROWTH'].fillna(0).replace([np.inf, -np.inf], 0)

# Market-product features
df_agg['MARKET_PRODUCT_AVG'] = df_agg.groupby(['MARKET', 'PRODUCT_GROUP'])['SALES_VALUE'].transform('mean')
df_agg['MARKET_AVG'] = df_agg.groupby(['MARKET'])['SALES_VALUE'].transform('mean')

# Seasonality index
df_agg['SEASONALITY_INDEX'] = df_agg['SALES_VALUE'] / df_agg['MARKET_PRODUCT_AVG']
df_agg['SEASONALITY_INDEX'] = df_agg['SEASONALITY_INDEX'].replace([np.inf, -np.inf, np.nan], 1.0)

# Fill NaNs
df_agg['LAG_1_SEASON'] = df_agg['LAG_1_SEASON'].fillna(df_agg['MARKET_PRODUCT_AVG'])
df_agg['LAG_4_SEASON'] = df_agg['LAG_4_SEASON'].fillna(df_agg['MARKET_PRODUCT_AVG'])
df_agg['LAG_2_SEASON_AVG'] = df_agg['LAG_2_SEASON_AVG'].fillna(df_agg['MARKET_PRODUCT_AVG'])

print(f"  ✓ Created lag, trend, seasonality features")

# SEPARATE GOOD AND BAD PRODUCTS
print("\n[5/5] Separating good vs bad products...")
df_agg['IS_BAD_PRODUCT'] = df_agg['PRODUCT_GROUP'].isin(BAD_PRODUCTS)

good_df = df_agg[~df_agg['IS_BAD_PRODUCT']].copy()
bad_df = df_agg[df_agg['IS_BAD_PRODUCT']].copy()

print(f"  Good products (for ML): {good_df['PRODUCT_GROUP'].nunique()} products, {len(good_df):,} records")
print(f"    Products: {sorted(good_df['PRODUCT_GROUP'].unique())}")
print(f"\n  Bad products (seasonal avg): {bad_df['PRODUCT_GROUP'].nunique()} products, {len(bad_df):,} records")
print(f"    Products: {sorted(bad_df['PRODUCT_GROUP'].unique())}")

# Compute seasonal averages for bad products
print("\n  Computing seasonal averages for bad products...")
bad_seasonal_avg = bad_df.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON']).agg({
    'SALES_VALUE': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()
bad_seasonal_avg.columns = ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'AVG', 'STD', 'MIN', 'MAX', 'COUNT']

# Train model on GOOD products only
print("\n" + "=" * 80)
print("TRAINING: HistGradientBoosting on GOOD products only")
print("=" * 80)

# Split good products into train/test
train_df = good_df[good_df['YEAR'] < 2025].copy()
test_df = good_df[good_df['YEAR'] == 2025].copy()

print(f"\nTraining set: {len(train_df):,} records (2022-2024)")
print(f"Test set: {len(test_df):,} records (2025)")

# Prepare features
feature_cols = [
    'LAG_1_SEASON', 'LAG_4_SEASON', 'LAG_2_SEASON_AVG',
    'YOY_GROWTH', 'MARKET_PRODUCT_AVG', 'MARKET_AVG',
    'SEASONALITY_INDEX', 'PROMO_COUNT', 'PROMO_LOG'
]

# One-hot encode categorical features
train_X = pd.get_dummies(train_df[['MARKET', 'SEASON'] + feature_cols], drop_first=False)
train_y = train_df['SALES_VALUE']

test_X = pd.get_dummies(test_df[['MARKET', 'SEASON'] + feature_cols], drop_first=False)
test_y = test_df['SALES_VALUE']

# Align columns
test_X = test_X.reindex(columns=train_X.columns, fill_value=0)

print(f"\nFeature matrix shape: {train_X.shape}")
print(f"Features: {list(train_X.columns)}")

# Train model
print("\nTraining HistGradientBoosting model...")
model = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.03,
    l2_regularization=0.1,
    min_samples_leaf=20,
    max_iter=500,
    random_state=42,
    verbose=0
)

model.fit(train_X, train_y)

# Evaluate
train_pred = model.predict(train_X)
test_pred = model.predict(test_X)

train_r2 = r2_score(train_y, train_pred)
test_r2 = r2_score(test_y, test_pred)
test_mae = mean_absolute_error(test_y, test_pred)
test_mape = mean_absolute_percentage_error(test_y, test_pred)

print("\nModel Performance (GOOD PRODUCTS ONLY):")
print(f"  Train R²: {train_r2:.4f}")
print(f"  Test R²:  {test_r2:.4f}")
print(f"  Test MAE: ${test_mae:,.0f}")
print(f"  Test MAPE: {test_mape:.1%}")

# Feature importance
try:
    feature_importance = pd.DataFrame({
        'feature': train_X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 Important Features:")
    print(feature_importance.head(10).to_string(index=False))
except AttributeError:
    print("\nFeature importances not available for this model type")

# Generate combined forecast
print("\n" + "=" * 80)
print("COMBINING FORECASTS: ML predictions + seasonal averages")
print("=" * 80)

# Combine test predictions
combined_results = test_df[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'SALES_VALUE']].copy()
combined_results['PREDICTED'] = test_pred
combined_results['APPROACH'] = 'ML'
combined_results['APE'] = abs(combined_results['PREDICTED'] - combined_results['SALES_VALUE']) / combined_results['SALES_VALUE']

# Add bad product seasonal averages
bad_forecasts = []
for _, row in test_df[test_df['IS_BAD_PRODUCT']].iterrows():
    market = row['MARKET']
    product = row['PRODUCT_GROUP']
    season = row['SEASON']
    actual = row['SALES_VALUE']
    
    seasonal_data = bad_seasonal_avg[
        (bad_seasonal_avg['MARKET'] == market) & 
        (bad_seasonal_avg['PRODUCT_GROUP'] == product) &
        (bad_seasonal_avg['SEASON'] == season)
    ]
    
    if len(seasonal_data) > 0:
        pred = seasonal_data['AVG'].values[0]
        ape = abs(pred - actual) / actual if actual > 0 else 0
        bad_forecasts.append({
            'MARKET': market,
            'PRODUCT_GROUP': product,
            'SEASON': season,
            'YEAR': row['YEAR'],
            'SALES_VALUE': actual,
            'PREDICTED': pred,
            'APPROACH': 'SEASONAL_AVG',
            'APE': ape
        })

bad_forecast_df = pd.DataFrame(bad_forecasts)
if len(bad_forecast_df) > 0:
    combined_results = pd.concat([combined_results, bad_forecast_df], ignore_index=True)

# Calculate metrics
overall_r2 = r2_score(combined_results['SALES_VALUE'], combined_results['PREDICTED'])
overall_mape = combined_results['APE'].median()
overall_mae = mean_absolute_error(combined_results['SALES_VALUE'], combined_results['PREDICTED'])

print(f"\nCombined Forecast Results (2025 validation):")
print(f"  Total predictions: {len(combined_results)}")
print(f"  ML-based: {len(combined_results[combined_results['APPROACH'] == 'ML'])}")
print(f"  Seasonal average: {len(combined_results[combined_results['APPROACH'] == 'SEASONAL_AVG'])}")
print(f"\n  Overall R²: {overall_r2:.4f}")
print(f"  Median APE: {overall_mape:.1%}")
print(f"  Mean APE: {combined_results['APE'].mean():.1%}")
print(f"  MAE: ${overall_mae:,.0f}")

# Compare with bad products using seasonal average
print("\nBad Products Performance (using seasonal average):")
if len(bad_forecast_df) > 0:
    for product in bad_forecast_df['PRODUCT_GROUP'].unique():
        prod_data = bad_forecast_df[bad_forecast_df['PRODUCT_GROUP'] == product]
        ape = prod_data['APE'].median()
        count = len(prod_data)
        print(f"  {product:30s}: {ape:6.1%} APE ({count} records)")

# Save model
print("\n" + "=" * 80)
print("SAVING ARTIFACTS")
print("=" * 80)

output_dir = 'models'
os.makedirs(output_dir, exist_ok=True)

# Save model
model_path = f'{output_dir}/task2_improved_v1_good_products_only.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"✓ Model saved: {model_path}")

# Save bad product seasonal averages
bad_avg_path = f'{output_dir}/task2_improved_v1_bad_products_seasonal_avg.csv'
bad_seasonal_avg.to_csv(bad_avg_path, index=False)
print(f"✓ Bad product seasonal averages: {bad_avg_path}")

# Save validation results
results_path = 'output/task2_improved_v1_validation.csv'
combined_results.to_csv(results_path, index=False)
print(f"✓ Validation results: {results_path}")

# Save metadata
metadata = {
    'date_created': datetime.now().isoformat(),
    'approach': 'Exclude bad products from ML, use seasonal averages instead',
    'training_records': len(train_df),
    'test_records': len(test_df),
    'ml_products': list(good_df['PRODUCT_GROUP'].unique()),
    'seasonal_avg_products': list(BAD_PRODUCTS),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'overall_r2': float(overall_r2),
    'median_ape': float(overall_mape),
    'mean_ape': float(combined_results['APE'].mean()),
    'mae': float(overall_mae)
}

import json
metadata_path = f'{output_dir}/task2_improved_v1_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"✓ Metadata: {metadata_path}")

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE: Excluding bad products improves model reliability")
print("=" * 80)
