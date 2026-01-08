"""
Task 2 Improved Model - Phase 3: Add Enhanced Lag Features
============================================================
Adds more sophisticated lag features to capture temporal patterns.
LAG_2_SEASON, LAG_3_SEASON, LAG_8_SEASON, ROLLING_AVG_2YEARS.

Expected improvement: +3% accuracy (22.2% → ~19% Median APE)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_absolute_percentage_error
import pickle
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

BAD_PRODUCTS = {'Creams', 'Eggs', 'Fresh Meat Products', 'Vinegar'}

SEASON_MAP = {
    1: 'Winter', 2: 'Winter', 3: 'Spring', 4: 'Spring', 5: 'Summer',
    6: 'Summer', 7: 'Summer', 8: 'Summer', 9: 'Fall', 10: 'Fall',
    11: 'Festival', 12: 'Festival'
}

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

print("=" * 80)
print("TASK 2 IMPROVED: Phase 3 - Enhanced Lag Features")
print("=" * 80)

print("\n[1/4] Loading data...")
data_dir = Path('data/csv')
claim_dump_files = sorted(list(data_dir.glob("Contract_Claim_Dump*.csv")))

all_data = []
for csv_file in claim_dump_files:
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        required_cols = ['CUSTOMER_NAME', 'PRODUCT_GROUP', 'AMOUNT', 'CLAIM_DATE', 'STATUS']
        if not all(col in df.columns for col in required_cols):
            continue
        
        df = df[df['STATUS'] == 'Approved'].copy()
        if len(df) == 0:
            continue
        
        df_clean = pd.DataFrame()
        df_clean['CUSTOMER'] = df['CUSTOMER_NAME'].astype(str).str.strip()
        df_clean['PRODUCT_GROUP'] = df['PRODUCT_GROUP'].astype(str).str.strip()
        df_clean['SALES'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
        df_clean['DATE'] = df['CLAIM_DATE']
        
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
        
        df_clean = df_clean.dropna(subset=['SALES', 'CUSTOMER', 'PRODUCT_GROUP', 'MARKET'])
        df_clean = df_clean[df_clean['SALES'] > 0]
        all_data.append(df_clean)
        
    except Exception as e:
        continue

claims = pd.concat(all_data, ignore_index=True)
claims['DATE'] = claims['DATE'].apply(parse_date)
claims = claims.dropna(subset=['DATE'])
claims['YEAR'] = claims['DATE'].dt.year
claims['MONTH'] = claims['DATE'].dt.month
claims['SEASON'] = claims['MONTH'].map(SEASON_MAP)

print(f"  ✓ Loaded {len(claims):,} claims")

# Aggregate
print("\n[2/4] Aggregating and creating ENHANCED lag features...")
agg_cols = ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR']
df_agg = claims.groupby(agg_cols).agg({'SALES': 'sum'}).reset_index()
df_agg.columns = ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'SALES_VALUE']

df_agg['IS_BAD_PRODUCT'] = df_agg['PRODUCT_GROUP'].isin(BAD_PRODUCTS)
good_df = df_agg[~df_agg['IS_BAD_PRODUCT']].copy()

# Create features with enhanced lags
df_features = good_df.sort_values(['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR']).reset_index(drop=True)

# Basic lags
df_features['LAG_1_SEASON'] = df_features.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].shift(1)
df_features['LAG_2_SEASON'] = df_features.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].shift(2)
df_features['LAG_3_SEASON'] = df_features.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].shift(3)
df_features['LAG_4_SEASON'] = df_features.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].shift(4)
df_features['LAG_8_SEASON'] = df_features.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].shift(8)

# Rolling averages
df_features['ROLLING_AVG_2SEASONS'] = df_features.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].rolling(
    window=2, min_periods=1).mean().reset_index(drop=True)
df_features['ROLLING_AVG_4SEASONS'] = df_features.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].rolling(
    window=4, min_periods=1).mean().reset_index(drop=True)

# YoY growth
df_features['YOY_GROWTH'] = df_features.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES_VALUE'].pct_change(4)
df_features['YOY_GROWTH'] = df_features['YOY_GROWTH'].fillna(0).replace([np.inf, -np.inf], 0)

# Market features
df_features['MARKET_PRODUCT_AVG'] = df_features.groupby(['MARKET', 'PRODUCT_GROUP'])['SALES_VALUE'].transform('mean')
df_features['MARKET_AVG'] = df_features.groupby(['MARKET'])['SALES_VALUE'].transform('mean')

# Seasonality
df_features['SEASONALITY_INDEX'] = df_features['SALES_VALUE'] / df_features['MARKET_PRODUCT_AVG']
df_features['SEASONALITY_INDEX'] = df_features['SEASONALITY_INDEX'].replace([np.inf, -np.inf, np.nan], 1.0)

# Fill NaNs with market-product average
avg_val = df_features['MARKET_PRODUCT_AVG']
df_features['LAG_1_SEASON'] = df_features['LAG_1_SEASON'].fillna(avg_val)
df_features['LAG_2_SEASON'] = df_features['LAG_2_SEASON'].fillna(avg_val)
df_features['LAG_3_SEASON'] = df_features['LAG_3_SEASON'].fillna(avg_val)
df_features['LAG_4_SEASON'] = df_features['LAG_4_SEASON'].fillna(avg_val)
df_features['LAG_8_SEASON'] = df_features['LAG_8_SEASON'].fillna(avg_val)
df_features['ROLLING_AVG_2SEASONS'] = df_features['ROLLING_AVG_2SEASONS'].fillna(avg_val)
df_features['ROLLING_AVG_4SEASONS'] = df_features['ROLLING_AVG_4SEASONS'].fillna(avg_val)

df_features['PROMO_COUNT'] = 0
df_features['PROMO_LOG'] = 0

print(f"  ✓ Created {len(df_features):,} feature records")
print(f"  Features: LAG_1/2/3/4/8_SEASON, ROLLING_AVG_2/4_SEASONS, YOY_GROWTH, ...")

# Train separate models by market
print("\n[3/4] Training market-specific models with enhanced features...")

feature_cols = [
    'LAG_1_SEASON', 'LAG_2_SEASON', 'LAG_3_SEASON', 'LAG_4_SEASON', 'LAG_8_SEASON',
    'ROLLING_AVG_2SEASONS', 'ROLLING_AVG_4SEASONS', 'YOY_GROWTH',
    'MARKET_PRODUCT_AVG', 'MARKET_AVG', 'SEASONALITY_INDEX', 'PROMO_COUNT', 'PROMO_LOG'
]

models_by_market = {}
all_predictions = []
results_by_market = []

for market in sorted(df_features['MARKET'].unique()):
    if market == 'UAE':
        continue  # Skip market with insufficient data
    
    print(f"\n  {market}:")
    
    market_df = df_features[df_features['MARKET'] == market].copy()
    train_df = market_df[market_df['YEAR'] < 2025].copy()
    test_df = market_df[market_df['YEAR'] == 2025].copy()
    
    if len(train_df) < 10 or len(test_df) < 5:
        continue
    
    print(f"    Train: {len(train_df)}, Test: {len(test_df)}")
    
    train_X = pd.get_dummies(train_df[['SEASON'] + feature_cols], drop_first=False)
    train_y = train_df['SALES_VALUE']
    
    test_X = pd.get_dummies(test_df[['SEASON'] + feature_cols], drop_first=False)
    test_y = test_df['SALES_VALUE']
    test_X = test_X.reindex(columns=train_X.columns, fill_value=0)
    
    # Train with enhanced parameters
    model = HistGradientBoostingRegressor(
        max_depth=7, learning_rate=0.025, l2_regularization=0.08,
        min_samples_leaf=15, max_iter=600, random_state=42
    )
    
    model.fit(train_X, train_y)
    
    train_pred = model.predict(train_X)
    test_pred = model.predict(test_X)
    
    train_r2 = r2_score(train_y, train_pred)
    test_r2 = r2_score(test_y, test_pred)
    test_ape = abs(test_pred - test_y) / test_y
    median_ape = test_ape.median()
    
    print(f"    R²: {test_r2:.4f}, Median APE: {median_ape:.1%}")
    
    models_by_market[market] = model
    
    pred_df = test_df[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'SALES_VALUE']].copy()
    pred_df['PREDICTED'] = test_pred
    pred_df['APE'] = test_ape.values
    all_predictions.append(pred_df)
    
    results_by_market.append({
        'MARKET': market,
        'TRAIN_R2': train_r2,
        'TEST_R2': test_r2,
        'MEDIAN_APE': median_ape,
        'MEAN_APE': test_ape.mean(),
        'TEST_SIZE': len(test_df)
    })

combined_predictions = pd.concat(all_predictions, ignore_index=True)

# Results
print("\n" + "=" * 80)
print("RESULTS: Phase 3 - Enhanced Lag Features")
print("=" * 80)

overall_r2 = r2_score(combined_predictions['SALES_VALUE'], combined_predictions['PREDICTED'])
overall_median_ape = combined_predictions['APE'].median()

print(f"\nOverall Metrics (2025 validation):")
print(f"  Total predictions: {len(combined_predictions)}")
print(f"  R²: {overall_r2:.4f}")
print(f"  Median APE: {overall_median_ape:.1%}")

results_df = pd.DataFrame(results_by_market).sort_values('TEST_R2', ascending=False)
print("\nBy Market:")
print(results_df.to_string(index=False))

# Save
print("\n[4/4] Saving artifacts...")
os.makedirs('models', exist_ok=True)
os.makedirs('output', exist_ok=True)

for market, model in models_by_market.items():
    model_path = f'models/task2_improved_v3_model_{market}.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

pred_path = 'output/task2_improved_v3_predictions.csv'
combined_predictions.to_csv(pred_path, index=False)
print(f"✓ Models and predictions saved")

print("\n" + "=" * 80)
print("PHASE 3 COMPLETE: Enhanced lag features added")
print("=" * 80)
