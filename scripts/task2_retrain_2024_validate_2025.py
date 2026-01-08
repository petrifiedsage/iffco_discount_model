"""
TASK 2: Retrain model using Contract_Claim_Dump (2022-2024), validate on 2025
Direct approach: Load raw Contract_Claim_Dump, aggregate by Market-Product-Season
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import joblib

warnings.filterwarnings('ignore')

print("=" * 80)
print("TASK 2: SALES FORECASTING - RETRAIN 2022-2024, VALIDATE ON 2025")
print("=" * 80)

DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("output")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

SEASON_MAP = {
    1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring',
    5: 'Summer', 6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall',
    11: 'Festival', 12: 'Festival'
}

MONTH_NAMES = {
    'january': 1, 'jan': 1,
    'february': 2, 'feb': 2,
    'march': 3, 'mar': 3,
    'april': 4, 'apr': 4,
    'may': 5,
    'june': 6, 'jun': 6,
    'july': 7, 'jul': 7,
    'august': 8, 'aug': 8,
    'september': 9, 'sep': 9, 'sept': 9,
    'october': 10, 'oct': 10,
    'november': 11, 'nov': 11,
    'december': 12, 'dec': 12
}

# ============================================================================
# STEP 1: Load Contract_Claim_Dump files
# ============================================================================
print("\n[STEP 1] Loading Contract_Claim_Dump files...")

train_records = []
test_records = []

# Get all files
all_files = sorted(list(DATA_DIR.glob("Contract_Claim_Dump*.csv")))
print(f"Found {len(all_files)} files total")

for csv_file in all_files:
    filename = csv_file.name
    
    # Determine year
    if "Jan22-Dec22" in filename:
        year = 2022
        target = train_records
    elif "Jan23-Dec23" in filename:
        year = 2023
        target = train_records
    elif "Jan24-Dec24" in filename:
        year = 2024
        target = train_records
    elif "Jan25-Nov25" in filename:
        year = 2025
        target = test_records
    else:
        continue
    
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Filter approved claims only
        if 'STATUS' not in df.columns:
            continue
        df = df[df['STATUS'] == 'Approved'].copy()
        if len(df) == 0:
            continue
        
        # Extract needed columns
        before_clean = len(df)
        df['YEAR'] = year
        # Convert MONTH from text to numeric
        df['MONTH'] = df['MONTH'].astype(str).str.lower().map(MONTH_NAMES)
        df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
        
        # Remove null values
        df = df.dropna(subset=['MONTH', 'AMOUNT', 'PRODUCT_GROUP'])
        df = df[df['AMOUNT'] > 0]
        
        if len(df) == 0:
            print(f"    WARNING: {filename} lost all rows during cleaning (from {before_clean})")
            continue
        
        # Add season
        df['SEASON'] = df['MONTH'].map(SEASON_MAP)
        df_no_season = df[df['SEASON'].isna()].shape[0]
        if df_no_season > 0:
            print(f"    WARNING: {filename} has {df_no_season} rows with unmapped seasons")
        df = df.dropna(subset=['SEASON'])
        
        # Extract market
        df['MARKET'] = df['GEOGRAPHICAL_CLUSTER'].astype(str).str.strip().str.upper()
        df['MARKET'] = df['MARKET'].where(df['MARKET'].isin(['ROUAE', 'ADH_AIN']), 'ROUAE')
        
        # Keep relevant columns
        df_clean = df[['YEAR', 'MONTH', 'SEASON', 'MARKET', 'PRODUCT_GROUP', 'AMOUNT']].copy()
        df_clean['PRODUCT_GROUP'] = df_clean['PRODUCT_GROUP'].astype(str).str.strip()
        
        target.append(df_clean)
        print(f"  [OK] {filename}: {len(df_clean):,} rows")
        
    except Exception as e:
        print(f"  [FAIL] {filename}: {str(e)[:60]}")

train_df = pd.concat(train_records, ignore_index=True) if train_records else pd.DataFrame()
test_df = pd.concat(test_records, ignore_index=True) if test_records else pd.DataFrame()

print(f"\n  Training data (2022-2024): {len(train_df):,} rows")
print(f"  Test data (2025): {len(test_df):,} rows")

if len(train_df) == 0 or len(test_df) == 0:
    print("ERROR: No training or test data loaded!")
    exit(1)

# ============================================================================
# STEP 2: Aggregate by Market-Product-Season
# ============================================================================
print("\n[STEP 2] Aggregating data...")

train_agg = train_df.groupby(
    ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR'], 
    as_index=False
).agg({
    'AMOUNT': ['sum', 'count', 'mean', 'std']
})
train_agg.columns = ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 
                     'TOTAL_SALES', 'CLAIM_COUNT', 'MEAN_SALE', 'STD_SALE']
train_agg['STD_SALE'] = train_agg['STD_SALE'].fillna(0)

test_agg = test_df.groupby(
    ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR'],
    as_index=False
).agg({
    'AMOUNT': ['sum', 'count', 'mean', 'std']
})
test_agg.columns = ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR',
                    'TOTAL_SALES', 'CLAIM_COUNT', 'MEAN_SALE', 'STD_SALE']
test_agg['STD_SALE'] = test_agg['STD_SALE'].fillna(0)

print(f"  Training aggregates: {len(train_agg)} groups ({train_agg['PRODUCT_GROUP'].nunique()} products)")
print(f"  Test aggregates: {len(test_agg)} groups ({test_agg['PRODUCT_GROUP'].nunique()} products)")

# ============================================================================
# STEP 3: Feature Engineering & Encoding
# ============================================================================
print("\n[STEP 3] Feature engineering...")

market_enc = LabelEncoder()
season_enc = LabelEncoder()
product_enc = LabelEncoder()

# Fit on training data
train_agg['market_code'] = market_enc.fit_transform(train_agg['MARKET'])
train_agg['season_code'] = season_enc.fit_transform(train_agg['SEASON'])
train_agg['product_code'] = product_enc.fit_transform(train_agg['PRODUCT_GROUP'])

# Transform test data
test_agg['market_code'] = market_enc.transform(test_agg['MARKET'])
test_agg['season_code'] = season_enc.transform(test_agg['SEASON'])
test_agg['product_code'] = product_enc.transform(test_agg['PRODUCT_GROUP'])

# Save encoders
joblib.dump(market_enc, MODEL_DIR / 'market_encoder_retrained.pkl')
joblib.dump(season_enc, MODEL_DIR / 'season_encoder_retrained.pkl')
joblib.dump(product_enc, MODEL_DIR / 'product_encoder_retrained.pkl')

# Prepare features
feature_cols = ['market_code', 'product_code', 'season_code', 'YEAR', 'CLAIM_COUNT', 'MEAN_SALE', 'STD_SALE']
X_train = train_agg[feature_cols].copy()
y_train = train_agg['TOTAL_SALES'].copy()

X_test = test_agg[feature_cols].copy()
y_test = test_agg['TOTAL_SALES'].copy()

print(f"  Training features: {X_train.shape}")
print(f"  Test features: {X_test.shape}")

# ============================================================================
# STEP 4: Train Model
# ============================================================================
print("\n[STEP 4] Training Gradient Boosting model...")

model = GradientBoostingRegressor(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    verbose=0
)

model.fit(X_train, y_train)
print("  [OK] Model trained successfully")

# Save model
joblib.dump(model, MODEL_DIR / 'task2_retrained_model.pkl')

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': feature_cols,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
print("\n  Feature Importance:")
print(feature_importance.to_string(index=False))

# ============================================================================
# STEP 5: Validation
# ============================================================================
print("\n[STEP 5] Validation on 2025 test data...")

y_pred = model.predict(X_test)
test_agg['PREDICTED_SALES'] = y_pred

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = mean_absolute_percentage_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n  Overall Metrics:")
print(f"    MAE:  ${mae:,.0f}")
print(f"    RMSE: ${rmse:,.0f}")
print(f"    MAPE: {mape:.2%}")
print(f"    R²:   {r2:.4f}")

# APE per sample
test_agg['APE'] = (
    np.abs(test_agg['PREDICTED_SALES'] - test_agg['TOTAL_SALES']) 
    / test_agg['TOTAL_SALES'] * 100
).round(2)

print(f"\n  APE Distribution:")
print(f"    Median: {test_agg['APE'].median():.2f}%")
print(f"    Mean:   {test_agg['APE'].mean():.2f}%")
print(f"    Std:    {test_agg['APE'].std():.2f}%")
print(f"    Min:    {test_agg['APE'].min():.2f}%")
print(f"    Max:    {test_agg['APE'].max():.2f}%")

# Product-level performance
print(f"\n  Top 5 Best Products (Lowest Median APE):")
prod_ape = test_agg.groupby('PRODUCT_GROUP')['APE'].median().nsmallest(5)
for prod, ape in prod_ape.items():
    cnt = len(test_agg[test_agg['PRODUCT_GROUP'] == prod])
    print(f"    {prod:30s}: {ape:6.2f}% ({cnt} forecasts)")

print(f"\n  Top 5 Worst Products (Highest Median APE):")
prod_ape_worst = test_agg.groupby('PRODUCT_GROUP')['APE'].median().nlargest(5)
for prod, ape in prod_ape_worst.items():
    cnt = len(test_agg[test_agg['PRODUCT_GROUP'] == prod])
    print(f"    {prod:30s}: {ape:6.2f}% ({cnt} forecasts)")

# ============================================================================
# STEP 6: Save Results
# ============================================================================
print("\n[STEP 6] Saving results...")

# Validation output (combine across MARKET)
val_combined = test_agg[[  
    'PRODUCT_GROUP', 'SEASON', 'MARKET', 'TOTAL_SALES', 'PREDICTED_SALES'
]].copy()

# Aggregate across markets
val_combined = val_combined.groupby(
    ['PRODUCT_GROUP', 'SEASON'], as_index=False
).agg({
    'TOTAL_SALES': 'sum',
    'PREDICTED_SALES': 'sum'
})

# Recalculate APE on combined values
val_combined['APE'] = (
    np.abs(val_combined['PREDICTED_SALES'] - val_combined['TOTAL_SALES']) 
    / val_combined['TOTAL_SALES'] * 100
).round(2)

# Sort
season_order = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3, 'Festival': 4}
val_combined['season_sort'] = val_combined['SEASON'].map(season_order)
val_combined = val_combined.sort_values(['PRODUCT_GROUP', 'season_sort']).drop('season_sort', axis=1)
val_combined = val_combined[['PRODUCT_GROUP', 'SEASON', 'TOTAL_SALES', 'PREDICTED_SALES', 'APE']]

val_path = OUTPUT_DIR / 'task2_2025_validation_retrained.csv'
val_combined.to_csv(val_path, index=False)
print(f"  [OK] Validation results: {val_path}")

# Combined forecast (combine ROUAE + ADH_AIN)
products = train_agg['PRODUCT_GROUP'].unique()
seasons = ['Winter', 'Spring', 'Summer', 'Fall', 'Festival']
markets = ['ROUAE', 'ADH_AIN']

forecast_list = []
for market in markets:
    for product in products:
        for season in seasons:
            # Get most recent year's stats (2024)
            hist = train_agg[
                (train_agg['MARKET'] == market) &
                (train_agg['PRODUCT_GROUP'] == product) &
                (train_agg['SEASON'] == season) &
                (train_agg['YEAR'] == 2024)
            ]
            
            if len(hist) == 0:
                # Use average across all years if 2024 not available
                hist = train_agg[
                    (train_agg['MARKET'] == market) &
                    (train_agg['PRODUCT_GROUP'] == product) &
                    (train_agg['SEASON'] == season)
                ]
            
            if len(hist) == 0:
                continue
            
            forecast_list.append({
                'MARKET': market,
                'PRODUCT_GROUP': product,
                'SEASON': season,
                'CLAIM_COUNT': hist['CLAIM_COUNT'].mean(),
                'MEAN_SALE': hist['MEAN_SALE'].mean(),
                'STD_SALE': hist['STD_SALE'].mean(),
            })

forecast_df = pd.DataFrame(forecast_list)

# Encode for prediction
forecast_df['market_code'] = market_enc.transform(forecast_df['MARKET'])
forecast_df['product_code'] = product_enc.transform(forecast_df['PRODUCT_GROUP'])
forecast_df['season_code'] = season_enc.transform(forecast_df['SEASON'])
forecast_df['YEAR'] = 2026

X_forecast = forecast_df[feature_cols].copy()
forecast_df['PREDICTED_SALES'] = model.predict(X_forecast)

# Combine across markets
combined = forecast_df.groupby(['PRODUCT_GROUP', 'SEASON'], as_index=False)['PREDICTED_SALES'].sum()

# Pivot by season
season_order = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3, 'Festival': 4}
combined['season_sort'] = combined['SEASON'].map(season_order)
combined = combined.sort_values(['PRODUCT_GROUP', 'season_sort']).drop('season_sort', axis=1)

combined_pivot = combined.pivot(index='PRODUCT_GROUP', columns='SEASON', values='PREDICTED_SALES').fillna(0)
combined_pivot['TOTAL_2026'] = combined_pivot.sum(axis=1)
combined_pivot = combined_pivot[[c for c in ['Winter', 'Spring', 'Summer', 'Fall', 'Festival'] if c in combined_pivot.columns] + ['TOTAL_2026']]
combined_pivot = combined_pivot.sort_values('TOTAL_2026', ascending=False)

forecast_path = OUTPUT_DIR / 'task2_2026_forecast_retrained.csv'
combined_pivot.to_csv(forecast_path)
print(f"  [OK] 2026 forecast: {forecast_path}")

print(f"\n  Top 10 products by 2026 forecast:")
print(combined_pivot.head(10))

print("\n" + "=" * 80)
print("[COMPLETE] TASK 2 RETRAIN/VALIDATION COMPLETE")
print("=" * 80)
