"""
TASK 2: Sales Forecasting - Market × Product × Season Granularity
Answers: "What sales value will a product achieve in a specific market/season?"

Example: Ice Creams in UAE during Summer 2026 → $4.2M

Aggregation: GEOGRAPHICAL_CLUSTER × PRODUCT_GROUP × SEASON × YEAR
Features: Market size, product share, seasonal lags, promo intensity
Output: Market-Product-Season predictions for 2026
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

print("="*80)
print("TASK 2: SALES FORECASTING - MARKET × PRODUCT × SEASON")
print("="*80)

DATA_DIR = Path("data/csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "task2_market_seasonal_model.pkl"
METADATA_PATH = MODEL_DIR / "task2_market_seasonal_metadata.pkl"
SEASONALITY_PATH = MODEL_DIR / "market_product_seasonality.csv"

# Season definitions (business-oriented)
SEASON_MAP = {
    1: 'Winter', 2: 'Winter',           # Jan-Feb: Low activity, post-holiday
    3: 'Spring', 4: 'Spring',           # Mar-Apr: Ramadan period (varies), moderate demand
    5: 'Summer', 6: 'Summer', 7: 'Summer', 8: 'Summer',  # May-Aug: Peak for Ice Cream, high tourism
    9: 'Fall', 10: 'Fall',              # Sep-Oct: Back-to-school, moderate
    11: 'Festival', 12: 'Festival'      # Nov-Dec: Year-end holidays, high demand
}

# ============================================================================
# STEP 1: Load Contract_Claim_Dump with Geographic Data
# ============================================================================
print("\n[STEP 1] Loading Contract_Claim_Dump files with geographic data...")

claim_dump_files = sorted(list(DATA_DIR.glob("Contract_Claim_Dump*.csv")))
print(f"Found {len(claim_dump_files)} Contract_Claim_Dump files")

all_data = []

for csv_file in claim_dump_files:
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        
        # Check required columns
        required_cols = ['CUSTOMER_NAME', 'PRODUCT_GROUP', 'AMOUNT', 'CLAIM_DATE', 'MONTH', 'STATUS']
        if not all(col in df.columns for col in required_cols):
            print(f"  [SKIP] {csv_file.name}: Missing required columns")
            continue
        
        # Only use Approved claims
        df = df[df['STATUS'] == 'Approved'].copy()
        if len(df) == 0:
            print(f"  [SKIP] {csv_file.name}: No approved claims")
            continue
        
        # Create clean dataset
        df_clean = pd.DataFrame()
        df_clean['CUSTOMER'] = df['CUSTOMER_NAME'].astype(str).str.strip()
        df_clean['PRODUCT_GROUP'] = df['PRODUCT_GROUP'].astype(str).str.strip()
        df_clean['SALES'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
        df_clean['DATE'] = df['CLAIM_DATE']
        
        # Extract geographic market with better parsing
        market = 'UAE'
        if 'GEOGRAPHICAL_CLUSTER' in df.columns:
            geo = df['GEOGRAPHICAL_CLUSTER'].astype(str).str.strip()
            market = geo
        elif 'LOCATION_NAME' in df.columns:
            market = df['LOCATION_NAME'].astype(str).str.strip()
        elif 'LOCATION' in df.columns:
            market = df['LOCATION'].astype(str).str.strip()
        
        df_clean['MARKET'] = market
        
        # Clean market names - standardize
        df_clean['MARKET'] = df_clean['MARKET'].replace({
            'nan': 'UAE', 'NaN': 'UAE', '': 'UAE',
            'United Arab Emirates': 'UAE',
            'U.A.E': 'UAE',
            'Dubai': 'Dubai_UAE',
            'Abu Dhabi': 'AbuDhabi_UAE',
            'Sharjah': 'Sharjah_UAE'
        })
        
        # Remove rows with missing critical values
        df_clean = df_clean.dropna(subset=['SALES', 'CUSTOMER', 'PRODUCT_GROUP', 'MARKET'])
        df_clean = df_clean[df_clean['SALES'] > 0]  # Only positive sales
        
        all_data.append(df_clean)
        print(f"  [OK] {csv_file.name}: {len(df_clean):,} approved records")
        
    except Exception as e:
        print(f"  [ERR] {csv_file.name}: {e}")

if not all_data:
    raise ValueError("No valid data loaded from Contract_Claim_Dump files")

# Combine all data
df = pd.concat(all_data, ignore_index=True)
print(f"\nTotal records loaded: {len(df):,}")
print(f"Unique markets: {df['MARKET'].nunique()}")
print(f"Markets: {sorted(df['MARKET'].unique())}")
print(f"Unique products: {df['PRODUCT_GROUP'].nunique()}")

# ============================================================================
# STEP 2: Parse dates and create monthly aggregation
# ============================================================================
print("\n[STEP 2] Date processing and monthly aggregation...")

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

df['DATE'] = df['DATE'].apply(parse_date)
df = df.dropna(subset=['DATE'])

print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")

# Create month key for aggregation
df['YEAR'] = df['DATE'].dt.year
df['MONTH'] = df['DATE'].dt.month
df['YEAR_MONTH'] = df['DATE'].dt.to_period('M')

# Aggregate to Market × Product × Month
df_monthly = df.groupby(['MARKET', 'PRODUCT_GROUP', 'YEAR', 'MONTH', 'YEAR_MONTH']).agg({
    'SALES': 'sum',
    'CUSTOMER': 'nunique'  # Number of unique customers
}).reset_index()

df_monthly.rename(columns={'CUSTOMER': 'CUSTOMER_COUNT'}, inplace=True)
df_monthly['MONTH_START'] = df_monthly['YEAR_MONTH'].dt.to_timestamp()

print(f"Monthly aggregated data: {len(df_monthly):,} rows")
print(f"Time periods: {df_monthly['YEAR_MONTH'].nunique()} months")
print(f"Market-Product combinations: {df_monthly.groupby(['MARKET', 'PRODUCT_GROUP']).ngroups:,}")

# ============================================================================
# STEP 3: Map to Seasons and Aggregate
# ============================================================================
print("\n[STEP 3] Mapping to seasons and aggregating...")

df_monthly['SEASON'] = df_monthly['MONTH'].map(SEASON_MAP)

# Aggregate to Market × Product × Season × Year
df_seasonal = df_monthly.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR']).agg({
    'SALES': 'sum',
    'CUSTOMER_COUNT': 'mean'
}).reset_index()

print(f"Seasonal aggregated data: {len(df_seasonal):,} rows")
print(f"Seasons: {sorted(df_seasonal['SEASON'].unique())}")
print(f"Years: {sorted(df_seasonal['YEAR'].unique())}")

# ============================================================================
# STEP 3.5: Merge Promotional Data
# ============================================================================
print("\n[STEP 3.5] Merging promotional data...")

PROMO_PATH = Path("results/promo_market_seasonal.csv")
if PROMO_PATH.exists():
    promo_df = pd.read_csv(PROMO_PATH)
    print(f"  Loaded {len(promo_df):,} promo records")
    
    # Merge promo data
    df_seasonal = df_seasonal.merge(
        promo_df[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'PROMO_COUNT', 'PROMO_LOG']],
        on=['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR'],
        how='left'
    )
    
    df_seasonal['PROMO_COUNT'] = df_seasonal['PROMO_COUNT'].fillna(0)
    df_seasonal['PROMO_LOG'] = df_seasonal['PROMO_LOG'].fillna(0)
    print(f"  ✓ Merged promo features (PROMO_COUNT, PROMO_LOG)")
else:
    print("  ! No promo data found, setting promo features to 0")
    df_seasonal['PROMO_COUNT'] = 0
    df_seasonal['PROMO_LOG'] = 0

# ============================================================================
# STEP 4: Feature Engineering - Market-Season Level
# ============================================================================
print("\n[STEP 4] Creating market-season features...")

df = df_seasonal.copy()

# Sort for lag calculations
df = df.sort_values(['MARKET', 'PRODUCT_GROUP', 'YEAR', 'SEASON'])

# Seasonal lags (previous season, same season last year)
for lag in [1, 4]:  # 1 season ago, 4 seasons ago (same season last year)
    df[f'LAG_{lag}_SEASON'] = df.groupby(['MARKET', 'PRODUCT_GROUP'])['SALES'].shift(lag)

# Rolling averages by season
df['LAG_2_SEASON_AVG'] = df.groupby(['MARKET', 'PRODUCT_GROUP'])['SALES'].transform(
    lambda x: x.shift(1).rolling(2, min_periods=1).mean()
)

# Market-level features
df['MARKET_TOTAL'] = df.groupby(['MARKET', 'YEAR', 'SEASON'])['SALES'].transform('sum')
df['MARKET_PRODUCT_SHARE'] = (df['SALES'] / (df['MARKET_TOTAL'] + 1)) * 100

# Product-level features across all markets
df['PRODUCT_TOTAL'] = df.groupby(['PRODUCT_GROUP', 'YEAR', 'SEASON'])['SALES'].transform('sum')

# Historical averages
df['MARKET_PRODUCT_AVG'] = df.groupby(['MARKET', 'PRODUCT_GROUP'])['SALES'].transform('mean')
df['MARKET_AVG'] = df.groupby('MARKET')['SALES'].transform('mean')
df['PRODUCT_AVG'] = df.groupby('PRODUCT_GROUP')['SALES'].transform('mean')

# Year-over-year growth
df['YOY_GROWTH'] = df.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON'])['SALES'].pct_change(periods=1).fillna(0) * 100

# Season encoding
df['SEASON_ENCODED'] = df['SEASON'].astype('category').cat.codes

# Filter rows with sufficient history
df = df[df['LAG_1_SEASON'].notna()].copy()

print(f"Data after feature engineering: {len(df):,} rows")

# ============================================================================
# STEP 5: Extract Seasonality Patterns
# ============================================================================
print("\n[STEP 5] Extracting market-product-season patterns...")

seasonality_data = []

for (market, product) in df.groupby(['MARKET', 'PRODUCT_GROUP']).groups.keys():
    subset = df[(df['MARKET'] == market) & (df['PRODUCT_GROUP'] == product)]
    
    for season in subset['SEASON'].unique():
        season_data = subset[subset['SEASON'] == season]
        avg_sales = season_data['SALES'].mean()
        overall_avg = subset['SALES'].mean()
        
        index = avg_sales / overall_avg if overall_avg > 0 else 1.0
        
        seasonality_data.append({
            'MARKET': market,
            'PRODUCT_GROUP': product,
            'SEASON': season,
            'SEASONALITY_INDEX': round(index, 4),
            'RECORDS': len(season_data)
        })

df_seasonality = pd.DataFrame(seasonality_data)
df_seasonality.to_csv(SEASONALITY_PATH, index=False)
print(f"  Saved seasonality to: {SEASONALITY_PATH}")
print(f"  Market-Product-Season combinations: {len(df_seasonality):,}")

# Merge seasonality back
df = df.merge(
    df_seasonality[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'SEASONALITY_INDEX']],
    on=['MARKET', 'PRODUCT_GROUP', 'SEASON'],
    how='left'
)
df['SEASONALITY_INDEX'] = df['SEASONALITY_INDEX'].fillna(1.0)

# ============================================================================
# STEP 6: Prepare training data
# ============================================================================
print("\n[STEP 6] Preparing training data...")

feature_cols = [
    'LAG_1_SEASON', 'LAG_4_SEASON', 'LAG_2_SEASON_AVG',
    'MARKET_PRODUCT_SHARE', 'CUSTOMER_COUNT',
    'MARKET_PRODUCT_AVG', 'MARKET_AVG', 'PRODUCT_AVG',
    'YOY_GROWTH', 'SEASON_ENCODED',
    'SEASONALITY_INDEX',
    'PROMO_COUNT', 'PROMO_LOG'  # Added promo features
]

# Log transform target
df['SALES_LOG'] = np.log1p(df['SALES'])

# Drop rows with NaN in features
df_train = df.dropna(subset=feature_cols + ['SALES_LOG']).copy()

print(f"Training data: {len(df_train):,} rows")

# Split: 2022-2024 training, 2025 testing
train_data = df_train[df_train['YEAR'] < 2025]
test_data = df_train[df_train['YEAR'] == 2025]

print(f"Training set: {len(train_data):,} rows ({train_data['YEAR'].min()}-{train_data['YEAR'].max()})")
print(f"Test set: {len(test_data):,} rows ({test_data['YEAR'].min()}-{test_data['YEAR'].max()})")

# ============================================================================
# STEP 7: Train model
# ============================================================================
print("\n[STEP 7] Training HistGradientBoosting model...")

X_train = train_data[feature_cols]
y_train = train_data['SALES_LOG']
X_test = test_data[feature_cols]
y_test = test_data['SALES_LOG']

model = HistGradientBoostingRegressor(
    max_depth=6,
    learning_rate=0.03,
    l2_regularization=0.1,
    min_samples_leaf=20,
    max_iter=500,
    verbose=0,
    random_state=42
)

model.fit(X_train, y_train)
print("  Model trained")

# ============================================================================
# STEP 8: Evaluate
# ============================================================================
print("\n[STEP 8] Model evaluation...")

y_pred_train_log = model.predict(X_train)
y_pred_test_log = model.predict(X_test)

y_train_actual = np.expm1(y_train)
y_pred_train_actual = np.expm1(y_pred_train_log)
y_test_actual = np.expm1(y_test)
y_pred_test_actual = np.expm1(y_pred_test_log)

train_r2 = r2_score(y_train_actual, y_pred_train_actual)
train_mae = mean_absolute_error(y_train_actual, y_pred_train_actual)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual))

test_r2 = r2_score(y_test_actual, y_pred_test_actual)
test_mae = mean_absolute_error(y_test_actual, y_pred_test_actual)
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual))

def ape(y_true, y_pred):
    return np.abs((y_true - y_pred) / (y_true + 1)) * 100

train_ape = ape(y_train_actual, y_pred_train_actual)
test_ape = ape(y_test_actual, y_pred_test_actual)

print(f"\n[TRAIN]")
print(f"  R-squared: {train_r2:.4f}")
print(f"  MAE: ${train_mae:,.0f}")
print(f"  RMSE: ${train_rmse:,.0f}")
print(f"  Median APE: {np.median(train_ape):.1f}%")

print(f"\n[TEST]")
print(f"  R-squared: {test_r2:.4f}")
print(f"  MAE: ${test_mae:,.0f}")
print(f"  RMSE: ${test_rmse:,.0f}")
print(f"  Median APE: {np.median(test_ape):.1f}%")

# ============================================================================
# STEP 9: Save artifacts
# ============================================================================
print("\n[STEP 9] Saving model artifacts...")

joblib.dump(model, MODEL_PATH)
print(f"  Model saved to: {MODEL_PATH}")

metadata = {
    'feature_cols': feature_cols,
    'markets': sorted(df['MARKET'].unique().tolist()),
    'products': sorted(df['PRODUCT_GROUP'].unique().tolist()),
    'seasons': sorted(df['SEASON'].unique().tolist()),
    'training_date': datetime.now().isoformat(),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'test_median_ape': float(np.median(test_ape)),
    'granularity': 'Market × Product_Group × Season',
    'data_source': 'Contract_Claim_Dump 2022-2025'
}

joblib.dump(metadata, METADATA_PATH)
print(f"  Metadata saved to: {METADATA_PATH}")

# Save historical seasonal data for inference
history_path = MODEL_DIR / "history_market_seasonal.pkl"
history_data = df_train[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'SALES', 'CUSTOMER_COUNT']].copy()
joblib.dump(history_data, history_path)
print(f"  Historical data saved to: {history_path}")

# Show 2025 predictions vs actuals by market-product
print("\n[STEP 10] Sample 2025 predictions vs actuals...")
test_results = test_data[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR']].copy()
test_results['ACTUAL'] = y_test_actual.values
test_results['PREDICTED'] = y_pred_test_actual
test_results['APE'] = ape(y_test_actual, y_pred_test_actual).values

print(test_results.nlargest(10, 'ACTUAL')[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'ACTUAL', 'PREDICTED', 'APE']])

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"\nModel ready for market-season forecasting!")
print(f"Markets: {len(metadata['markets'])}")
print(f"Products: {len(metadata['products'])}")
print(f"Seasons: {len(metadata['seasons'])}")
print(f"Test Median APE: {metadata['test_median_ape']:.1f}%")
print(f"\nBusiness Question Answered: 'What sales will [Product] achieve in [Market] during [Season]?'")
