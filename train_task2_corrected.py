"""
TASK 2: Sales Forecasting Model - CORRECTED VERSION
Uses Contract_Claim_Dump files (2022-2025) with proper:
- Customer: CUSTOMER_NAME
- Product: PRODUCT_GROUP  (actual product category, not SKU)
- Amount: AMOUNT
- Date: CLAIM_DATE + MONTH
- Monthly aggregation by Customer + PRODUCT_GROUP
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from datetime import datetime
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings('ignore')

print("="*80)
print("TASK 2: SALES FORECASTING - CORRECTED PIPELINE")
print("="*80)

DATA_DIR = Path("data/csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "task2_reboot_model.pkl"
STATES_PATH = MODEL_DIR / "task2_reboot_states.pkl"
METADATA_PATH = MODEL_DIR / "task2_reboot_metadata.pkl"

# ============================================================================
# STEP 1: Load Contract_Claim_Dump files (2022-2025)
# ============================================================================
print("\n[STEP 1] Loading Contract_Claim_Dump files...")

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
        df_clean['MONTH_NAME'] = df['MONTH'].astype(str).str.strip()
        
        # Remove rows with missing critical values
        df_clean = df_clean.dropna(subset=['SALES', 'CUSTOMER', 'PRODUCT_GROUP'])
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
print(f"Unique customers: {df['CUSTOMER'].nunique():,}")
print(f"Unique products: {df['PRODUCT_GROUP'].nunique()}")
print(f"Products: {df['PRODUCT_GROUP'].unique()}")

# ============================================================================
# STEP 2: Parse dates and create monthly aggregation
# ============================================================================
print("\n[STEP 2] Date processing and aggregation...")

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

df['DATE'] = df['DATE'].apply(parse_date)
df = df.dropna(subset=['DATE'])

print(f"Date range: {df['DATE'].min()} to {df['DATE'].max()}")

# Create month key for aggregation
df['YEAR_MONTH'] = df['DATE'].dt.to_period('M')

# Aggregate to monthly level (Customer + Product_Group + Month)
df_monthly = df.groupby(['CUSTOMER', 'PRODUCT_GROUP', 'YEAR_MONTH']).agg({
    'SALES': 'sum',
    'MONTH_NAME': 'first'
}).reset_index()

df_monthly['MONTH_START'] = df_monthly['YEAR_MONTH'].dt.to_timestamp()
df_monthly = df_monthly.sort_values(['CUSTOMER', 'PRODUCT_GROUP', 'MONTH_START'])

print(f"Monthly aggregated data: {len(df_monthly):,} rows")
print(f"Time periods: {df_monthly['YEAR_MONTH'].nunique()} months")
print(f"Customer-Product combinations: {df_monthly.groupby(['CUSTOMER', 'PRODUCT_GROUP']).ngroups:,}")

# ============================================================================
# STEP 3: Feature engineering
# ============================================================================
print("\n[STEP 3] Creating features...")

df = df_monthly.copy()

# Create lag features by Customer-Product group
for lag in [1, 3, 6, 12]:
    df[f'LAG_{lag}'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].shift(lag)

# Calculate averages
df['LAG_3_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform(
    lambda x: x.shift(1).rolling(3, min_periods=1).mean()
)
df['LAG_6_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform(
    lambda x: x.shift(1).rolling(6, min_periods=1).mean()
)

# Time features
df['MONTH'] = df['MONTH_START'].dt.month
df['QUARTER'] = df['MONTH_START'].dt.quarter
df['YEAR'] = df['MONTH_START'].dt.year
df['IS_SUMMER'] = df['MONTH'].isin([6, 7, 8]).astype(int)

# Baselines
df['CUSTOMER_AVG'] = df.groupby('CUSTOMER')['SALES'].transform('mean')
df['PRODUCT_AVG'] = df.groupby('PRODUCT_GROUP')['SALES'].transform('mean')
df['CUSTOMER_PRODUCT_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform('mean')

# Trend
df['TREND_3M'] = df.groupby(['CUSTOMER', 'PRODUCT_GROUP'])['SALES'].transform(
    lambda x: (x.shift(1) - x.shift(4)) / 4
).fillna(0)

# Filter rows with sufficient history
df = df[df['LAG_1'].notna()].copy()

print(f"Data after feature engineering: {len(df):,} rows")

# ============================================================================
# STEP 4: Extract seasonality patterns
# ============================================================================
print("\n[STEP 4] Extracting seasonality patterns...")

seasonality_data = []

for product in df['PRODUCT_GROUP'].unique():
    prod_data = df[df['PRODUCT_GROUP'] == product]
    
    # Get monthly averages across all customers
    monthly_avg = prod_data.groupby('MONTH')['SALES'].mean()
    
    if len(monthly_avg) > 0:
        overall_avg = monthly_avg.mean()
        
        for month, value in monthly_avg.items():
            index = value / overall_avg if overall_avg > 0 else 1.0
            
            # Calculate confidence based on data availability
            month_records = len(prod_data[prod_data['MONTH'] == month])
            confidence = min(1.0, month_records / 50)  # Normalize to 50 records
            
            seasonality_data.append({
                'PRODUCT_GROUP': product,
                'MONTH': month,
                'SEASONALITY_INDEX': round(index, 4),
                'CONFIDENCE': round(confidence, 3),
                'RECORDS': month_records
            })

df_seasonality = pd.DataFrame(seasonality_data)

# Fill missing months with 1.0
for product in df['PRODUCT_GROUP'].unique():
    existing_months = df_seasonality[df_seasonality['PRODUCT_GROUP'] == product]['MONTH'].unique()
    for month in range(1, 13):
        if month not in existing_months:
            df_seasonality = pd.concat([df_seasonality, pd.DataFrame([{
                'PRODUCT_GROUP': product,
                'MONTH': month,
                'SEASONALITY_INDEX': 1.0,
                'CONFIDENCE': 0.5,
                'RECORDS': 0
            }])], ignore_index=True)

df_seasonality = df_seasonality.sort_values(['PRODUCT_GROUP', 'MONTH'])

print(f"Extracted seasonality for {df_seasonality['PRODUCT_GROUP'].nunique()} products")
print(f"Sample seasonality patterns:")
for product in df_seasonality['PRODUCT_GROUP'].unique()[:3]:
    indices = df_seasonality[df_seasonality['PRODUCT_GROUP'] == product]['SEASONALITY_INDEX'].values
    print(f"  {product}: min={indices.min():.2f}x, max={indices.max():.2f}x, avg={indices.mean():.2f}x")

# Save seasonality
seasonality_path = MODEL_DIR / "product_seasonality_correct.csv"
df_seasonality.to_csv(seasonality_path, index=False)
print(f"  Saved to: {seasonality_path}")

# ============================================================================
# STEP 5: Merge seasonality into training data
# ============================================================================
print("\n[STEP 5] Merging seasonality...")

df = df.merge(
    df_seasonality[['PRODUCT_GROUP', 'MONTH', 'SEASONALITY_INDEX', 'CONFIDENCE']],
    left_on=['PRODUCT_GROUP', 'MONTH'],
    right_on=['PRODUCT_GROUP', 'MONTH'],
    how='left'
)
df['SEASONALITY_INDEX'] = df['SEASONALITY_INDEX'].fillna(1.0)
df['CONFIDENCE'] = df['CONFIDENCE'].fillna(0.5)

# ============================================================================
# STEP 6: Prepare training data
# ============================================================================
print("\n[STEP 6] Preparing training data...")

# Select features
feature_cols = [
    'LAG_1', 'LAG_3', 'LAG_6', 'LAG_12',
    'LAG_3_AVG', 'LAG_6_AVG',
    'CUSTOMER_AVG', 'PRODUCT_AVG', 'CUSTOMER_PRODUCT_AVG',
    'TREND_3M',
    'MONTH', 'QUARTER', 'YEAR', 'IS_SUMMER',
    'SEASONALITY_INDEX'
]

# Log transform target
df['SALES_LOG'] = np.log1p(df['SALES'])

# Drop rows with NaN in features
df_train = df.dropna(subset=feature_cols + ['SALES_LOG']).copy()

print(f"Training data: {len(df_train):,} rows")

# Split data: 2022-2024 for training, 2025 for testing
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
    max_depth=7,
    learning_rate=0.02,
    l2_regularization=0.1,
    min_samples_leaf=30,
    max_iter=400,
    verbose=0
)

model.fit(X_train, y_train)
print("  Model trained")

# ============================================================================
# STEP 8: Evaluate
# ============================================================================
print("\n[STEP 8] Model evaluation...")

# Predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# Convert back from log
y_train_actual = np.expm1(y_train)
y_test_actual = np.expm1(y_test)
y_pred_train_actual = np.expm1(y_pred_train)
y_pred_test_actual = np.expm1(y_pred_test)

# Metrics
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
train_mae = mean_absolute_error(y_train_actual, y_pred_train_actual)
test_mae = mean_absolute_error(y_test_actual, y_pred_test_actual)
train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual))

# APE
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

# Save model
joblib.dump(model, MODEL_PATH)
print(f"  Model saved to: {MODEL_PATH}")

# Save metadata
metadata = {
    'feature_cols': feature_cols,
    'products': sorted(df['PRODUCT_GROUP'].unique().tolist()),
    'customers': sorted(df['CUSTOMER'].unique().tolist()),
    'training_date': datetime.now().isoformat(),
    'train_r2': float(train_r2),
    'test_r2': float(test_r2),
    'test_median_ape': float(np.median(test_ape)),
    'data_source': 'Contract_Claim_Dump files 2022-2025'
}

joblib.dump(metadata, METADATA_PATH)
print(f"  Metadata saved to: {METADATA_PATH}")

# Save states (for inference) - vectorized version
print("  Building customer-product states...")

# Create states more efficiently
def build_states_row(group):
    customer = group['CUSTOMER'].iloc[0]
    product = group['PRODUCT_GROUP'].iloc[0]
    recent = group.nlargest(12, 'MONTH_START')['SALES'].values[::-1]  # Last 12, reversed
    
    return {
        'customer': customer,
        'product': product,
        'recent_history': recent.tolist(),
        'customer_avg': float(group['SALES'].mean()),
        'last_date': str(group['MONTH_START'].max()),
    }

states = {}
for key, group in df.groupby(['CUSTOMER', 'PRODUCT_GROUP']):
    if len(group) >= 12:
        states[key] = build_states_row(group)

print(f"  Built states for {len(states):,} customer-product combinations")

joblib.dump(states, STATES_PATH)
print(f"  States saved to: {STATES_PATH}")

# Save historical monthly data for inference
print("\n  Building historical monthly data...")
history_monthly_path = MODEL_DIR / "history_monthly.pkl"

# Prepare clean historical data
history_data = df_monthly[['CUSTOMER', 'PRODUCT_GROUP', 'YEAR_MONTH', 'MONTH_START', 'SALES', 'MONTH_NAME']].copy()
history_data['MONTH'] = history_data['MONTH_START'].dt.month  # Add numeric month
history_data['YEAR'] = history_data['MONTH_START'].dt.year    # Add numeric year
history_data['Plot_Date'] = history_data['MONTH_START']       # For plotting compatibility
history_data = history_data.sort_values(['CUSTOMER', 'PRODUCT_GROUP', 'MONTH_START'])

# Save as pickle for fast loading
joblib.dump(history_data, history_monthly_path)
print(f"  Historical data saved to: {history_monthly_path}")
print(f"  Total historical records: {len(history_data):,}")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print(f"\nModel ready for inference!")
print(f"Products: {len(metadata['products'])}")
print(f"Customers: {len(metadata['customers'])}")
print(f"Test Median APE: {metadata['test_median_ape']:.1f}%")
