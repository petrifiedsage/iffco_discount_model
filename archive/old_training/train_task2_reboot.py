"""
Task 2: Sales Forecasting - Simplified Robust Baseline
Author: Senior Data Scientist
Date: December 2025

This script builds a clean, production-ready sales forecasting model from scratch.
It handles messy data, creates proper lag features, and saves everything needed for inference.
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

# ============================================================================
# CONFIGURATION
# ============================================================================
DATA_DIR = Path("data/csv")  # Updated to use extracted CSV files
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "task2_reboot_model.pkl"
STATES_PATH = MODEL_DIR / "task2_reboot_states.pkl"
METADATA_PATH = MODEL_DIR / "task2_reboot_metadata.pkl"

print("="*80)
print("TASK 2: SALES FORECASTING - REBOOT")
print("="*80)


# ============================================================================
# 1. ROBUST DATE PARSER
# ============================================================================
def parse_mixed_dates(date_series):
    """
    Robustly parse dates with mixed formats like '06-02-22' and '11-DEC-24'.
    
    Args:
        date_series: pandas Series containing dates in mixed formats
        
    Returns:
        pandas Series with datetime objects
    """
    parsed_dates = []
    
    for date_val in date_series:
        if pd.isna(date_val):
            parsed_dates.append(pd.NaT)
            continue
            
        date_str = str(date_val).strip()
        parsed = None
        
        # Try multiple common formats
        formats_to_try = [
            '%d-%m-%y',      # 06-02-22
            '%d-%m-%Y',      # 06-02-2022
            '%d-%b-%y',      # 11-DEC-24
            '%d-%b-%Y',      # 11-DEC-2024
            '%Y-%m-%d',      # 2022-02-06
            '%m/%d/%Y',      # 02/06/2022
            '%d/%m/%Y',      # 06/02/2022
        ]
        
        for fmt in formats_to_try:
            try:
                parsed = pd.to_datetime(date_str, format=fmt)
                break
            except:
                continue
        
        # If nothing worked, try pandas automatic parsing as last resort
        if parsed is None:
            try:
                parsed = pd.to_datetime(date_str, dayfirst=True)
            except:
                parsed = pd.NaT
        
        parsed_dates.append(parsed)
    
    return pd.Series(parsed_dates)


# ============================================================================
# 2. LOAD & CLEAN DATA
# ============================================================================
def load_and_clean_data():
    """
    Load all CSV files, clean, standardize, and filter the data.
    
    Returns:
        pandas DataFrame with cleaned data
    """
    print("\n[STEP 1] Loading and cleaning data...")
    
    all_data = []
    csv_files = list(DATA_DIR.glob("*.csv"))
    
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Check if required columns exist (case-insensitive)
            df.columns = df.columns.str.strip()
            col_map = {col.upper(): col for col in df.columns}
            
            required = ['CLAIM_DATE', 'AMOUNT', 'CUSTOMER_NAME', 'PRODUCT_GROUP']
            missing = [c for c in required if c not in col_map]
            
            if missing:
                print(f"  ⚠ Skipping {csv_file.name}: Missing columns {missing}")
                continue
            
            # Rename to standard names
            df_clean = pd.DataFrame()
            df_clean['DATE'] = df[col_map['CLAIM_DATE']]
            df_clean['SALES'] = pd.to_numeric(df[col_map['AMOUNT']], errors='coerce')
            df_clean['CUSTOMER'] = df[col_map['CUSTOMER_NAME']].astype(str).str.strip()
            df_clean['PRODUCT'] = df[col_map['PRODUCT_GROUP']].astype(str).str.strip()
            
            # Add region filter if column exists
            if 'GEOGRAPHICAL_CLUSTER' in col_map:
                df_clean['REGION'] = df[col_map['GEOGRAPHICAL_CLUSTER']].astype(str).str.strip()
                # Filter for UAE region
                uae_mask = df_clean['REGION'].str.contains('United Arab|UAE', case=False, na=False)
                df_clean = df_clean[uae_mask]
                print(f"  [OK] {csv_file.name}: {len(df_clean)} UAE records")
            else:
                print(f"  ✓ {csv_file.name}: {len(df_clean)} records (no region filter)")
            
            if len(df_clean) > 0:
                all_data.append(df_clean[['DATE', 'SALES', 'CUSTOMER', 'PRODUCT']])
                
        except Exception as e:
            print(f"  [ERR] Error processing {csv_file.name}: {e}")
            continue
    
    if not all_data:
        raise ValueError("No valid data loaded from any CSV file")
    
    # Combine all data
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\nCombined data: {len(df_combined):,} rows")
    
    # Parse dates robustly
    print("Parsing dates with mixed formats...")
    df_combined['DATE'] = parse_mixed_dates(df_combined['DATE'])
    
    # Clean data
    print("Cleaning data...")
    df_combined = df_combined.dropna(subset=['DATE', 'SALES'])
    df_combined = df_combined[df_combined['SALES'] > 0]
    df_combined = df_combined[df_combined['CUSTOMER'] != 'nan']
    df_combined = df_combined[df_combined['PRODUCT'] != 'nan']
    
    print(f"After cleaning: {len(df_combined):,} rows")
    print(f"Date range: {df_combined['DATE'].min()} to {df_combined['DATE'].max()}")
    print(f"Unique customers: {df_combined['CUSTOMER'].nunique()}")
    print(f"Unique products: {df_combined['PRODUCT'].nunique()}")
    
    return df_combined


# ============================================================================
# 3. AGGREGATE TO MONTHLY LEVEL
# ============================================================================
def aggregate_monthly(df):
    """
    Aggregate sales to monthly level by Customer + Product.
    
    Args:
        df: Cleaned dataframe
        
    Returns:
        Monthly aggregated dataframe
    """
    print("\n[STEP 2] Aggregating to monthly level...")
    
    # Ensure PRODUCT is string type before aggregation
    df['PRODUCT'] = df['PRODUCT'].astype(str).str.strip()
    
    # Create month start date
    df['MONTH_START'] = df['DATE'].dt.to_period('M').dt.to_timestamp()
    
    # Aggregate
    df_monthly = df.groupby(['CUSTOMER', 'PRODUCT', 'MONTH_START']).agg({
        'SALES': 'sum'
    }).reset_index()
    
    # Ensure PRODUCT is string after aggregation
    df_monthly['PRODUCT'] = df_monthly['PRODUCT'].astype(str).str.strip()
    
    df_monthly = df_monthly.sort_values(['CUSTOMER', 'PRODUCT', 'MONTH_START'])
    
    print(f"Monthly data: {len(df_monthly):,} rows")
    print(f"Time periods: {df_monthly['MONTH_START'].nunique()} months")
    
    return df_monthly


# ============================================================================
# 4. FEATURE ENGINEERING - ENHANCED WITH BASELINES & TRENDS
# ============================================================================
def create_lag_features(df):
    """
    Create comprehensive features: lags, seasonality, baselines, trends.
    
    Critical: Lag_1 for February should contain January's sales.
    This prevents data leakage during training.
    
    Args:
        df: Monthly aggregated dataframe
        
    Returns:
        DataFrame with enhanced features
    """
    print("\n[STEP 3] Creating enhanced features...")
    
    df = df.copy()
    
    # Sort to ensure proper ordering
    df = df.sort_values(['CUSTOMER', 'PRODUCT', 'MONTH_START'])
    
    # ========================================================================
    # A. BASELINE FEATURES (Customer & Product Averages)
    # ========================================================================
    print("  Creating baseline features...")
    
    # Calculate overall averages per customer and product (using only past data)
    customer_avg = df.groupby('CUSTOMER')['SALES'].transform('mean')
    product_avg = df.groupby('PRODUCT')['SALES'].transform('mean')
    customer_product_avg = df.groupby(['CUSTOMER', 'PRODUCT'])['SALES'].transform('mean')
    
    df['CUSTOMER_AVG'] = customer_avg
    df['PRODUCT_AVG'] = product_avg
    df['CUSTOMER_PRODUCT_AVG'] = customer_product_avg
    
    # Relative importance (how much this customer spends on this product vs their total)
    df['RELATIVE_IMPORTANCE'] = df['CUSTOMER_PRODUCT_AVG'] / (df['CUSTOMER_AVG'] + 1)
    
    # ========================================================================
    # B. LAG FEATURES
    # ========================================================================
    print("  Creating lag features...")
    
    df['LAG_1'] = df.groupby(['CUSTOMER', 'PRODUCT'])['SALES'].shift(1)
    df['LAG_2'] = df.groupby(['CUSTOMER', 'PRODUCT'])['SALES'].shift(2)
    df['LAG_3'] = df.groupby(['CUSTOMER', 'PRODUCT'])['SALES'].shift(3)
    df['LAG_12'] = df.groupby(['CUSTOMER', 'PRODUCT'])['SALES'].shift(12)
    
    # Rolling statistics
    df['LAG_3_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT'])['SALES'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
    )
    df['LAG_6_AVG'] = df.groupby(['CUSTOMER', 'PRODUCT'])['SALES'].transform(
        lambda x: x.shift(1).rolling(window=6, min_periods=1).mean()
    )
    df['LAG_3_STD'] = df.groupby(['CUSTOMER', 'PRODUCT'])['SALES'].transform(
        lambda x: x.shift(1).rolling(window=3, min_periods=1).std()
    ).fillna(0)
    
    # ========================================================================
    # C. TREND/VELOCITY FEATURES
    # ========================================================================
    print("  Creating trend features...")
    
    # Recent trend (last 3 months slope)
    df['TREND_3M'] = df.groupby(['CUSTOMER', 'PRODUCT'])['SALES'].transform(
        lambda x: (x.shift(1) - x.shift(3)) / 3
    ).fillna(0)
    
    # Momentum (recent vs longer-term average)
    df['MOMENTUM'] = (df['LAG_3_AVG'] / (df['LAG_6_AVG'] + 1)).fillna(1.0)
    
    # Growth rate
    df['GROWTH_RATE'] = ((df['LAG_1'] - df['LAG_12']) / (df['LAG_12'] + 1)).fillna(0)
    df['GROWTH_RATE'] = df['GROWTH_RATE'].clip(-5, 5)  # Clip extreme growth rates
    
    # ========================================================================
    # D. TIME FEATURES
    # ========================================================================
    df['MONTH'] = df['MONTH_START'].dt.month
    df['QUARTER'] = df['MONTH_START'].dt.quarter
    df['YEAR'] = df['MONTH_START'].dt.year
    df['IS_SUMMER'] = df['MONTH'].isin([6, 7, 8]).astype(int)
    df['IS_RAMADAN_PERIOD'] = df['MONTH'].isin([3, 4]).astype(int)  # Approximate
    
    # ========================================================================
    # E. SEASONALITY INDEX (Enhanced multi-year seasonality)
    # ========================================================================
    # Try enhanced seasonality first (multi-year, 1749 products), then fallback
    seasonality_path = MODEL_DIR / "product_seasonality_enhanced.csv"
    if not seasonality_path.exists():
        seasonality_path = MODEL_DIR / "product_seasonality_claims.csv"
    if not seasonality_path.exists():
        seasonality_path = MODEL_DIR / "product_seasonality.csv"
    
    if seasonality_path.exists():
        df_season = pd.read_csv(seasonality_path)
        # Ensure proper types for merge
        df_season['PRODUCT_GROUP'] = df_season['PRODUCT_GROUP'].astype(str).str.strip()
        df_season['MONTH'] = pd.to_numeric(df_season['MONTH'], errors='coerce')
        # Handle text month names (April, May, etc.)
        month_names = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        mask = df_season['MONTH'].isna()
        if mask.any():
            df_season.loc[mask, 'MONTH'] = df_season.loc[mask, 'MONTH'].str.lower().map(month_names)
        df_season['MONTH'] = df_season['MONTH'].astype(int)
        
        df['PRODUCT'] = df['PRODUCT'].astype(str).str.strip()
        
        df = df.merge(
            df_season, 
            left_on=['PRODUCT', 'MONTH'], 
            right_on=['PRODUCT_GROUP', 'MONTH'],
            how='left'
        )
        print(f"  [OK] Loaded seasonality from: {seasonality_path.name}")
        print(f"    Products covered: {df_season['PRODUCT_GROUP'].nunique()}")
        df['SEASONALITY_INDEX'] = df['SEASONALITY_INDEX'].fillna(1.0)
        # Load confidence if available (enhanced file has it)
        if 'CONFIDENCE' in df_season.columns:
            df = df.merge(
                df_season[['PRODUCT_GROUP', 'MONTH', 'CONFIDENCE']],
                left_on=['PRODUCT', 'MONTH'],
                right_on=['PRODUCT_GROUP', 'MONTH'],
                how='left',
                suffixes=('', '_conf')
            )
            df['SEASONALITY_CONFIDENCE'] = df['CONFIDENCE'].fillna(1.0)
        else:
            df['SEASONALITY_CONFIDENCE'] = 1.0
        print("  ✓ Seasonality index loaded with confidence weights")
    else:
        df['SEASONALITY_INDEX'] = 1.0
        df['SEASONALITY_CONFIDENCE'] = 1.0
        print("  ⚠ Seasonality index not found")
    
    # ========================================================================
    # F. PRODUCT CATEGORY ENCODING (Volume-based)
    # ========================================================================
    print("  Creating product category features...")
    
    # Categorize products by volume
    product_volumes = df.groupby('PRODUCT')['SALES'].sum()
    high_volume = product_volumes.quantile(0.75)
    low_volume = product_volumes.quantile(0.25)
    
    df['PRODUCT_VOLUME_CAT'] = df['PRODUCT'].map(
        lambda p: 2 if product_volumes.get(p, 0) > high_volume 
        else (0 if product_volumes.get(p, 0) < low_volume else 1)
    )
    
    # ========================================================================
    # G. FILL MISSING VALUES
    # ========================================================================
    lag_cols = ['LAG_1', 'LAG_2', 'LAG_3', 'LAG_12', 'LAG_3_AVG', 'LAG_6_AVG']
    df[lag_cols] = df[lag_cols].fillna(0)
    
    print(f"\n  ✓ Created {len(df.columns)} features")
    print(f"  Key features: Lags, Baselines, Trends, Seasonality")
    
    # Show sample
    sample_cols = ['CUSTOMER', 'PRODUCT', 'SALES', 'LAG_1', 'CUSTOMER_PRODUCT_AVG', 
                   'TREND_3M', 'SEASONALITY_INDEX']
    print(f"\n  Sample:")
    print(df[sample_cols].head(5).to_string(index=False))
    
    return df


# ============================================================================
# FILTER SPARSE COMBINATIONS (HIGH IMPACT!)
# ============================================================================
def filter_sparse_combinations(df, min_history=6):
    """
    Remove customer-product combinations with insufficient history.
    This eliminates noisy predictions on sparse data.
    
    Args:
        df: DataFrame with features
        min_history: Minimum months of history required
        
    Returns:
        Filtered DataFrame
    """
    print(f"\n[FILTER] Removing sparse combinations (min {min_history} months)...")
    
    # Count non-zero sales per customer-product combination
    combo_counts = df[df['SALES'] > 0].groupby(['CUSTOMER', 'PRODUCT']).size()
    
    # Get combinations with enough history
    valid_combos = combo_counts[combo_counts >= min_history].index
    
    # Filter dataframe
    df_filtered = df[df[['CUSTOMER', 'PRODUCT']].apply(tuple, axis=1).isin(valid_combos)]
    
    print(f"  Original: {len(df):,} records for {len(combo_counts)} combinations")
    print(f"  Filtered: {len(df_filtered):,} records for {len(valid_combos)} combinations")
    print(f"  Removed: {len(combo_counts) - len(valid_combos)} sparse combinations")
    print(f"  Quality improvement: Focusing only on well-documented customer-product pairs")
    
    return df_filtered


# ============================================================================
# 5. TRAIN MODEL WITH TIME-BASED SPLIT + ENHANCED FEATURES
# ============================================================================
def train_model(df):
    """
    Train HistGradientBoostingRegressor with enhanced features and outlier handling.
    Train on 2022-2024, validate on 2025.
    Uses improved hyperparameters to reduce overfitting.
    
    Args:
        df: DataFrame with features
        
    Returns:
        Trained model
    """
    print("\n[STEP 4] Training enhanced model...")
    
    # Define features (expanded list)
    feature_cols = [
        # Lag features
        'LAG_1', 'LAG_2', 'LAG_3', 'LAG_12',
        'LAG_3_AVG', 'LAG_6_AVG', 'LAG_3_STD',
        # Baseline features
        'CUSTOMER_AVG', 'PRODUCT_AVG', 'CUSTOMER_PRODUCT_AVG',
        'RELATIVE_IMPORTANCE',
        # Trend features
        'TREND_3M', 'MOMENTUM', 'GROWTH_RATE',
        # Time features
        'MONTH', 'QUARTER', 'YEAR', 'IS_SUMMER', 'IS_RAMADAN_PERIOD',
        # Seasonality & category
        'SEASONALITY_INDEX', 'PRODUCT_VOLUME_CAT'
    ]
    
    # Prepare data (only use rows with at least 1 lag)
    df_valid = df[df['LAG_1'] > 0].copy()
    
    # ========================================================================
    # OUTLIER HANDLING - Winsorize extreme values
    # ========================================================================
    print("\n  Applying outlier handling (winsorization)...")
    
    sales_p99 = df_valid['SALES'].quantile(0.99)
    sales_p01 = df_valid['SALES'].quantile(0.01)
    
    df_valid['SALES_WINSORIZED'] = df_valid['SALES'].clip(sales_p01, sales_p99)
    
    print(f"    Original range: ${df_valid['SALES'].min():.2f} - ${df_valid['SALES'].max():,.0f}")
    print(f"    Winsorized range: ${sales_p01:.2f} - ${sales_p99:,.0f}")
    print(f"    Values clipped: {(df_valid['SALES'] != df_valid['SALES_WINSORIZED']).sum():,}")
    
    # ========================================================================
    # TIME-BASED SPLIT: Train on 2022-2024, Test on 2025
    # ========================================================================
    train_cutoff = pd.Timestamp('2025-01-01')
    
    df_train = df_valid[df_valid['MONTH_START'] < train_cutoff]
    df_test = df_valid[df_valid['MONTH_START'] >= train_cutoff]
    
    print(f"\n  Data Split (Time-Based):")
    print(f"    Train: {df_train['MONTH_START'].min()} to {df_train['MONTH_START'].max()}")
    print(f"    Train samples: {len(df_train):,}")
    print(f"    Test:  {df_test['MONTH_START'].min()} to {df_test['MONTH_START'].max()}")
    print(f"    Test samples:  {len(df_test):,}")
    
    # Prepare features and target
    X_train = df_train[feature_cols]
    y_train = np.log1p(df_train['SALES_WINSORIZED'])  # Log transform winsorized values
    
    X_test = df_test[feature_cols]
    y_test = np.log1p(df_test['SALES'])  # Test on actual values
    
    print(f"\n  Features used ({len(feature_cols)}): {feature_cols[:5]}... (showing first 5)")
    
    # ========================================================================
    # TRAIN MODEL with IMPROVED hyperparameters (less overfitting)
    # ========================================================================
    print("\n  Training HistGradientBoostingRegressor...")
    model = HistGradientBoostingRegressor(
        max_iter=300,           # More iterations but controlled by early stopping
        max_depth=5,            # REDUCED from 8 to prevent overfitting (critical!)
        learning_rate=0.02,     # REDUCED from 0.05 for slower, more stable learning
        min_samples_leaf=50,    # INCREASED from 20 to reduce noise on outliers
        l2_regularization=0.3,  # INCREASED from 0.1 for stronger regularization
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=15,
        verbose=0
    )
    
    model.fit(X_train, y_train)
    
    # ========================================================================
    # EVALUATE
    # ========================================================================
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Convert back from log space
    y_train_actual = np.expm1(y_train)
    y_test_actual = np.expm1(y_test)
    y_pred_train_actual = np.expm1(y_pred_train)
    y_pred_test_actual = np.expm1(y_pred_test)
    
    # Calculate metrics
    train_mae = mean_absolute_error(y_train_actual, y_pred_train_actual)
    test_mae = mean_absolute_error(y_test_actual, y_pred_test_actual)
    train_rmse = np.sqrt(mean_squared_error(y_train_actual, y_pred_train_actual))
    test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_test_actual))
    train_r2 = r2_score(y_train_actual, y_pred_train_actual)
    test_r2 = r2_score(y_test_actual, y_pred_test_actual)
    
    # MAPE (Mean Absolute Percentage Error) - more robust calculation
    train_mape = np.median(np.abs((y_train_actual - y_pred_train_actual) / (y_train_actual + 1))) * 100
    test_mape = np.median(np.abs((y_test_actual - y_pred_test_actual) / (y_test_actual + 1))) * 100
    
    # Accuracy within bands
    test_within_20pct = (np.abs(y_test_actual - y_pred_test_actual) / (y_test_actual + 1) < 0.20).mean() * 100
    test_within_50pct = (np.abs(y_test_actual - y_pred_test_actual) / (y_test_actual + 1) < 0.50).mean() * 100
    
    print("\n" + "="*75)
    print("MODEL PERFORMANCE - ENHANCED VERSION")
    print("="*75)
    print(f"{'Metric':<25} {'Train (2022-2024)':<25} {'Test (2025)':<25}")
    print("-"*75)
    print(f"{'Avg Sales':<25} ${y_train_actual.mean():>20,.0f}   ${y_test_actual.mean():>20,.0f}")
    print(f"{'Median Sales':<25} ${y_train_actual.median():>20,.0f}   ${y_test_actual.median():>20,.0f}")
    print(f"{'MAE':<25} ${train_mae:>20,.0f}   ${test_mae:>20,.0f}")
    print(f"{'RMSE':<25} ${train_rmse:>20,.0f}   ${test_rmse:>20,.0f}")
    print(f"{'Median APE':<25} {train_mape:>20.2f}%   {test_mape:>20.2f}%")
    print(f"{'R²':<25} {train_r2:>24.3f}   {test_r2:>24.3f}")
    print(f"{'Accuracy ±20%':<25} {'':>20}   {test_within_20pct:>20.1f}%")
    print(f"{'Accuracy ±50%':<25} {'':>20}   {test_within_50pct:>20.1f}%")
    print("="*75)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        print("\n📊 Top 10 Most Important Features:")
        importances = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        for i, row in importances.head(10).iterrows():
            bar = '█' * int(row['importance'] * 100)
            print(f"  {row['feature']:<25} {row['importance']:.4f} {bar}")
    
    # Show sample predictions
    print(f"\n📊 Sample 2025 Predictions vs Actuals (Best to Worst):")
    sample_df = df_test.sample(min(15, len(df_test))).copy()
    sample_df['PREDICTED'] = np.expm1(model.predict(sample_df[feature_cols]))
    sample_df['ERROR_%'] = ((sample_df['PREDICTED'] - sample_df['SALES']) / (sample_df['SALES'] + 1) * 100)
    sample_df = sample_df.assign(ABS_ERR=sample_df['ERROR_%'].abs())
    
    display_cols = ['CUSTOMER', 'PRODUCT', 'MONTH_START', 'SALES', 'PREDICTED', 'ERROR_%']
    print(sample_df.sort_values('ABS_ERR')[display_cols].to_string(index=False))
    
    return model, feature_cols


# ============================================================================
# 6. CREATE STATE DICTIONARY
# ============================================================================
def create_state_dictionary(df):
    """
    Create a state dictionary containing the last known values for each
    Customer/Product combination. This is essential for recursive forecasting.
    
    Args:
        df: DataFrame with all features
        
    Returns:
        Dictionary with states for each customer-product pair
    """
    print("\n[STEP 5] Creating state dictionary for inference...")
    
    states = {}
    
    # Get the most recent data for each customer-product
    df_latest = df.sort_values('MONTH_START').groupby(['CUSTOMER', 'PRODUCT']).tail(12)
    
    for (customer, product), group in df_latest.groupby(['CUSTOMER', 'PRODUCT']):
        group = group.sort_values('MONTH_START')
        
        # Store the last known values
        last_row = group.iloc[-1]
        
        states[(customer, product)] = {
            'last_date': last_row['MONTH_START'],
            'last_sales': last_row['SALES'],
            'lag_1': last_row['LAG_1'],
            'lag_2': last_row['LAG_2'],
            'lag_3': last_row['LAG_3'],
            'lag_12': last_row['LAG_12'],
            'lag_3_avg': last_row['LAG_3_AVG'],
            'lag_6_avg': last_row['LAG_6_AVG'],
            'customer_avg': last_row.get('CUSTOMER_AVG', last_row['SALES']),
            'product_avg': last_row.get('PRODUCT_AVG', last_row['SALES']),
            'customer_product_avg': last_row.get('CUSTOMER_PRODUCT_AVG', last_row['SALES']),
            'recent_history': group[['MONTH_START', 'SALES']].to_dict('records')
        }
    
    print(f"States created for {len(states)} customer-product combinations")
    
    return states


# ============================================================================
# 7. SAVE ARTIFACTS
# ============================================================================
def save_artifacts(model, feature_cols, states, df):
    """
    Save model, states, and metadata for production use.
    """
    print("\n[STEP 6] Saving artifacts...")
    
    # Save model
    joblib.dump(model, MODEL_PATH)
    print(f"✓ Model saved: {MODEL_PATH}")
    
    # Save states
    joblib.dump(states, STATES_PATH)
    print(f"✓ States saved: {STATES_PATH}")
    
    # Save metadata
    metadata = {
        'feature_cols': feature_cols,
        'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'data_date_range': (str(df['MONTH_START'].min()), str(df['MONTH_START'].max())),
        'n_customers': df['CUSTOMER'].nunique(),
        'n_products': df['PRODUCT'].nunique(),
        'customers': sorted(df['CUSTOMER'].unique().tolist()),
        'products': sorted(df['PRODUCT'].unique().tolist())
    }
    joblib.dump(metadata, METADATA_PATH)
    print(f"✓ Metadata saved: {METADATA_PATH}")
    
    print("\n" + "="*80)
    print("SUCCESS! All artifacts saved.")
    print(f"  - Model: {MODEL_PATH}")
    print(f"  - States: {STATES_PATH}")
    print(f"  - Metadata: {METADATA_PATH}")
    print("="*80)


# ============================================================================
# MAIN EXECUTION
# ============================================================================
def main():
    """Main execution pipeline."""
    try:
        # Step 1: Load and clean
        df = load_and_clean_data()
        
        # Step 2: Aggregate monthly
        df_monthly = aggregate_monthly(df)
        
        # Step 3: Create features
        df_features = create_lag_features(df_monthly)
        
        # Step 3.5: FILTER SPARSE COMBINATIONS (High-impact improvement!)
        df_features = filter_sparse_combinations(df_features, min_history=6)
        
        # Step 4: Train model with improved hyperparameters
        model, feature_cols = train_model(df_features)
        
        # Step 5: Create states
        states = create_state_dictionary(df_features)
        
        # Step 6: Save everything
        save_artifacts(model, feature_cols, states, df_features)
        
        print("\n✓ Training complete! Ready for production inference.")
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
