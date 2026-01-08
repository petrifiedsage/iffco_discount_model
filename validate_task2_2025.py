"""
Validate Task 2 Market-Season Model:
- Trains on 2022-2024 data
- Forecasts 2025 data
- Compares forecast vs actual 2025
- Shows accuracy metrics and errors
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = Path("models/task2_market_seasonal_model.pkl")
METADATA_PATH = Path("models/task2_market_seasonal_metadata.pkl")
HISTORY_PATH = Path("models/history_market_seasonal.pkl")
SEASONALITY_PATH = Path("models/market_product_seasonality.csv")

print("="*100)
print("TASK 2: VALIDATION - FORECAST 2025 vs ACTUAL 2025")
print("="*100)

# Load model artifacts
print("\nLoading trained model (trained on 2022-2024)...")
model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
history_df = joblib.load(HISTORY_PATH)
seasonality_df = pd.read_csv(SEASONALITY_PATH)

print(f"✓ Model loaded")
print(f"  Markets: {metadata['markets']}")
print(f"  Products: {len(metadata['products'])}")
print(f"  Seasons: {metadata['seasons']}")

feature_cols = metadata['feature_cols']

# Season order
season_order = ['Winter', 'Spring', 'Summer', 'Fall', 'Festival']

# ============================================================================
# Build 2025 forecast dataset
# ============================================================================
print("\nBuilding 2025 forecast dataset...")

forecast_rows = []

for market in metadata['markets']:
    for product in metadata['products']:
        # Get historical data for this market-product
        hist = history_df[(history_df['MARKET'] == market) & 
                         (history_df['PRODUCT_GROUP'] == product)]
        
        if hist.empty:
            continue
        
        # Get 2024 data as base
        hist_2024 = hist[hist['YEAR'] == 2024].copy()
        hist_2023 = hist[hist['YEAR'] == 2023].copy()
        
        if hist_2024.empty:
            continue
        
        base_year = hist_2024
        
        # Create forecast rows for each season in 2025
        for season in season_order:
            # Get lag from previous season and same season last year
            prev_season_idx = season_order.index(season) - 1
            prev_season = season_order[prev_season_idx]
            
            # LAG_1_SEASON: previous season in 2024
            lag_1_data = hist_2024[hist_2024['SEASON'] == prev_season]
            lag_1_season = lag_1_data['SALES'].values[0] if not lag_1_data.empty else base_year['SALES'].mean()
            
            # LAG_4_SEASON: same season in 2024 (year ago)
            lag_4_data = hist_2024[hist_2024['SEASON'] == season]
            lag_4_season = lag_4_data['SALES'].values[0] if not lag_4_data.empty else base_year['SALES'].mean()
            
            # LAG_2_SEASON_AVG
            if season_order.index(season) >= 2:
                recent_seasons = [season_order[season_order.index(season) - 1], 
                                season_order[season_order.index(season) - 2]]
                lag_2_avg = hist_2024[hist_2024['SEASON'].isin(recent_seasons)]['SALES'].mean()
            else:
                lag_2_avg = lag_1_season
            
            # Get seasonality index
            season_data = seasonality_df[
                (seasonality_df['MARKET'] == market) & 
                (seasonality_df['PRODUCT_GROUP'] == product) & 
                (seasonality_df['SEASON'] == season)
            ]
            seasonality_index = season_data['SEASONALITY_INDEX'].values[0] if not season_data.empty else 1.0
            
            # Calculate features
            market_product_avg = hist['SALES'].mean()
            market_avg = history_df[history_df['MARKET'] == market]['SALES'].mean()
            product_avg = history_df[history_df['PRODUCT_GROUP'] == product]['SALES'].mean()
            customer_count = base_year['CUSTOMER_COUNT'].mean()
            
            # Market share
            market_total = history_df[
                (history_df['MARKET'] == market) & 
                (history_df['YEAR'] == 2024) & 
                (history_df['SEASON'] == season)
            ]['SALES'].sum()
            
            market_product_share = (lag_4_season / (market_total + 1)) * 100 if market_total > 0 else 5.0
            
            # YoY growth (2023 vs 2024)
            same_season_2023 = hist_2023[hist_2023['SEASON'] == season]
            yoy_growth = 0.0
            if not same_season_2023.empty and not lag_4_data.empty:
                prev_val = same_season_2023['SALES'].values[0]
                if prev_val > 0:
                    yoy_growth = ((lag_4_season - prev_val) / prev_val) * 100
            
            # Season encoding
            season_encoded = season_order.index(season)
            
            # Promo features - use historical promo data for 2024
            promo_count = 0
            promo_log = 0
            
            forecast_rows.append({
                'MARKET': market,
                'PRODUCT_GROUP': product,
                'SEASON': season,
                'YEAR': 2025,
                'LAG_1_SEASON': lag_1_season,
                'LAG_4_SEASON': lag_4_season,
                'LAG_2_SEASON_AVG': lag_2_avg,
                'MARKET_PRODUCT_SHARE': market_product_share,
                'CUSTOMER_COUNT': customer_count,
                'MARKET_PRODUCT_AVG': market_product_avg,
                'MARKET_AVG': market_avg,
                'PRODUCT_AVG': product_avg,
                'YOY_GROWTH': yoy_growth,
                'SEASON_ENCODED': season_encoded,
                'SEASONALITY_INDEX': seasonality_index,
                'PROMO_COUNT': promo_count,
                'PROMO_LOG': promo_log
            })

forecast_df = pd.DataFrame(forecast_rows)
print(f"Built {len(forecast_df)} forecast scenarios for 2025")

# ============================================================================
# Generate 2025 predictions
# ============================================================================
print("\nGenerating 2025 predictions...")

X_forecast = forecast_df[feature_cols]
y_pred_log = model.predict(X_forecast)
y_pred = np.expm1(y_pred_log)

forecast_df['PREDICTED_SALES'] = np.maximum(0, y_pred)

# ============================================================================
# Get actual 2025 data
# ============================================================================
print("\nRetrieving actual 2025 data...")

# Get 2025 actuals from history
actual_2025 = history_df[history_df['YEAR'] == 2025][['MARKET', 'PRODUCT_GROUP', 'SEASON', 'SALES']].copy()
actual_2025.rename(columns={'SALES': 'ACTUAL_SALES'}, inplace=True)

print(f"Found {len(actual_2025)} actual 2025 records")

# ============================================================================
# Merge forecasts with actuals
# ============================================================================
print("\nMerging forecasts with actuals...")

comparison = forecast_df[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'PREDICTED_SALES']].merge(
    actual_2025,
    on=['MARKET', 'PRODUCT_GROUP', 'SEASON'],
    how='inner'
)

print(f"Matched {len(comparison)} market-product-season combinations")

# ============================================================================
# Calculate metrics
# ============================================================================
print("\n" + "="*100)
print("OVERALL METRICS (2025 FORECAST vs ACTUAL)")
print("="*100)

y_actual = comparison['ACTUAL_SALES']
y_predicted = comparison['PREDICTED_SALES']

r2 = r2_score(y_actual, y_predicted)
mae = mean_absolute_error(y_actual, y_predicted)
rmse = np.sqrt(mean_squared_error(y_actual, y_predicted))
mape = np.mean(np.abs((y_actual - y_predicted) / (y_actual + 1))) * 100
median_ape = np.median(np.abs((y_actual - y_predicted) / (y_actual + 1))) * 100

print(f"\nR-squared (R²):           {r2:.4f}  (0-1, higher is better)")
print(f"MAE (Mean Absolute Error): ${mae:,.0f}  (avg prediction error)")
print(f"RMSE (Root Mean Sq Error): ${rmse:,.0f}  (penalizes large errors)")
print(f"MAPE (Mean Abs % Error):   {mape:.1f}%  (avg % error)")
print(f"Median APE:                {median_ape:.1f}%  (median % error)")

# Interpretation
if r2 >= 0.8:
    print(f"\n✅ EXCELLENT: R² {r2:.2f} indicates model explains {r2*100:.0f}% of variance")
elif r2 >= 0.7:
    print(f"\n✅ GOOD: R² {r2:.2f} indicates model explains {r2*100:.0f}% of variance")
elif r2 >= 0.5:
    print(f"\n⚠️  FAIR: R² {r2:.2f} indicates model explains {r2*100:.0f}% of variance")
else:
    print(f"\n❌ POOR: R² {r2:.2f} indicates weak predictive power")

if median_ape <= 15:
    print(f"✅ EXCELLENT: Median APE {median_ape:.1f}% is very accurate")
elif median_ape <= 25:
    print(f"✅ GOOD: Median APE {median_ape:.1f}% is acceptable")
elif median_ape <= 40:
    print(f"⚠️  FAIR: Median APE {median_ape:.1f}% needs improvement")
else:
    print(f"❌ POOR: Median APE {median_ape:.1f}% is too high")

# ============================================================================
# Metrics by Market
# ============================================================================
print("\n" + "="*100)
print("ACCURACY BY MARKET")
print("="*100)

for market in metadata['markets']:
    market_data = comparison[comparison['MARKET'] == market]
    if market_data.empty:
        continue
    
    y_a = market_data['ACTUAL_SALES']
    y_p = market_data['PREDICTED_SALES']
    
    r2_m = r2_score(y_a, y_p)
    mae_m = mean_absolute_error(y_a, y_p)
    ape_m = np.median(np.abs((y_a - y_p) / (y_a + 1))) * 100
    
    print(f"\n{market:12s} | R²: {r2_m:.3f} | MAE: ${mae_m:>10,.0f} | Median APE: {ape_m:>5.1f}% | {len(market_data)} combos")

# ============================================================================
# Metrics by Product
# ============================================================================
print("\n" + "="*100)
print("ACCURACY BY PRODUCT")
print("="*100)

by_product = []
for product in metadata['products']:
    prod_data = comparison[comparison['PRODUCT_GROUP'] == product]
    if prod_data.empty:
        continue
    
    y_a = prod_data['ACTUAL_SALES']
    y_p = prod_data['PREDICTED_SALES']
    
    r2_p = r2_score(y_a, y_p)
    ape_p = np.median(np.abs((y_a - y_p) / (y_a + 1))) * 100
    
    by_product.append({
        'Product': product,
        'R2': r2_p,
        'Median_APE': ape_p,
        'Count': len(prod_data)
    })

prod_df = pd.DataFrame(by_product).sort_values('Median_APE')
print(prod_df.to_string(index=False))

# ============================================================================
# Top Errors
# ============================================================================
print("\n" + "="*100)
print("LARGEST PREDICTION ERRORS (Top 15)")
print("="*100)

comparison['ERROR'] = np.abs(comparison['ACTUAL_SALES'] - comparison['PREDICTED_SALES'])
comparison['APE'] = np.abs((comparison['ACTUAL_SALES'] - comparison['PREDICTED_SALES']) / (comparison['ACTUAL_SALES'] + 1)) * 100

errors = comparison.nlargest(15, 'ERROR')[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'ACTUAL_SALES', 'PREDICTED_SALES', 'APE']]

for idx, row in errors.iterrows():
    print(f"\n{row['MARKET']:10s} | {row['PRODUCT_GROUP']:25s} | {row['SEASON']:8s}")
    print(f"  Actual: ${row['ACTUAL_SALES']:>12,.0f}  |  Predicted: ${row['PREDICTED_SALES']:>12,.0f}  |  Error: {row['APE']:>5.1f}%")

# ============================================================================
# Save detailed comparison
# ============================================================================
print("\n" + "="*100)
print("SAVING RESULTS")
print("="*100)

output_file = OUTPUT_DIR / "task2_2025_validation_forecast_vs_actual.csv"
comparison_output = comparison[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'ACTUAL_SALES', 'PREDICTED_SALES', 'APE']].copy()
comparison_output = comparison_output.sort_values(['MARKET', 'PRODUCT_GROUP', 'SEASON'])
comparison_output.to_csv(output_file, index=False)

print(f"\n✓ Saved detailed comparison to: {output_file}")

# Summary by Market-Product
summary_file = OUTPUT_DIR / "task2_2025_validation_summary.csv"
summary = comparison.groupby(['MARKET', 'PRODUCT_GROUP']).agg({
    'ACTUAL_SALES': 'sum',
    'PREDICTED_SALES': 'sum',
    'APE': 'median'
}).reset_index()
summary.rename(columns={'APE': 'MEDIAN_APE'}, inplace=True)
summary['ERROR_PERCENT'] = np.abs((summary['ACTUAL_SALES'] - summary['PREDICTED_SALES']) / (summary['ACTUAL_SALES'] + 1)) * 100
summary = summary.sort_values('ERROR_PERCENT')
summary.to_csv(summary_file, index=False)

print(f"✓ Saved summary to: {summary_file}")

print("\n" + "="*100)
print("✅ VALIDATION COMPLETE")
print("="*100)
print(f"\nConclusion:")
print(f"- Model trained on 2022-2024 data")
print(f"- Forecasted 2025 data")
print(f"- Compared {len(comparison)} market-product-season combinations")
print(f"- Overall Median APE: {median_ape:.1f}%")
print(f"- R² Score: {r2:.4f}")
print(f"\nFiles generated:")
print(f"1. {output_file.name}")
print(f"2. {summary_file.name}")
