"""
Generate 2026 forecasts at Market × Product × Season granularity.
Answers: "What sales will Ice Creams achieve in UAE during Summer 2026?"

Output: CSV with columns: MARKET, PRODUCT_GROUP, SEASON, PREDICTED_SALES
"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib

MODEL_DIR = Path("models")
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "task2_market_seasonal_model.pkl"
METADATA_PATH = MODEL_DIR / "task2_market_seasonal_metadata.pkl"
HISTORY_PATH = MODEL_DIR / "history_market_seasonal.pkl"
SEASONALITY_PATH = MODEL_DIR / "market_product_seasonality.csv"

print("="*80)
print("TASK 2: 2026 MARKET-SEASON FORECASTING")
print("="*80)

# Load model artifacts
print("\nLoading model artifacts...")
model = joblib.load(MODEL_PATH)
metadata = joblib.load(METADATA_PATH)
history_df = joblib.load(HISTORY_PATH)
seasonality_df = pd.read_csv(SEASONALITY_PATH)

print(f"✓ Model loaded")
print(f"  Markets: {metadata['markets']}")
print(f"  Products: {len(metadata['products'])}")
print(f"  Seasons: {metadata['seasons']}")
print(f"  Feature count: {len(metadata['feature_cols'])}")

feature_cols = metadata['feature_cols']

# Season order for sequential forecasting (business seasons)
season_order = ['Winter', 'Spring', 'Summer', 'Fall', 'Festival']

# ============================================================================
# Build 2026 forecast base
# ============================================================================
print("\nBuilding 2026 forecast dataset...")

forecast_rows = []

for market in metadata['markets']:
    for product in metadata['products']:
        # Get historical data for this market-product
        hist = history_df[(history_df['MARKET'] == market) & 
                         (history_df['PRODUCT_GROUP'] == product)]
        
        if hist.empty:
            continue
        
        # Get 2024 data as base (most recent complete year)
        hist_2024 = hist[hist['YEAR'] == 2024].copy()
        hist_2025 = hist[hist['YEAR'] == 2025].copy()
        
        # Use 2025 if available, else 2024
        base_year = hist_2025 if not hist_2025.empty else hist_2024
        
        if base_year.empty:
            continue
        
        # Create forecast rows for each season in 2026
        for season in season_order:
            # Get lag from previous season and same season last year
            prev_season_idx = season_order.index(season) - 1
            prev_season = season_order[prev_season_idx]
            
            # LAG_1_SEASON: previous season in 2025
            lag_1_data = hist_2025[hist_2025['SEASON'] == prev_season]
            lag_1_season = lag_1_data['SALES'].values[0] if not lag_1_data.empty else base_year['SALES'].mean()
            
            # LAG_4_SEASON: same season in 2025 (year ago)
            lag_4_data = hist_2025[hist_2025['SEASON'] == season]
            lag_4_season = lag_4_data['SALES'].values[0] if not lag_4_data.empty else base_year['SALES'].mean()
            
            # LAG_2_SEASON_AVG: average of last 2 seasons
            if season_order.index(season) >= 2:
                recent_seasons = [season_order[season_order.index(season) - 1], 
                                season_order[season_order.index(season) - 2]]
                lag_2_avg = hist_2025[hist_2025['SEASON'].isin(recent_seasons)]['SALES'].mean()
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
                (history_df['YEAR'] == 2025) & 
                (history_df['SEASON'] == season)
            ]['SALES'].sum()
            
            market_product_share = (lag_4_season / (market_total + 1)) * 100 if market_total > 0 else 5.0
            
            # YoY growth
            same_season_2024 = hist_2024[hist_2024['SEASON'] == season]
            yoy_growth = 0.0
            if not same_season_2024.empty and not lag_4_data.empty:
                prev_val = same_season_2024['SALES'].values[0]
                if prev_val > 0:
                    yoy_growth = ((lag_4_season - prev_val) / prev_val) * 100
            
            # Season encoding
            season_encoded = season_order.index(season)
            
            # Promo features - assume no promotions for 2026 forecast (conservative)
            promo_count = 0
            promo_log = 0
            
            forecast_rows.append({
                'MARKET': market,
                'PRODUCT_GROUP': product,
                'SEASON': season,
                'YEAR': 2026,
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
print(f"Built {len(forecast_df)} forecast scenarios")

# ============================================================================
# Generate predictions
# ============================================================================
print("\nGenerating 2026 predictions...")

X_forecast = forecast_df[feature_cols]
y_pred_log = model.predict(X_forecast)
y_pred = np.expm1(y_pred_log)

forecast_df['PREDICTED_SALES'] = np.maximum(0, y_pred)

# ============================================================================
# Save outputs
# ============================================================================

# Full forecast
output_path = OUTPUT_DIR / "task2_2026_market_season_forecast.csv"
output_cols = ['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'PREDICTED_SALES']
forecast_df[output_cols].to_csv(output_path, index=False)

print(f"\n✓ Saved: {output_path}")

# Create pivot table for easy viewing
pivot = forecast_df.pivot_table(
    index=['MARKET', 'PRODUCT_GROUP'],
    columns='SEASON',
    values='PREDICTED_SALES',
    aggfunc='sum'
)
pivot = pivot.reindex(columns=season_order)
pivot['TOTAL_2026'] = pivot.sum(axis=1)
pivot = pivot.sort_values('TOTAL_2026', ascending=False)

pivot_path = OUTPUT_DIR / "task2_2026_market_season_pivot.csv"
pivot.to_csv(pivot_path)

print(f"✓ Saved: {pivot_path}")

# ============================================================================
# Display results
# ============================================================================

print("\n" + "="*80)
print("2026 FORECAST - TOP MARKET-PRODUCT COMBINATIONS")
print("="*80)

display_df = pivot.head(15).copy()
for col in display_df.columns:
    display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")

print(display_df.to_string())

print("\n" + "="*80)
print("EXAMPLE QUERIES")
print("="*80)

# Show some business question examples
examples = [
    ('ROUAE', 'Ice creams', 'Q2'),
    ('ROUAE', 'Oils', 'Q1'),
    ('ADH_AIN', 'Biscuits', 'Q4'),
]

for market, product, season in examples:
    result = forecast_df[
        (forecast_df['MARKET'] == market) & 
        (forecast_df['PRODUCT_GROUP'] == product) & 
        (forecast_df['SEASON'] == season)
    ]
    if not result.empty:
        sales = result['PREDICTED_SALES'].values[0]
        print(f"Q: What sales will {product} achieve in {market} during {season} 2026?")
        print(f"A: ${sales:,.0f}")
        print()

print("="*80)
print("✓ FORECASTING COMPLETE")
print("="*80)
