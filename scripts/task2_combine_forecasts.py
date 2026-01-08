"""
Task 2: Combine ROUAE and ADH-AIN forecasts and show predicted claim amounts per product group
"""

import pandas as pd
import numpy as np

# Load the market-season forecast data
market_forecast = pd.read_csv('output/task2_2026_market_season_pivot.csv')

print("=" * 80)
print("TASK 2: COMBINED FORECAST (ROUAE + ADH-AIN)")
print("=" * 80)

print("\n📊 Input data shape:", market_forecast.shape)
print("Unique markets:", market_forecast['MARKET'].unique().tolist())
print("Unique products:", market_forecast['PRODUCT_GROUP'].nunique())

# Aggregate across markets (combine ROUAE + ADH_AIN)
combined = (
    market_forecast
    .groupby('PRODUCT_GROUP')[['Winter', 'Spring', 'Summer', 'Fall', 'Festival', 'TOTAL_2026']]
    .sum()
    .reset_index()
)

# Sort by total forecast (descending)
combined = combined.sort_values('TOTAL_2026', ascending=False).reset_index(drop=True)

print("\n📈 PREDICTED 2026 CLAIM AMOUNTS BY PRODUCT GROUP (Combined ROUAE + ADH-AIN):")
print("=" * 100)

# Add rank column
combined.insert(0, 'Rank', range(1, len(combined) + 1))

# Format output
display_df = combined.copy()
for col in ['Winter', 'Spring', 'Summer', 'Fall', 'Festival', 'TOTAL_2026']:
    display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")

print(display_df.to_string(index=False))

# Summary statistics
print("\n" + "=" * 100)
print("SUMMARY STATISTICS:")
print(f"  Total products: {len(combined)}")
print(f"  Total 2026 forecast (all products): ${combined['TOTAL_2026'].sum():,.0f}")
print(f"  Average per product: ${combined['TOTAL_2026'].mean():,.0f}")
print(f"  Median per product: ${combined['TOTAL_2026'].median():,.0f}")
print(f"  Min: ${combined['TOTAL_2026'].min():,.0f}")
print(f"  Max: ${combined['TOTAL_2026'].max():,.0f}")

# Top 10 products
print("\n🏆 TOP 10 PRODUCTS BY 2026 FORECAST:")
print("=" * 100)
top10 = combined.head(10).copy()
for col in ['Winter', 'Spring', 'Summer', 'Fall', 'Festival', 'TOTAL_2026']:
    top10[col] = top10[col].apply(lambda x: f"${x:,.0f}")
print(top10[['Rank', 'PRODUCT_GROUP', 'Winter', 'Spring', 'Summer', 'Fall', 'Festival', 'TOTAL_2026']].to_string(index=False))

# Bottom 10 products
print("\n📉 BOTTOM 10 PRODUCTS BY 2026 FORECAST:")
print("=" * 100)
bottom10 = combined.tail(10).copy()
for col in ['Winter', 'Spring', 'Summer', 'Fall', 'Festival', 'TOTAL_2026']:
    bottom10[col] = bottom10[col].apply(lambda x: f"${x:,.0f}")
print(bottom10[['Rank', 'PRODUCT_GROUP', 'Winter', 'Spring', 'Summer', 'Fall', 'Festival', 'TOTAL_2026']].to_string(index=False))

# Market breakdown for context
print("\n" + "=" * 100)
print("MARKET BREAKDOWN (for reference):")
print("=" * 100)
market_totals = market_forecast.groupby('MARKET')['TOTAL_2026'].sum()
print(f"  ROUAE total: ${market_totals.get('ROUAE', 0):,.0f}")
print(f"  ADH_AIN total: ${market_totals.get('ADH_AIN', 0):,.0f}")
print(f"  Combined total: ${market_totals.sum():,.0f}")

# Save combined output
combined.to_csv('output/task2_2026_combined_forecast_by_product.csv', index=False)
print("\n✅ Saved combined forecast: output/task2_2026_combined_forecast_by_product.csv")
