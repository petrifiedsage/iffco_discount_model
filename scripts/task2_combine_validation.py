"""
Task 2: Combine ROUAE and ADH-AIN validation metrics and recalculate APE
Output format: PRODUCT_GROUP, SEASON, ACTUAL_SALES, PREDICTED_SALES, APE (no MARKET column)
"""

import pandas as pd
import numpy as np

# Load validation data
val_data = pd.read_csv('output/task2_2025_validation_forecast_vs_actual.csv')

print("=" * 80)
print("TASK 2: COMBINED VALIDATION METRICS (ROUAE + ADH-AIN)")
print("=" * 80)

print(f"\n📊 Input validation data shape: {val_data.shape}")
print(f"Unique markets: {val_data['MARKET'].unique().tolist()}")
print(f"Unique products: {val_data['PRODUCT_GROUP'].nunique()}")
print(f"Unique seasons: {val_data['SEASON'].nunique()}")

# Combine by PRODUCT_GROUP and SEASON (sum across MARKET)
combined_val = (
    val_data
    .groupby(['PRODUCT_GROUP', 'SEASON'], as_index=False)
    .agg({
        'ACTUAL_SALES': 'sum',
        'PREDICTED_SALES': 'sum'
    })
)

# Recalculate APE on combined values
# APE = |PREDICTED - ACTUAL| / ACTUAL * 100
combined_val['APE'] = (
    np.abs(combined_val['PREDICTED_SALES'] - combined_val['ACTUAL_SALES']) 
    / combined_val['ACTUAL_SALES'] * 100
).round(10)

# Sort by PRODUCT_GROUP then SEASON (Winter, Spring, Summer, Fall, Festival)
season_order = {'Winter': 0, 'Spring': 1, 'Summer': 2, 'Fall': 3, 'Festival': 4}
combined_val['season_sort'] = combined_val['SEASON'].map(season_order)
combined_val = combined_val.sort_values(['PRODUCT_GROUP', 'season_sort']).drop('season_sort', axis=1)

# Reorder columns
combined_val = combined_val[['PRODUCT_GROUP', 'SEASON', 'ACTUAL_SALES', 'PREDICTED_SALES', 'APE']]

print(f"\n📈 Combined validation data shape: {combined_val.shape}")
print(f"\nFirst 10 rows:")
print(combined_val.head(10).to_string(index=False))

# Summary statistics
print("\n" + "=" * 80)
print("SUMMARY STATISTICS:")
print("=" * 80)
print(f"  Total rows: {len(combined_val)}")
print(f"  Median APE: {combined_val['APE'].median():.2f}%")
print(f"  Mean APE: {combined_val['APE'].mean():.2f}%")
print(f"  Min APE: {combined_val['APE'].min():.2f}%")
print(f"  Max APE: {combined_val['APE'].max():.2f}%")
print(f"  Std Dev APE: {combined_val['APE'].std():.2f}%")

# Products with highest/lowest median APE
print("\n🏆 TOP 5 PRODUCTS BY MEDIAN APE (Best Performance):")
print("=" * 80)
product_ape = combined_val.groupby('PRODUCT_GROUP')['APE'].agg(['median', 'mean', 'count']).reset_index()
product_ape = product_ape.sort_values('median').head(5)
product_ape.columns = ['PRODUCT_GROUP', 'MEDIAN_APE', 'MEAN_APE', 'FORECAST_COUNT']
print(product_ape[['PRODUCT_GROUP', 'MEDIAN_APE', 'MEAN_APE', 'FORECAST_COUNT']].to_string(index=False))

print("\n📉 BOTTOM 5 PRODUCTS BY MEDIAN APE (Worst Performance):")
print("=" * 80)
product_ape_worst = combined_val.groupby('PRODUCT_GROUP')['APE'].agg(['median', 'mean', 'count']).reset_index()
product_ape_worst = product_ape_worst.sort_values('median', ascending=False).head(5)
product_ape_worst.columns = ['PRODUCT_GROUP', 'MEDIAN_APE', 'MEAN_APE', 'FORECAST_COUNT']
print(product_ape_worst[['PRODUCT_GROUP', 'MEDIAN_APE', 'MEAN_APE', 'FORECAST_COUNT']].to_string(index=False))

# Save output
combined_val.to_csv('output/task2_2025_validation_combined.csv', index=False)
print("\n✅ Saved combined validation: output/task2_2025_validation_combined.csv")
