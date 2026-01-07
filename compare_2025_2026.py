"""
Compare 2025 actual (by product+quarter) with 2026 forecast (by product+quarter).
Outputs a side-by-side summary showing trends.
"""

import pandas as pd
from pathlib import Path

OUTPUT_DIR = Path("output")

# Load 2025 actual quarterly
actual_2025 = pd.read_csv(OUTPUT_DIR / "quarterly_claims_by_product_year.csv")
actual_2025 = actual_2025[actual_2025['YEAR'] == 2025]
actual_2025 = actual_2025.rename(columns={'PRODUCT_GROUP': 'Product'})
actual_2025 = actual_2025[['Product', 'Q1', 'Q2', 'Q3', 'Q4']]

# Load 2026 forecast quarterly
forecast_2026 = pd.read_csv(OUTPUT_DIR / "task2_forecast_2026_quarterly.csv")
forecast_2026 = forecast_2026.rename(columns={'PRODUCT_GROUP': 'Product'})

# Merge
comparison = actual_2025.merge(forecast_2026, on='Product', how='outer', suffixes=('_2025', '_2026'))

# Fill NaN with 0
for col in comparison.columns:
    if col != 'Product':
        comparison[col] = comparison[col].fillna(0).astype(float)

# Calculate totals
comparison['Total_2025'] = comparison[['Q1_2025', 'Q2_2025', 'Q3_2025', 'Q4_2025']].sum(axis=1)
comparison['Total_2026'] = comparison[['Q1_2026', 'Q2_2026', 'Q3_2026', 'Q4_2026']].sum(axis=1)
comparison['YoY_Change_%'] = (
    (comparison['Total_2026'] - comparison['Total_2025']) / (comparison['Total_2025'] + 1) * 100
).round(2)

# Reorder columns
cols_order = [
    'Product',
    'Q1_2025', 'Q2_2025', 'Q3_2025', 'Q4_2025', 'Total_2025',
    'Q1_2026', 'Q2_2026', 'Q3_2026', 'Q4_2026', 'Total_2026',
    'YoY_Change_%'
]
comparison = comparison[cols_order].sort_values('Total_2026', ascending=False)

# Format for display
display_df = comparison.copy()
for col in comparison.columns:
    if col not in ['Product', 'YoY_Change_%']:
        display_df[col] = display_df[col].apply(lambda x: f"${x:,.0f}")
    elif col == 'YoY_Change_%':
        display_df[col] = display_df[col].apply(lambda x: f"{x:+.1f}%")

# Save
out_path = OUTPUT_DIR / "task2_2025_vs_2026_by_product_quarter.csv"
comparison.to_csv(out_path, index=False)

print("2025 ACTUAL vs 2026 FORECAST (by Product Group & Quarter)")
print("=" * 120)
print(display_df.to_string(index=False))
print("\n" + "=" * 120)
print(f"Saved: {out_path}")

# Summary
print("\nTOP PRODUCTS BY 2026 FORECAST:")
top5 = comparison.nlargest(5, 'Total_2026')[['Product', 'Total_2025', 'Total_2026', 'YoY_Change_%']]
for idx, row in top5.iterrows():
    print(f"  {row['Product']:25s} | 2025: ${row['Total_2025']:>12,.0f} | 2026: ${row['Total_2026']:>12,.0f} | YoY: {row['YoY_Change_%']:>+6.1f}%")
