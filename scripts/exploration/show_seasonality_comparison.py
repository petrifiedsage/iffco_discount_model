"""Quick comparison of old vs new seasonality"""
import pandas as pd

print('='*80)
print('ENHANCED SEASONALITY: TOP 20 MOST SEASONAL PRODUCTS')
print('='*80)

df = pd.read_csv('models/product_seasonality_enhanced.csv')

# Calculate variance for each product
variance = df.groupby('PRODUCT_GROUP')['SEASONALITY_INDEX'].var().sort_values(ascending=False)

print(f'\nRank | Product                        | Variance     | Range              | Months')
print('-'*80)

for i, (product, var) in enumerate(variance.head(20).items(), 1):
    indices = df[df['PRODUCT_GROUP'] == product]['SEASONALITY_INDEX']
    min_idx = indices.min()
    max_idx = indices.max()
    num_months = len(indices)
    
    range_str = f'{min_idx:.2f}x - {max_idx:.2f}x'
    product_str = str(product)[:28]
    print(f'{i:4} | {product_str:<30} | {var:12.4f} | {range_str:<18} | {num_months:6}')

print('\n' + '='*80)
print('COMPARISON: Old vs New Seasonality')
print('='*80)

old_df = pd.read_csv('models/product_seasonality_claims.csv')
new_df = pd.read_csv('models/product_seasonality_enhanced.csv')

print(f'\nOld seasonality file (product_seasonality_claims.csv):')
print(f'  Rows: {len(old_df):,}')
print(f'  Unique products: {old_df["PRODUCT_GROUP"].nunique()}')
print(f'  Index range: {old_df["SEASONALITY_INDEX"].min():.2f}x - {old_df["SEASONALITY_INDEX"].max():.2f}x')

print(f'\nNew seasonality file (product_seasonality_enhanced.csv):')
print(f'  Rows: {len(new_df):,}')
print(f'  Unique products: {new_df["PRODUCT_GROUP"].nunique()}')
print(f'  Index range: {new_df["SEASONALITY_INDEX"].min():.2f}x - {new_df["SEASONALITY_INDEX"].max():.2f}x')

improvement = (new_df['PRODUCT_GROUP'].nunique() - old_df['PRODUCT_GROUP'].nunique()) / old_df['PRODUCT_GROUP'].nunique() * 100

print(f'\n✓ Coverage improvement: +{improvement:.0f}% ({new_df["PRODUCT_GROUP"].nunique() - old_df["PRODUCT_GROUP"].nunique()} more products)')
print(f'✓ Seasonality range improvement: {(new_df["SEASONALITY_INDEX"].max() / old_df["SEASONALITY_INDEX"].max()):.1f}x higher peak')
print(f'✓ Data source: 1.04M+ multi-year claims (FY21-26)')
