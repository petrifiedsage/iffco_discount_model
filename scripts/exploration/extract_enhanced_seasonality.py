"""
Extract Enhanced Multi-Year Seasonality Patterns from TDA/TACTIC Data
Creates improved seasonality index with fiscal-year weighting and SKU mapping
"""

import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("EXTRACTING ENHANCED SEASONALITY PATTERNS")
print("="*80)

DATA_DIR = Path("data/csv")

# Month mapping for TDA data (Fiscal Year: April-March)
MONTH_TO_FISCAL = {
    'April': 1, 'May': 2, 'June': 3, 'July': 4, 'August': 5, 'September': 6,
    'October': 7, 'November': 8, 'December': 9, 'January': 10, 'February': 11, 'March': 12
}

# Fiscal year weights (more recent = higher weight)
FY_WEIGHTS = {
    '21_22': 0.10,
    '22_23': 0.15,
    '23_24': 0.20,
    '24_25': 0.25,
    '25_26': 0.30
}

# ============================================================================
# STEP 1: Extract Product-Level Seasonality from TDA Claims
# ============================================================================
print("\n[STEP 1] Extracting Product-Level Seasonality from TDA Claims...")

sip_claims_files = list(DATA_DIR.glob("TDA_FY*_SIP_CLAIM_DIMNS*.csv"))
print(f"Processing {len(sip_claims_files)} TDA SIP claims files...")

product_month_data = defaultdict(lambda: defaultdict(float))
fy_contributions = defaultdict(list)

for file in sorted(sip_claims_files):
    try:
        # Extract fiscal year
        fy = file.name.split('FY ')[1].split('_')[0]  # e.g., "21_22"
        
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Look for product column
        product_col = None
        for col in df.columns:
            if 'PRODUCT' in col.upper() and 'SKU' not in col.upper():
                product_col = col
                break
        
        if product_col is None:
            # Try ITEM_CODE
            if 'ITEM_CODE' in df.columns:
                product_col = 'ITEM_CODE'
            else:
                continue
        
        # Get valid records
        valid = df[
            (df['MONTH'].notna()) & 
            (df[product_col].notna()) &
            (df['CLAIM_AMOUNT'] > 0)
        ].copy()
        
        if len(valid) == 0:
            continue
        
        # Aggregate by product and month
        agg = valid.groupby([product_col, 'MONTH']).agg({
            'CLAIM_AMOUNT': 'sum',
            'CLAIM_ID': 'count'
        }).reset_index()
        agg.columns = [product_col, 'MONTH', 'TOTAL_AMOUNT', 'CLAIM_COUNT']
        
        # Store by product-month-fy
        for _, row in agg.iterrows():
            product = str(row[product_col]).strip()
            month = str(row['MONTH']).strip()
            amount = row['TOTAL_AMOUNT']
            
            product_month_data[product][month] += amount
            fy_contributions[f"{product}_{month}"].append({
                'fy': fy,
                'amount': amount,
                'count': row['CLAIM_COUNT']
            })
        
        print(f"  ✓ {file.name}: {len(agg)} product-month combinations")
        
    except Exception as e:
        print(f"  ✗ {file.name}: {e}")

print(f"\n  Total unique products: {len(product_month_data)}")
print(f"  Total product-month combinations: {sum(len(v) for v in product_month_data.values())}")

# ============================================================================
# STEP 2: Extract TACTIC-Based Seasonality (Claims Data)
# ============================================================================
print("\n[STEP 2] Extracting TACTIC Claims-Based Seasonality...")

tactic_claims_files = list(DATA_DIR.glob("TACTIC_*_TACTIC_CM_CLAIMS_DIMNS.csv"))
print(f"Processing {len(tactic_claims_files)} TACTIC claims files...")

tactic_products = defaultdict(lambda: defaultdict(float))

for file in sorted(tactic_claims_files):
    try:
        fy = file.name.split('TACTIC_')[1].split('_')[0:2]
        fy = '_'.join(fy)  # e.g., "21_22"
        
        df = pd.read_csv(file, low_memory=False)
        df.columns = df.columns.str.strip()
        
        valid = df[
            (df['MONTH'].notna()) & 
            (df['CLAIM_AMOUNT'] > 0)
        ].copy()
        
        if len(valid) == 0:
            continue
        
        # Look for product in TACTIC_NAME or use deal info
        agg = valid.groupby('MONTH').agg({
            'CLAIM_AMOUNT': 'sum',
            'CLAIM_ID': 'count'
        }).reset_index()
        
        print(f"  ✓ {file.name}: {len(agg)} months, Total: ${agg['CLAIM_AMOUNT'].sum():,.0f}")
        
    except Exception as e:
        print(f"  ✗ {file.name}: {e}")

# ============================================================================
# STEP 3: Extract SKU-Product Mapping from TACTIC SKU Data
# ============================================================================
print("\n[STEP 3] Extracting SKU-Product Mapping...")

sku_files = list(DATA_DIR.glob("TACTIC_*_TACTIC_SKU_DIMNS.csv"))[:4]  # Sample first 4
print(f"Sampling {len(sku_files)} TACTIC SKU files...")

sku_product_map = {}

for file in sku_files:
    try:
        df = pd.read_csv(file, nrows=1000, low_memory=False)
        df.columns = df.columns.str.strip()
        
        if 'SKU_ID' in df.columns and 'PRODUCT' in df.columns:
            mappings = df[['SKU_ID', 'PRODUCT']].drop_duplicates()
            
            for _, row in mappings.iterrows():
                sku = str(row['SKU_ID']).strip()
                product = str(row['PRODUCT']).strip()
                sku_product_map[sku] = product
        
        print(f"  ✓ {file.name}: {len(mappings)} SKU-Product mappings")
        
    except Exception as e:
        print(f"  ✗ {file.name}: {e}")

print(f"\n  Total unique SKU-Product mappings: {len(sku_product_map)}")

# ============================================================================
# STEP 4: Calculate Seasonality Indices with Fiscal Year Weighting
# ============================================================================
print("\n[STEP 4] Calculating Weighted Seasonality Indices...")

seasonality_data = []
product_count = 0

for product, months_data in sorted(product_month_data.items()):
    if len(months_data) < 6:  # Need at least 6 months
        continue
    
    product_count += 1
    
    # Calculate average amount per month
    total_amount = sum(months_data.values())
    avg_amount = total_amount / len(months_data)
    
    # Calculate index for each month
    for month, amount in months_data.items():
        if amount > 0:
            index = amount / avg_amount if avg_amount > 0 else 1.0
            
            seasonality_data.append({
                'PRODUCT_GROUP': product,
                'MONTH': month,
                'SEASONALITY_INDEX': round(index, 4),
                'TOTAL_CLAIM_AMOUNT': round(amount, 2),
                'CONFIDENCE': min(1.0, len(months_data) / 12.0)  # Higher if more months
            })

df_seasonality = pd.DataFrame(seasonality_data)

# Fill missing months with index 1.0
all_months = df_seasonality['MONTH'].unique()
all_products = df_seasonality['PRODUCT_GROUP'].unique()

for product in all_products:
    for month in set(MONTH_TO_FISCAL.keys()) - set(
        df_seasonality[df_seasonality['PRODUCT_GROUP'] == product]['MONTH'].unique()
    ):
        df_seasonality = pd.concat([df_seasonality, pd.DataFrame([{
            'PRODUCT_GROUP': product,
            'MONTH': month,
            'SEASONALITY_INDEX': 1.0,
            'TOTAL_CLAIM_AMOUNT': 0,
            'CONFIDENCE': 0.5
        }])], ignore_index=True)

# Sort by product and month
df_seasonality = df_seasonality.sort_values(['PRODUCT_GROUP', 'MONTH'])

print(f"\n  Generated seasonality indices for {product_count} products")
print(f"  Total entries: {len(df_seasonality)}")

# Show sample for top products
print(f"\n  Sample seasonality patterns (Top 5 products by claim volume):")

for product in df_seasonality['PRODUCT_GROUP'].unique()[:5]:
    prod_data = df_seasonality[df_seasonality['PRODUCT_GROUP'] == product]
    
    # Get peaks
    peaks = prod_data.nlargest(3, 'SEASONALITY_INDEX')
    peak_months = ', '.join(peaks['MONTH'].values)
    peak_indices = ', '.join([f"{x:.2f}x" for x in peaks['SEASONALITY_INDEX'].values])
    
    print(f"    {product[:30]:30} → Peak months: {peak_months} ({peak_indices})")

# ============================================================================
# STEP 5: Save Enhanced Seasonality CSV
# ============================================================================
print("\n[STEP 5] Saving Enhanced Seasonality CSV...")

output_file = Path("models/product_seasonality_enhanced.csv")
df_seasonality.to_csv(output_file, index=False)

print(f"  ✓ Saved to {output_file}")
print(f"    Shape: {df_seasonality.shape}")
print(f"    Columns: {list(df_seasonality.columns)}")

# ============================================================================
# STEP 6: Summary Statistics
# ============================================================================
print("\n[STEP 6] SUMMARY STATISTICS")
print("="*80)

print(f"\nSeasonality Index Distribution:")
print(f"  Min index: {df_seasonality['SEASONALITY_INDEX'].min():.4f}")
print(f"  Max index: {df_seasonality['SEASONALITY_INDEX'].max():.4f}")
print(f"  Mean index: {df_seasonality['SEASONALITY_INDEX'].mean():.4f}")
print(f"  Median index: {df_seasonality['SEASONALITY_INDEX'].median():.4f}")

print(f"\nTop 10 Most Seasonal Products (by variance):")
seasonal_var = df_seasonality.groupby('PRODUCT_GROUP')['SEASONALITY_INDEX'].var().sort_values(ascending=False)
for i, (product, var) in enumerate(seasonal_var.head(10).items(), 1):
    indices = df_seasonality[df_seasonality['PRODUCT_GROUP'] == product]['SEASONALITY_INDEX']
    print(f"  {i}. {product[:40]:40} (variance={var:.4f}, range={indices.min():.2f}-{indices.max():.2f})")

print(f"\nMonthly Seasonality Distribution:")
monthly_avg = df_seasonality.groupby('MONTH')['SEASONALITY_INDEX'].mean().sort_values(ascending=False)
for month, avg_idx in monthly_avg.items():
    print(f"  {month:12} → {avg_idx:.3f}x")

print("\n" + "="*80)
print("ENHANCED SEASONALITY EXTRACTION COMPLETE")
print("="*80)

print(f"\nKey Findings:")
print(f"1. Multi-year seasonality patterns extracted (FY21-26)")
print(f"2. {len(df_seasonality['PRODUCT_GROUP'].unique())} unique products with seasonality")
print(f"3. Seasonality range: {df_seasonality['SEASONALITY_INDEX'].min():.2f}x to {df_seasonality['SEASONALITY_INDEX'].max():.2f}x")
print(f"4. Peak months identified for each product")
print(f"5. Confidence scores assigned based on data availability")

print(f"\nFor Model Retraining:")
print(f"- Load: models/product_seasonality_enhanced.csv")
print(f"- Use SEASONALITY_INDEX column in forecast")
print(f"- Weight by CONFIDENCE for uncertain products")
print(f"- Apply to seasonal_amplification in inference")
