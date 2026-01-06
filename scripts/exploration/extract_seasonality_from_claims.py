"""
Extract Enhanced Seasonality Patterns from SIP/TACTIC Claims Data
Uses ACTUAL claims instead of planned activities for more accurate seasonality
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")
OUTPUT_DIR = Path("models")

print("="*80)
print("ENHANCED SEASONALITY EXTRACTION - CLAIMS-BASED")
print("="*80)

# ============================================================================
# STEP 1: Load TACTIC Claims Data
# ============================================================================
print("\n[Step 1] Loading TACTIC Claims Data...")

try:
    df_tactic = pd.read_csv(DATA_DIR / 'TACTIC_CM_CLAIMS_DIMNS.csv', low_memory=False)
    df_tactic.columns = df_tactic.columns.str.strip()
    
    print(f"  ✓ TACTIC Claims loaded: {len(df_tactic):,} records")
    print(f"  Columns: {list(df_tactic.columns[:10])}")
    
    # Extract relevant columns
    if 'CLAIM_AMOUNT' in df_tactic.columns and 'MONTH' in df_tactic.columns:
        df_tactic['CLAIM_AMOUNT'] = pd.to_numeric(df_tactic['CLAIM_AMOUNT'], errors='coerce')
        df_tactic = df_tactic[df_tactic['CLAIM_AMOUNT'] > 0].copy()
        
        print(f"  Valid claims: {len(df_tactic):,}")
        print(f"  Total claim value: ${df_tactic['CLAIM_AMOUNT'].sum():,.0f}")
        
except Exception as e:
    print(f"  ✗ Error loading TACTIC: {e}")
    df_tactic = pd.DataFrame()


# ============================================================================
# STEP 2: Load SIP Claims Data (if available)
# ============================================================================
print("\n[Step 2] Loading SIP Claims Data...")

try:
    # SIP_CLAIM_DIMNS is large, so read with chunks or specific columns
    df_sip = pd.read_csv(DATA_DIR / 'SIP_CLAIM_DIMNS.csv', low_memory=False)
    df_sip.columns = df_sip.columns.str.strip()
    
    print(f"  ✓ SIP Claims loaded: {len(df_sip):,} records")
    
    # Look for amount and month columns
    amount_cols = [c for c in df_sip.columns if 'AMOUNT' in c.upper() or 'VALUE' in c.upper()]
    month_cols = [c for c in df_sip.columns if 'MONTH' in c.upper()]
    
    print(f"  Amount columns found: {amount_cols[:3]}")
    print(f"  Month columns found: {month_cols[:3]}")
    
except Exception as e:
    print(f"  ⚠ SIP Claims too large or error: {e}")
    print(f"  Proceeding with TACTIC only...")
    df_sip = pd.DataFrame()


# ============================================================================
# STEP 3: Extract Product Names from Claims
# ============================================================================
print("\n[Step 3] Mapping Claims to Product Categories...")

# Define product keywords to map TACTIC_NAME to our product categories
product_keywords = {
    'Chocolate': ['CHOCOLATE', 'CHOC', 'TIFFANY', 'QUANTA', 'PICCADELI'],
    'Biscuits': ['BISCUIT', 'COOKIE', 'WAFER'],
    'Ice creams': ['ICE', 'CREAM', 'IGLOO', 'LONDON DAIRY'],
    'Snacks': ['SNACK', 'CHIPS'],
    'Flour': ['FLOUR', 'ATTA'],
    'Oils': ['OIL', 'GHEE'],
    'Eggs': ['EGG'],
    'Butter & Margarine': ['BUTTER', 'MARGARINE'],
    'Cheese': ['CHEESE'],
    'Milk': ['MILK', 'DAIRY'],
    'Fruit & Veg Conc & Preparation': ['FRUIT', 'VEG', 'JUICE'],
    'Nuts': ['NUT', 'ALMOND', 'CASHEW'],
    'Condiments - Wet': ['KETCHUP', 'SAUCE', 'MAYO'],
    'Sauces': ['SAUCE'],
    'Pasta': ['PASTA', 'MACARONI']
}

def map_to_product_group(tactic_name):
    """Map TACTIC_NAME to product group using keywords"""
    if pd.isna(tactic_name):
        return 'Other'
    
    tactic_upper = str(tactic_name).upper()
    
    for product, keywords in product_keywords.items():
        if any(kw in tactic_upper for kw in keywords):
            return product
    
    return 'Other'


if not df_tactic.empty and 'TACTIC_NAME' in df_tactic.columns:
    df_tactic['PRODUCT_GROUP'] = df_tactic['TACTIC_NAME'].apply(map_to_product_group)
    
    # Show mapping results
    product_counts = df_tactic['PRODUCT_GROUP'].value_counts()
    print(f"\n  Product mapping results:")
    for prod, count in product_counts.head(15).items():
        pct = count / len(df_tactic) * 100
        print(f"    {prod}: {count:,} claims ({pct:.1f}%)")


# ============================================================================
# STEP 4: Map Month Names to Numbers
# ============================================================================
print("\n[Step 4] Processing Month Information...")

month_map = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4,
    'May': 5, 'June': 6, 'July': 7, 'August': 8,
    'September': 9, 'October': 10, 'November': 11, 'December': 12,
    'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'Jun': 6,
    'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
}

if not df_tactic.empty and 'MONTH' in df_tactic.columns:
    df_tactic['MONTH_NUM'] = df_tactic['MONTH'].map(month_map)
    
    # Remove records without valid month
    df_tactic = df_tactic[df_tactic['MONTH_NUM'].notna()].copy()
    df_tactic['MONTH_NUM'] = df_tactic['MONTH_NUM'].astype(int)
    
    print(f"  ✓ Months mapped: {len(df_tactic):,} records with valid months")


# ============================================================================
# STEP 5: Calculate Seasonality Index by Product & Month
# ============================================================================
print("\n[Step 5] Calculating Seasonality Indices...")

if not df_tactic.empty:
    # Aggregate claims by product and month (sum claim amounts)
    seasonality_raw = df_tactic.groupby(['PRODUCT_GROUP', 'MONTH_NUM']).agg({
        'CLAIM_AMOUNT': ['sum', 'count']
    }).reset_index()
    
    seasonality_raw.columns = ['PRODUCT_GROUP', 'MONTH_NUM', 'TOTAL_CLAIMS', 'CLAIM_COUNT']
    
    print(f"\n  Aggregated to {len(seasonality_raw)} product-month combinations")
    
    # Calculate seasonality index for each product
    seasonality_results = []
    
    for product in seasonality_raw['PRODUCT_GROUP'].unique():
        if product == 'Other':  # Skip 'Other' category
            continue
            
        prod_data = seasonality_raw[seasonality_raw['PRODUCT_GROUP'] == product].copy()
        
        # Ensure all 12 months present
        all_months = pd.DataFrame({'MONTH_NUM': range(1, 13)})
        prod_complete = all_months.merge(prod_data, on='MONTH_NUM', how='left')
        prod_complete['PRODUCT_GROUP'] = product
        prod_complete['TOTAL_CLAIMS'] = prod_complete['TOTAL_CLAIMS'].fillna(0)
        prod_complete['CLAIM_COUNT'] = prod_complete['CLAIM_COUNT'].fillna(0)
        
        # Calculate index (relative to average month)
        avg_claims = prod_complete['TOTAL_CLAIMS'].mean()
        
        if avg_claims > 0:
            prod_complete['SEASONALITY_INDEX'] = prod_complete['TOTAL_CLAIMS'] / avg_claims
        else:
            prod_complete['SEASONALITY_INDEX'] = 1.0
        
        # Smooth extreme values (clip between 0.2 and 2.5)
        prod_complete['SEASONALITY_INDEX'] = prod_complete['SEASONALITY_INDEX'].clip(0.2, 2.5)
        
        # Store results
        for _, row in prod_complete.iterrows():
            seasonality_results.append({
                'PRODUCT_GROUP': product,
                'MONTH': int(row['MONTH_NUM']),
                'SEASONALITY_INDEX': round(row['SEASONALITY_INDEX'], 6),
                'TOTAL_CLAIMS_USD': round(row['TOTAL_CLAIMS'], 2),
                'CLAIM_COUNT': int(row['CLAIM_COUNT'])
            })
    
    # Create final dataframe
    df_seasonality = pd.DataFrame(seasonality_results)
    
    print(f"\n  ✓ Seasonality calculated for {df_seasonality['PRODUCT_GROUP'].nunique()} products")
    print(f"  ✓ Total patterns: {len(df_seasonality)}")
    
    # Show sample patterns
    print("\n  📊 Sample Seasonality Patterns:")
    for product in ['Chocolate', 'Ice creams', 'Biscuits', 'Snacks'][:3]:
        if product in df_seasonality['PRODUCT_GROUP'].values:
            prod_season = df_seasonality[df_seasonality['PRODUCT_GROUP'] == product].copy()
            prod_season = prod_season.sort_values('MONTH')
            
            print(f"\n    {product}:")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for _, row in prod_season.iterrows():
                month_name = month_names[int(row['MONTH'])-1]
                index = row['SEASONALITY_INDEX']
                claims = row['TOTAL_CLAIMS_USD']
                bar = '█' * int(index * 10)
                print(f"      {month_name}: {index:.2f} {bar} (${claims:,.0f})")
    
    # ============================================================================
    # STEP 6: Save Results
    # ============================================================================
    print(f"\n[Step 6] Saving seasonality index...")
    
    output_path = OUTPUT_DIR / 'product_seasonality_claims.csv'
    df_seasonality.to_csv(output_path, index=False)
    
    print(f"  ✓ Saved to: {output_path}")
    print(f"\n  Summary:")
    print(f"    Products: {df_seasonality['PRODUCT_GROUP'].nunique()}")
    print(f"    Patterns: {len(df_seasonality)}")
    print(f"    Index range: {df_seasonality['SEASONALITY_INDEX'].min():.2f} - {df_seasonality['SEASONALITY_INDEX'].max():.2f}")
    
    print("\n" + "="*80)
    print("SUCCESS! Claims-based seasonality patterns extracted.")
    print("="*80)
    
else:
    print("\n❌ No valid data to process")
