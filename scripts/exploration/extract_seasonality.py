"""
Extract Seasonality Patterns from SIP and TACTIC data
Merge with actual sales data to create comprehensive features
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")

print("="*80)
print("EXTRACTING SEASONALITY & PROMOTIONAL FEATURES")
print("="*80)

# ============================================================================
# 1. Load SIP PLAN Data (Has Product Groups and Months)
# ============================================================================
print("\n[Step 1] Loading SIP Plan Data...")

sip_plan_files = ['SIP_PLAN_DIMNS.csv', 'SIP_PLAN_DIMNS_21_22.csv']
all_sip_plans = []

for file in sip_plan_files:
    file_path = DATA_DIR / file
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.strip()
            
            # Extract relevant columns
            if 'PRODUCT_GROUP' in df.columns and 'MONTH' in df.columns:
                # Try to find budget/amount columns
                budget_cols = [c for c in df.columns if 'BUDGET' in c.upper() or 'AMOUNT' in c.upper()]
                
                subset = df[['PRODUCT_GROUP', 'MONTH', 'TACTIC_TYPE']].copy()
                
                if budget_cols:
                    for bc in budget_cols[:2]:  # Take first 2 budget columns
                        subset[bc] = pd.to_numeric(df[bc], errors='coerce')
                
                all_sip_plans.append(subset)
                print(f"  ✓ Loaded {file}: {len(df):,} records")
        except Exception as e:
            print(f"  ✗ Error loading {file}: {e}")

if all_sip_plans:
    df_sip = pd.concat(all_sip_plans, ignore_index=True)
    print(f"\n  Combined SIP Plans: {len(df_sip):,} records")
    
    # Map month names to numbers
    month_map = {
        'January': 1, 'February': 2, 'March': 3, 'April': 4,
        'May': 5, 'June': 6, 'July': 7, 'August': 8,
        'September': 9, 'October': 10, 'November': 11, 'December': 12
    }
    df_sip['MONTH_NUM'] = df_sip['MONTH'].map(month_map)
    
    # Count promotional activities by product and month
    promo_activity = df_sip.groupby(['PRODUCT_GROUP', 'MONTH_NUM']).size().reset_index(name='PROMO_COUNT')
    
    print(f"\n  Seasonality patterns extracted for {promo_activity['PRODUCT_GROUP'].nunique()} products")
    
    # Calculate seasonality index for each product
    seasonality_index = {}
    
    for product in promo_activity['PRODUCT_GROUP'].unique():
        prod_data = promo_activity[promo_activity['PRODUCT_GROUP'] == product]
        
        # Create a complete 12-month view
        all_months = pd.DataFrame({'MONTH_NUM': range(1, 13)})
        prod_complete = all_months.merge(prod_data, on='MONTH_NUM', how='left')
        prod_complete['PROMO_COUNT'] = prod_complete['PROMO_COUNT'].fillna(0)
        
        # Calculate index (relative to average)
        avg_count = prod_complete['PROMO_COUNT'].mean()
        if avg_count > 0:
            prod_complete['SEASONALITY_INDEX'] = prod_complete['PROMO_COUNT'] / avg_count
        else:
            prod_complete['SEASONALITY_INDEX'] = 1.0
        
        seasonality_index[product] = prod_complete.set_index('MONTH_NUM')['SEASONALITY_INDEX'].to_dict()
    
    print(f"\n  📊 Top 10 Products - Seasonality Patterns:")
    print(f"  (Index > 1.0 = Higher promotional activity)")
    print()
    
    top_products = promo_activity.groupby('PRODUCT_GROUP')['PROMO_COUNT'].sum().nlargest(10).index
    
    for product in top_products:
        print(f"  🔹 {product}:")
        indices = seasonality_index[product]
        
        # Show visual bar chart
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        
        for m in range(1, 13):
            idx = indices.get(m, 1.0)
            bar = '█' * int(idx * 10)
            marker = ' ⭐' if idx > 1.5 else ''
            print(f"     {month_names[m-1]}: {idx:>5.2f} {bar}{marker}")
        print()

# ============================================================================
# 2. Save Seasonality Index for use in training
# ============================================================================
if all_sip_plans:
    print("\n[Step 2] Saving seasonality features...")
    
    # Create a flat dataframe for easy lookup
    seasonality_rows = []
    for product, month_indices in seasonality_index.items():
        for month, index in month_indices.items():
            seasonality_rows.append({
                'PRODUCT_GROUP': product,
                'MONTH': month,
                'SEASONALITY_INDEX': index
            })
    
    df_seasonality = pd.DataFrame(seasonality_rows)
    
    # Save to models folder
    output_path = Path("models") / "product_seasonality.csv"
    output_path.parent.mkdir(exist_ok=True)
    df_seasonality.to_csv(output_path, index=False)
    
    print(f"  ✓ Saved seasonality index: {output_path}")
    print(f"  Products: {df_seasonality['PRODUCT_GROUP'].nunique()}")
    print(f"  Records: {len(df_seasonality)}")

# ============================================================================
# 3. Analyze from actual sales data
# ============================================================================
print("\n[Step 3] Analyzing actual sales seasonality...")

# Parse dates function
def parse_mixed_dates(date_series):
    parsed_dates = []
    for date_val in date_series:
        if pd.isna(date_val):
            parsed_dates.append(pd.NaT)
            continue
        date_str = str(date_val).strip()
        parsed = None
        formats_to_try = ['%d-%m-%y', '%d-%m-%Y', '%d-%b-%y', '%d-%b-%Y', '%Y-%m-%d', '%m/%d/%Y', '%d/%m/%Y']
        for fmt in formats_to_try:
            try:
                parsed = pd.to_datetime(date_str, format=fmt)
                break
            except:
                continue
        if parsed is None:
            try:
                parsed = pd.to_datetime(date_str, dayfirst=True)
            except:
                parsed = pd.NaT
        parsed_dates.append(parsed)
    return pd.Series(parsed_dates)

# Load a sample of sales data
try:
    sample_files = ['Jan 2024.csv', 'Jun 2024.csv', 'Dec 2023.csv']
    all_sales = []
    
    for file in sample_files:
        file_path = DATA_DIR / 'contracts_claims_extracted' / file
        if file_path.exists():
            df = pd.read_csv(file_path, low_memory=False)
            df.columns = df.columns.str.strip()
            col_map = {col.upper(): col for col in df.columns}
            
            if all(['CLAIM_DATE' in col_map, 'AMOUNT' in col_map, 'PRODUCT_GROUP' in col_map]):
                df_clean = pd.DataFrame()
                df_clean['DATE'] = parse_mixed_dates(df[col_map['CLAIM_DATE']])
                df_clean['SALES'] = pd.to_numeric(df[col_map['AMOUNT']], errors='coerce')
                df_clean['PRODUCT'] = df[col_map['PRODUCT_GROUP']].astype(str).str.strip()
                all_sales.append(df_clean)
    
    if all_sales:
        df_sales = pd.concat(all_sales, ignore_index=True)
        df_sales = df_sales.dropna(subset=['DATE', 'SALES'])
        df_sales = df_sales[df_sales['SALES'] > 0]
        df_sales['MONTH'] = df_sales['DATE'].dt.month
        
        # Calculate sales seasonality
        sales_by_prod_month = df_sales.groupby(['PRODUCT', 'MONTH'])['SALES'].agg(['sum', 'mean', 'count']).reset_index()
        
        print(f"  Analyzed {len(df_sales):,} sales records")
        print(f"\n  📈 Top 5 Products - Actual Sales Seasonality:")
        
        top_sales_products = df_sales.groupby('PRODUCT')['SALES'].sum().nlargest(5).index
        
        for product in top_sales_products:
            prod_sales = sales_by_prod_month[sales_by_prod_month['PRODUCT'] == product]
            
            # Create complete 12-month view
            all_months = pd.DataFrame({'MONTH': range(1, 13)})
            prod_complete = all_months.merge(prod_sales, on='MONTH', how='left')
            prod_complete['sum'] = prod_complete['sum'].fillna(0)
            
            # Calculate index
            avg_sales = prod_complete['sum'].mean()
            if avg_sales > 0:
                prod_complete['INDEX'] = prod_complete['sum'] / avg_sales
            else:
                prod_complete['INDEX'] = 1.0
            
            print(f"\n  🔹 {product}:")
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            
            for _, row in prod_complete.iterrows():
                m = int(row['MONTH'])
                idx = row['INDEX']
                bar = '█' * int(idx * 10)
                marker = ' 🔥' if idx > 1.3 else ''
                print(f"     {month_names[m-1]}: {idx:>5.2f} {bar}{marker}")

except Exception as e:
    print(f"  Error analyzing sales data: {e}")

print("\n" + "="*80)
print("SEASONALITY EXTRACTION COMPLETE")
print("="*80)
print("\n💡 Next Steps:")
print("  1. Seasonality index saved to models/product_seasonality.csv")
print("  2. This can be merged into training data as additional features")
print("  3. Re-run train_task2_reboot.py to incorporate seasonality")
