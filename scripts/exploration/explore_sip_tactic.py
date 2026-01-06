"""
Explore SIP and TACTIC datasets for seasonality and promotional patterns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data")

print("="*80)
print("SIP & TACTIC DATA EXPLORATION - SEASONALITY ANALYSIS")
print("="*80)

# List of files to explore
sip_files = [
    'SIP_CLAIM_DIMNS.csv',
    'SIP_PLAN_DIMNS.csv',
    'SIP_SKU_DIMNS.csv',
    'SIP_CLAIM_DIMNS_21_22.csv',
    'SIP_PLAN_DIMNS_21_22.csv',
    'SIP_SKU_DIMNS_21_22.csv'
]

tactic_files = [
    'TACTIC_CM_CLAIMS_DIMNS.csv',
    'TACTIC_PLAN_DIMNS.csv',
    'TACTIC_SKU_DIMNS.csv'
]

print("\n" + "="*80)
print("1. SIP DATASETS (Sales Incentive Programs)")
print("="*80)

for file in sip_files:
    file_path = DATA_DIR / file
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"\n📁 {file}")
            print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"   Columns: {list(df.columns)[:15]}")
            
            # Check for date columns
            date_cols = [col for col in df.columns if any(x in col.upper() for x in ['DATE', 'MONTH', 'PERIOD', 'TIME'])]
            if date_cols:
                print(f"   Date columns: {date_cols}")
                for dcol in date_cols[:3]:
                    print(f"     - {dcol}: {df[dcol].head(5).tolist()}")
            
            # Check for product columns
            prod_cols = [col for col in df.columns if any(x in col.upper() for x in ['PRODUCT', 'SKU', 'ITEM'])]
            if prod_cols:
                print(f"   Product columns: {prod_cols[:3]}")
                if len(prod_cols) > 0:
                    print(f"     - Unique values: {df[prod_cols[0]].nunique()}")
            
            # Check for amount/sales columns
            amt_cols = [col for col in df.columns if any(x in col.upper() for x in ['AMOUNT', 'SALES', 'VALUE', 'REVENUE'])]
            if amt_cols:
                print(f"   Amount columns: {amt_cols[:3]}")
            
            # Sample data
            print(f"\n   📊 Sample Data:")
            print(df.head(3).to_string(index=False, max_cols=10))
            
        except Exception as e:
            print(f"\n❌ Error reading {file}: {e}")
    else:
        print(f"\n⚠️  {file} - Not found")

print("\n" + "="*80)
print("2. TACTIC DATASETS (Tactical Planning)")
print("="*80)

for file in tactic_files:
    file_path = DATA_DIR / file
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"\n📁 {file}")
            print(f"   Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
            print(f"   Columns: {list(df.columns)[:15]}")
            
            # Check for date columns
            date_cols = [col for col in df.columns if any(x in col.upper() for x in ['DATE', 'MONTH', 'PERIOD', 'TIME'])]
            if date_cols:
                print(f"   Date columns: {date_cols}")
                for dcol in date_cols[:3]:
                    print(f"     - {dcol}: {df[dcol].head(5).tolist()}")
            
            # Check for product columns
            prod_cols = [col for col in df.columns if any(x in col.upper() for x in ['PRODUCT', 'SKU', 'ITEM'])]
            if prod_cols:
                print(f"   Product columns: {prod_cols[:3]}")
            
            # Check for promo/seasonality
            promo_cols = [col for col in df.columns if any(x in col.upper() for x in ['PROMO', 'CAMPAIGN', 'SEASONAL', 'TACTIC'])]
            if promo_cols:
                print(f"   Promo/Seasonality columns: {promo_cols[:5]}")
            
            # Sample data
            print(f"\n   📊 Sample Data:")
            print(df.head(3).to_string(index=False, max_cols=10))
            
        except Exception as e:
            print(f"\n❌ Error reading {file}: {e}")
    else:
        print(f"\n⚠️  {file} - Not found")

print("\n" + "="*80)
print("3. DETAILED SEASONALITY ANALYSIS")
print("="*80)

# Focus on the most promising files
key_files = {
    'SIP_CLAIM_DIMNS.csv': 'SIP Claims',
    'SIP_PLAN_DIMNS.csv': 'SIP Plans',
    'TACTIC_CM_CLAIMS_DIMNS.csv': 'Tactic Claims',
    'TACTIC_PLAN_DIMNS.csv': 'Tactic Plans'
}

for file, label in key_files.items():
    file_path = DATA_DIR / file
    if file_path.exists():
        try:
            df = pd.read_csv(file_path, low_memory=False)
            print(f"\n{'='*70}")
            print(f"📈 {label} ({file})")
            print(f"{'='*70}")
            
            # Try to identify key columns
            df.columns = df.columns.str.strip()
            
            # Find product group column
            prod_col = None
            for col in df.columns:
                if 'PRODUCT_GROUP' in col.upper() or 'PRODUCT GROUP' in col.upper():
                    prod_col = col
                    break
            
            # Find date/month column
            date_col = None
            for col in df.columns:
                if any(x in col.upper() for x in ['CLAIM_DATE', 'PLAN_DATE', 'MONTH', 'PERIOD']):
                    date_col = col
                    break
            
            # Find amount column
            amt_col = None
            for col in df.columns:
                if any(x in col.upper() for x in ['AMOUNT', 'VALUE', 'SALES']):
                    amt_col = col
                    break
            
            print(f"   Key columns identified:")
            print(f"     - Product: {prod_col}")
            print(f"     - Date: {date_col}")
            print(f"     - Amount: {amt_col}")
            
            if prod_col and date_col and amt_col:
                # Parse dates
                df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                df['MONTH'] = df[date_col].dt.month
                df['YEAR'] = df[date_col].dt.year
                
                # Clean amount
                df[amt_col] = pd.to_numeric(df[amt_col], errors='coerce')
                
                # Drop nulls
                df_clean = df.dropna(subset=[prod_col, 'MONTH', amt_col])
                
                # Calculate monthly seasonality by product
                seasonality = df_clean.groupby([prod_col, 'MONTH'])[amt_col].agg(['sum', 'count', 'mean']).reset_index()
                seasonality.columns = ['PRODUCT', 'MONTH', 'Total', 'Count', 'Avg']
                
                print(f"\n   📊 Monthly Seasonality Pattern:")
                print(f"   Products analyzed: {seasonality['PRODUCT'].nunique()}")
                
                # Show top products
                top_products = df_clean.groupby(prod_col)[amt_col].sum().nlargest(5).index
                
                for product in top_products:
                    prod_season = seasonality[seasonality['PRODUCT'] == product].sort_values('MONTH')
                    if len(prod_season) > 0:
                        print(f"\n   🔹 {product}:")
                        # Calculate seasonality index (avg month / overall avg)
                        overall_avg = prod_season['Avg'].mean()
                        prod_season['Index'] = (prod_season['Avg'] / overall_avg * 100).round(1)
                        
                        # Show month-by-month
                        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                     'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
                        for _, row in prod_season.iterrows():
                            month_idx = int(row['MONTH']) - 1
                            month_name = month_names[month_idx]
                            bar = '█' * int(row['Index'] / 10)
                            print(f"      {month_name}: {row['Index']:>6.1f}% {bar}")
                
        except Exception as e:
            print(f"   ❌ Error analyzing {file}: {e}")

print("\n" + "="*80)
print("EXPLORATION COMPLETE")
print("="*80)
