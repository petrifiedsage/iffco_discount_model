"""
Data Exploration Script for Task 2 Sales Forecasting
Run this before training to understand the data characteristics
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = Path("data/contracts_claims_extracted")

print("="*80)
print("DATA EXPLORATION FOR SALES FORECASTING")
print("="*80)

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

# Load all files
all_data = []
csv_files = list(DATA_DIR.glob("*.csv"))

print(f"\n1. FILES FOUND: {len(csv_files)} CSV files")

file_info = []
for csv_file in csv_files:
    try:
        df = pd.read_csv(csv_file, low_memory=False)
        df.columns = df.columns.str.strip()
        
        # Basic info
        info = {
            'file': csv_file.name,
            'rows': len(df),
            'has_claim_date': 'CLAIM_DATE' in [c.upper() for c in df.columns],
            'has_amount': 'AMOUNT' in [c.upper() for c in df.columns],
            'has_customer': 'CUSTOMER_NAME' in [c.upper() for c in df.columns],
            'has_product': 'PRODUCT_GROUP' in [c.upper() for c in df.columns],
            'has_geo': 'GEOGRAPHICAL_CLUSTER' in [c.upper() for c in df.columns],
        }
        
        # Date range
        if info['has_claim_date']:
            col_map = {col.upper(): col for col in df.columns}
            dates = parse_mixed_dates(df[col_map['CLAIM_DATE']])
            info['min_date'] = dates.min()
            info['max_date'] = dates.max()
            info['year'] = dates.dt.year.mode()[0] if len(dates.dropna()) > 0 else None
        
        file_info.append(info)
        
        # Load for combined analysis
        if all(info.values()):
            col_map = {col.upper(): col for col in df.columns}
            df_clean = pd.DataFrame()
            df_clean['DATE'] = df[col_map['CLAIM_DATE']]
            df_clean['SALES'] = pd.to_numeric(df[col_map['AMOUNT']], errors='coerce')
            df_clean['CUSTOMER'] = df[col_map['CUSTOMER_NAME']].astype(str).str.strip()
            df_clean['PRODUCT'] = df[col_map['PRODUCT_GROUP']].astype(str).str.strip()
            df_clean['REGION'] = df[col_map['GEOGRAPHICAL_CLUSTER']].astype(str).str.strip()
            all_data.append(df_clean)
            
    except Exception as e:
        print(f"  Error reading {csv_file.name}: {e}")

# Display file info
print("\n2. FILE DETAILS:")
df_files = pd.DataFrame(file_info)
df_files = df_files.sort_values('year')
print(df_files[['file', 'rows', 'year', 'min_date', 'max_date']].to_string(index=False))

# Combine and analyze
if all_data:
    df_combined = pd.concat(all_data, ignore_index=True)
    print(f"\n3. COMBINED RAW DATA: {len(df_combined):,} rows")
    
    # Parse dates
    df_combined['DATE'] = parse_mixed_dates(df_combined['DATE'])
    
    # Clean
    df_combined = df_combined.dropna(subset=['DATE', 'SALES'])
    df_combined = df_combined[df_combined['SALES'] > 0]
    df_combined = df_combined[df_combined['CUSTOMER'] != 'nan']
    df_combined = df_combined[df_combined['PRODUCT'] != 'nan']
    
    # UAE Filter
    uae_mask = df_combined['REGION'].str.contains('ROUAE|United Arab|UAE', case=False, na=False)
    df_uae = df_combined[uae_mask]
    
    print(f"\n4. AFTER CLEANING & UAE FILTER: {len(df_uae):,} rows")
    print(f"   Date range: {df_uae['DATE'].min()} to {df_uae['DATE'].max()}")
    print(f"   Unique customers: {df_uae['CUSTOMER'].nunique()}")
    print(f"   Unique products: {df_uae['PRODUCT'].nunique()}")
    
    # Year distribution
    df_uae['YEAR'] = df_uae['DATE'].dt.year
    print(f"\n5. YEAR DISTRIBUTION:")
    print(df_uae['YEAR'].value_counts().sort_index())
    
    # Monthly distribution
    df_uae['MONTH_START'] = df_uae['DATE'].dt.to_period('M').dt.to_timestamp()
    monthly = df_uae.groupby('MONTH_START').agg({
        'SALES': ['sum', 'count'],
        'CUSTOMER': 'nunique',
        'PRODUCT': 'nunique'
    }).reset_index()
    monthly.columns = ['Month', 'Total_Sales', 'Transactions', 'Customers', 'Products']
    
    print(f"\n6. MONTHLY AGGREGATION: {len(monthly)} months")
    print(monthly.tail(12).to_string(index=False))
    
    # Sales statistics
    print(f"\n7. SALES STATISTICS:")
    print(df_uae['SALES'].describe())
    
    # Top customers
    print(f"\n8. TOP 10 CUSTOMERS (by total sales):")
    top_cust = df_uae.groupby('CUSTOMER')['SALES'].sum().sort_values(ascending=False).head(10)
    for i, (cust, sales) in enumerate(top_cust.items(), 1):
        print(f"   {i}. {cust[:50]:<50} ${sales:>15,.0f}")
    
    # Product distribution
    print(f"\n9. PRODUCT DISTRIBUTION:")
    prod_dist = df_uae.groupby('PRODUCT')['SALES'].agg(['sum', 'count']).sort_values('sum', ascending=False)
    prod_dist.columns = ['Total_Sales', 'Transactions']
    print(prod_dist.head(15).to_string())
    
    # 2025 data check
    df_2025 = df_uae[df_uae['YEAR'] == 2025]
    df_pre2025 = df_uae[df_uae['YEAR'] < 2025]
    
    print(f"\n10. TRAIN/TEST SPLIT PREVIEW:")
    print(f"    Training data (2022-2024): {len(df_pre2025):,} rows")
    print(f"      - Date range: {df_pre2025['DATE'].min()} to {df_pre2025['DATE'].max()}")
    print(f"      - Customers: {df_pre2025['CUSTOMER'].nunique()}")
    print(f"      - Products: {df_pre2025['PRODUCT'].nunique()}")
    print(f"    Test data (2025): {len(df_2025):,} rows")
    print(f"      - Date range: {df_2025['DATE'].min()} to {df_2025['DATE'].max()}")
    print(f"      - Customers: {df_2025['CUSTOMER'].nunique()}")
    print(f"      - Products: {df_2025['PRODUCT'].nunique()}")
    
    # Customer-Product combinations
    cp_2024 = set(zip(df_pre2025['CUSTOMER'], df_pre2025['PRODUCT']))
    cp_2025 = set(zip(df_2025['CUSTOMER'], df_2025['PRODUCT']))
    overlap = len(cp_2024 & cp_2025)
    new_in_2025 = len(cp_2025 - cp_2024)
    
    print(f"\n11. CUSTOMER-PRODUCT COMBINATIONS:")
    print(f"    Training (2022-2024): {len(cp_2024):,} unique pairs")
    print(f"    Test (2025): {len(cp_2025):,} unique pairs")
    print(f"    Overlap: {overlap:,} pairs ({overlap/len(cp_2025)*100:.1f}% of 2025)")
    print(f"    New in 2025: {new_in_2025:,} pairs")
    
    print("\n" + "="*80)
    print("EXPLORATION COMPLETE")
    print("="*80)
