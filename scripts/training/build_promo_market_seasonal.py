"""
Aggregate TACTIC promotional data to Market × Product × Season level.
Output: results/promo_market_seasonal.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re

DATA_DIR = Path("data/csv")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Business seasons
SEASON_MAP = {
    1: 'Winter', 2: 'Winter',
    3: 'Spring', 4: 'Spring',
    5: 'Summer', 6: 'Summer', 7: 'Summer', 8: 'Summer',
    9: 'Fall', 10: 'Fall',
    11: 'Festival', 12: 'Festival'
}

print("Loading TACTIC promotional data...")

# Load TACTIC Plan files
plan_files = sorted(DATA_DIR.glob("TACTIC_*_TACTIC_PLAN_DIMNS.csv"))
claims_files = sorted(DATA_DIR.glob("TACTIC_*_TACTIC_CM_CLAIMS_DIMNS.csv"))

all_promo = []

for plan_file in plan_files:
    try:
        df = pd.read_csv(plan_file, low_memory=False)
        
        if 'PLAN_STATUS' not in df.columns:
            print(f"[SKIP] {plan_file.name}: Missing required columns")
            continue
        
        # Only approved tactics
        df = df[df['PLAN_STATUS'] == 'Approved'].copy()
        
        if df.empty:
            continue
        
        # Extract product from PRODUCT_GROUP column directly
        if 'PRODUCT_GROUP' in df.columns:
            df['PRODUCT_GROUP'] = df['PRODUCT_GROUP'].astype(str).str.strip()
        else:
            continue
        
        # Extract market from various columns
        if 'GEO_CLUST' in df.columns:
            df['MARKET'] = df['GEO_CLUST'].astype(str).str.strip()
        elif 'TACTIC_LOCATION' in df.columns:
            df['MARKET'] = df['TACTIC_LOCATION'].astype(str).str.strip()
        elif 'COUNTRY' in df.columns:
            df['MARKET'] = df['COUNTRY'].astype(str).str.strip()
        else:
            df['MARKET'] = 'UAE'
        
        df['MARKET'] = df['MARKET'].replace({'nan': 'UAE', 'NaN': 'UAE', '': 'UAE'})
        
        # Parse dates and extract period
        if 'PLAN_CREATION_DATE' in df.columns:
            df['START_DATE'] = pd.to_datetime(df['PLAN_CREATION_DATE'], errors='coerce', format='%d-%b-%y')
        elif 'PLAN_LAST_UPDATED_DATE' in df.columns:
            df['START_DATE'] = pd.to_datetime(df['PLAN_LAST_UPDATED_DATE'], errors='coerce', format='%d-%b-%y')
        else:
            continue
        
        df = df.dropna(subset=['START_DATE'])
        df['YEAR'] = df['START_DATE'].dt.year
        df['MONTH'] = df['START_DATE'].dt.month
        df['SEASON'] = df['MONTH'].map(SEASON_MAP)
        
        # Promo amount - use a default value per tactic as TACTIC files don't have cost
        # We'll count number of tactics as intensity metric
        df['PROMO_AMOUNT'] = 1000.0  # Default cost per tactic for weighting
        df['PROMO_INTENSITY'] = 1  # Count of tactics
        
        # Clean
        df = df.dropna(subset=['PRODUCT_GROUP', 'MARKET', 'SEASON', 'YEAR'])
        df = df[df['PROMO_AMOUNT'] > 0]
        
        all_promo.append(df[['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'PROMO_AMOUNT']])
        print(f"[OK] {plan_file.name}: {len(df)} approved promo records")
        
    except Exception as e:
        print(f"[ERR] {plan_file.name}: {e}")

if not all_promo:
    print("No TACTIC data found, creating empty promo file")
    # Create empty structure
    empty_df = pd.DataFrame(columns=['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR', 'PROMO_AMOUNT', 'PROMO_COUNT'])
    empty_df.to_csv(OUTPUT_DIR / "promo_market_seasonal.csv", index=False)
else:
    df_all = pd.concat(all_promo, ignore_index=True)
    
    # Aggregate to market-product-season-year
    promo_agg = df_all.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR']).agg({
        'PROMO_AMOUNT': 'sum'
    }).reset_index()
    
    promo_agg['PROMO_COUNT'] = df_all.groupby(['MARKET', 'PRODUCT_GROUP', 'SEASON', 'YEAR']).size().values
    promo_agg['PROMO_LOG'] = np.log1p(promo_agg['PROMO_AMOUNT'])
    promo_agg['PROMO_COUNT_LOG'] = np.log1p(promo_agg['PROMO_COUNT'])
    
    out_path = OUTPUT_DIR / "promo_market_seasonal.csv"
    promo_agg.to_csv(out_path, index=False)
    
    print(f"\nSaved: {out_path}")
    print(f"Total rows: {len(promo_agg)}")
    print(f"Date range: {promo_agg['YEAR'].min()}-{promo_agg['YEAR'].max()}")
    print(f"\nSample:")
    print(promo_agg.head(10))
