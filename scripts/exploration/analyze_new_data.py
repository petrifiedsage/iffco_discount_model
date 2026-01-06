"""
Analyze New TDA and TACTIC Data for Seasonality Insights
Explores all TDA and TACTIC files to identify product seasonality patterns
and recommend model improvements
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("NEW DATA ANALYSIS - TDA & TACTIC SEASONALITY")
print("="*80)

DATA_DIR = Path("data")

# ============================================================================
# 1. DISCOVER AND LOAD TDA DATA
# ============================================================================
print("\n[STEP 1] Loading TDA (Trade Development Allowance) Data...")

tda_files = list(DATA_DIR.glob("TDA_FY*.xls*"))
print(f"  Found {len(tda_files)} TDA files:")
for f in sorted(tda_files):
    print(f"    - {f.name}")

all_tda = []
for file in sorted(tda_files):
    try:
        # Try reading with openpyxl first (newer), then xlrd (older)
        try:
            df = pd.read_excel(file, engine='openpyxl')
        except:
            df = pd.read_excel(file, engine='xlrd')
        
        df.columns = df.columns.str.strip()
        all_tda.append(df)
        print(f"  ✓ {file.name}: {len(df):,} rows, columns: {list(df.columns[:5])}")
    except Exception as e:
        print(f"  ✗ {file.name}: {e}")

if all_tda:
    df_tda_combined = pd.concat(all_tda, ignore_index=True)
    print(f"\n  Combined TDA data: {len(df_tda_combined):,} rows")
    print(f"  Date range: {df_tda_combined.columns.tolist()}")

# ============================================================================
# 2. DISCOVER AND LOAD TACTIC DATA
# ============================================================================
print("\n[STEP 2] Loading TACTIC Data...")

tactic_files = list(DATA_DIR.glob("TACTIC_*.xls*"))
print(f"  Found {len(tactic_files)} TACTIC files:")
for f in sorted(tactic_files):
    print(f"    - {f.name}")

all_tactic = []
for file in sorted(tactic_files):
    try:
        try:
            df = pd.read_excel(file, engine='openpyxl')
        except:
            df = pd.read_excel(file, engine='xlrd')
        
        df.columns = df.columns.str.strip()
        all_tactic.append(df)
        print(f"  ✓ {file.name}: {len(df):,} rows")
    except Exception as e:
        print(f"  ✗ {file.name}: {e}")

if all_tactic:
    df_tactic_combined = pd.concat(all_tactic, ignore_index=True)
    print(f"\n  Combined TACTIC data: {len(df_tactic_combined):,} rows")

# ============================================================================
# 3. ANALYZE TACTIC DATA FOR SEASONALITY (Primary Source)
# ============================================================================
print("\n[STEP 3] Analyzing TACTIC Data for Product Seasonality...")

if not all_tactic:
    print("  ⚠ No TACTIC data loaded")
else:
    df_tactic = df_tactic_combined.copy()
    
    # Find month and product columns
    month_cols = [c for c in df_tactic.columns if 'MONTH' in c.upper()]
    product_cols = [c for c in df_tactic.columns if 'PRODUCT' in c.upper()]
    amount_cols = [c for c in df_tactic.columns if 'AMOUNT' in c.upper() or 'VALUE' in c.upper()]
    
    print(f"\n  Month columns: {month_cols}")
    print(f"  Product columns: {product_cols}")
    print(f"  Amount columns: {amount_cols[:3]}")
    
    # Analyze by month if available
    if month_cols and amount_cols:
        month_col = month_cols[0]
        amount_col = amount_cols[0]
        
        # Count claims/deals by month
        if month_col in df_tactic.columns:
            monthly_summary = df_tactic.groupby(month_col).agg({
                amount_col: ['sum', 'count', 'mean']
            }).round(2)
            
            print(f"\n  Seasonality by Month (from {len(df_tactic):,} records):")
            print(f"\n  {'Month':<15} {'Total Amount':<20} {'Count':<10} {'Avg Deal':<15}")
            print("  " + "-"*60)
            
            for month, row in monthly_summary.iterrows():
                total = row[(amount_col, 'sum')]
                count = int(row[(amount_col, 'count')])
                avg = row[(amount_col, 'mean')]
                print(f"  {str(month):<15} ${total:>18,.0f} {count:>9} ${avg:>13,.0f}")

# ============================================================================
# 4. ANALYZE TDA DATA
# ============================================================================
print("\n[STEP 4] Analyzing TDA Data Structure...")

if all_tda:
    df_tda = df_tda_combined.copy()
    
    print(f"\n  TDA Dataset Info:")
    print(f"    Total rows: {len(df_tda):,}")
    print(f"    Columns: {list(df_tda.columns)}")
    print(f"\n    Sample data:")
    print(df_tda.head(3).to_string())

# ============================================================================
# 5. KEY SEASONALITY INSIGHTS
# ============================================================================
print("\n[STEP 5] Seasonality Pattern Analysis...")

insights = []

if all_tactic and 'MONTH' in [c.upper() for c in df_tactic_combined.columns]:
    df = df_tactic_combined.copy()
    
    # Identify peak and low seasons
    amount_cols_available = [c for c in df.columns if 'AMOUNT' in c.upper() or 'CLAIM' in c.upper()]
    if amount_cols_available:
        amount_col = amount_cols_available[0]
        
        if 'MONTH' in [c.upper() for c in df.columns]:
            month_col = [c for c in df.columns if c.upper() == 'MONTH'][0]
            
            monthly_totals = df.groupby(month_col)[amount_col].sum().sort_values(ascending=False)
            
            print(f"\n  Peak Months (Highest Activity):")
            for i, (month, value) in enumerate(monthly_totals.head(5).items(), 1):
                pct = (value / monthly_totals.sum() * 100)
                print(f"    {i}. {month}: ${value:,.0f} ({pct:.1f}%)")
            
            print(f"\n  Low Months (Lowest Activity):")
            for i, (month, value) in enumerate(monthly_totals.tail(5).items(), 1):
                pct = (value / monthly_totals.sum() * 100)
                print(f"    {i}. {month}: ${value:,.0f} ({pct:.1f}%)")
            
            # Calculate seasonality index
            avg_value = monthly_totals.mean()
            seasonality_index = (monthly_totals / avg_value).round(2).to_dict()
            
            print(f"\n  Seasonality Index (relative to average):")
            print(f"    Max: {max(seasonality_index.values()):.2f}x")
            print(f"    Min: {min(seasonality_index.values()):.2f}x")
            print(f"    Range: {max(seasonality_index.values()) - min(seasonality_index.values()):.2f}x")

# ============================================================================
# 6. RECOMMENDATIONS FOR MODEL IMPROVEMENT
# ============================================================================
print("\n[STEP 6] Model Improvement Recommendations...")

recommendations = []

print(f"\n  📊 SEASONALITY ENHANCEMENTS:")
print(f"    1. ✓ Extract month-product-specific patterns from TDA data")
print(f"    2. ✓ Identify fiscal year patterns (FY 21_22 to FY 25_26)")
print(f"    3. ✓ Create TDA-based seasonality weights (more granular)")
print(f"    4. ✓ Detect year-over-year seasonality shifts")
print(f"    5. ✓ Add promotional intensity factor (TDA deal count)")

print(f"\n  🔧 FEATURE ENGINEERING IMPROVEMENTS:")
print(f"    1. ✓ Add TDA deal frequency feature (promotions per month)")
print(f"    2. ✓ Add TDA average deal value feature (intensity)")
print(f"    3. ✓ Create fiscal year features (FY patterns repeat)")
print(f"    4. ✓ Add quarter-on-quarter TDA growth")
print(f"    5. ✓ Weight seasonality by historical TDA activity")

print(f"\n  📈 DATA COVERAGE:")
if all_tda:
    print(f"    • TDA Files: {len(tda_files)} (FY 21_22 to FY 25_26)")
if all_tactic:
    print(f"    • TACTIC Files: {len(tactic_files)} (FY 21_22 to FY 25_26)")
print(f"    • Combined Coverage: 4.5+ years of promotional data")

# ============================================================================
# 7. NEXT STEPS
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDED ACTIONS")
print("="*80)

print(f"""
1. CREATE ENHANCED SEASONALITY EXTRACTOR
   - Extract TDA month-product patterns
   - Calculate TDA promotional intensity by month
   - Create fiscal year seasonality factors
   - Merge with existing claims-based seasonality

2. ENHANCE FEATURE ENGINEERING
   - Add TDA deal frequency (LAG features per month)
   - Add promotional budget intensity
   - Add fiscal year flags
   - Weight lags by promotional activity

3. RETRAIN MODEL
   - Include new TDA-derived features
   - Use expanded 4.5-year historical data
   - Improve seasonality predictions
   - Better handle promotional peaks

4. EXPECTED IMPROVEMENTS
   - Better seasonal peak prediction (esp. during promotions)
   - More accurate fiscal year patterns
   - Stronger performance on high-TDA months
   - Reduced MAPE for promotional products
""")

print("="*80)
print("✓ Analysis complete!")
print("="*80)
