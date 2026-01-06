"""
TDA (Trade Display Allowance) Effectiveness Analysis Framework
Analyzing TDA impact on product groups across fiscal years
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

DATA_DIR = Path("data")

print("="*80)
print("TDA EFFECTIVENESS ANALYSIS - FRAMEWORK")
print("="*80)

# Load all TDA files
tda_files = sorted(list(DATA_DIR.glob("TDA*.xls*")))
all_tda = []

for tda_file in tda_files:
    df = pd.read_excel(tda_file, sheet_name=0)
    df['FISCAL_YEAR'] = tda_file.stem.split()[-1]  # Extract FY from filename
    all_tda.append(df)
    
df_tda_combined = pd.concat(all_tda, ignore_index=True)

# Convert date columns to datetime
df_tda_combined['TACTIC_START_DATE'] = pd.to_datetime(df_tda_combined['TACTIC_START_DATE'], errors='coerce')
df_tda_combined['TACTIC_END_DATE'] = pd.to_datetime(df_tda_combined['TACTIC_END_DATE'], errors='coerce')

print(f"\n\nCombined TDA Dataset:")
print(f"  Total records: {len(df_tda_combined):,}")
print(f"  Fiscal years: {sorted(df_tda_combined['FISCAL_YEAR'].unique())}")
print(f"  Date range: {df_tda_combined['TACTIC_START_DATE'].min()} to {df_tda_combined['TACTIC_END_DATE'].max()}")

# Key Analysis 1: TDA by Product Group
print("\n" + "="*80)
print("[1] TDA RECORDS BY PRODUCT GROUP")
print("="*80)

tda_by_product = df_tda_combined.groupby('PRODUCT_GROUP').size().sort_values(ascending=False)
print(f"\nTop 15 Product Groups by TDA Count:")
print(tda_by_product.head(15))

# Key Analysis 2: Tactic Types
print("\n" + "="*80)
print("[2] TDA TACTIC TYPES")
print("="*80)

tactic_dist = df_tda_combined['TACTIC_TYPE'].value_counts()
print(f"\nTactic Type Distribution:")
print(tactic_dist)

# Key Analysis 3: Budget Spent Type
print("\n" + "="*80)
print("[3] BUDGET SPENT TYPE CLASSIFICATION")
print("="*80)

budget_dist = df_tda_combined['BUDGET_SPENT_TYPE'].value_counts()
print(f"\nBudget Spent Types:")
print(budget_dist)

# Key Analysis 4: Activity Types
print("\n" + "="*80)
print("[4] PROMOTIONAL ACTIVITY TYPES")
print("="*80)

activity_dist = df_tda_combined['DETAILS_ACTIVITY_TYPE'].value_counts()
print(f"\nActivity Type Distribution (top 10):")
print(activity_dist.head(10))

# Key Analysis 5: Status
print("\n" + "="*80)
print("[5] TDA PLAN STATUS")
print("="*80)

status_dist = df_tda_combined['DETAILS_STATUS'].value_counts()
print(f"\nPlan Status Distribution:")
print(status_dist)

# Key Analysis 6: Geographic Distribution
print("\n" + "="*80)
print("[6] GEOGRAPHIC DISTRIBUTION")
print("="*80)

location_dist = df_tda_combined['SIP_LOCATION'].value_counts()
print(f"\nTDA by Location:")
print(location_dist)

# Key Analysis 7: Channel Type
print("\n" + "="*80)
print("[7] CHANNEL DISTRIBUTION")
print("="*80)

channel_dist = df_tda_combined['CHANNEL'].value_counts()
print(f"\nTDA by Channel (top 10):")
print(channel_dist.head(10))

# Key Analysis 8: Product Group × Tactic Type Matrix
print("\n" + "="*80)
print("[8] PRODUCT GROUP × TACTIC TYPE MATRIX")
print("="*80)

matrix = pd.crosstab(
    df_tda_combined['PRODUCT_GROUP'], 
    df_tda_combined['TACTIC_TYPE'],
    margins=True
)
print(f"\nTop 15 products by tactic distribution:")
print(matrix.head(15))

# Key Analysis 9: Timeline Analysis
print("\n" + "="*80)
print("[9] TDA ACTIVITY BY MONTH & FISCAL YEAR")
print("="*80)

# Parse MONTH column if available
if 'MONTH' in df_tda_combined.columns:
    month_dist = df_tda_combined['MONTH'].value_counts().sort_index()
    print(f"\nTDA Activity by Month:")
    print(month_dist.head(12))

fy_dist = df_tda_combined['FISCAL_YEAR'].value_counts().sort_index()
print(f"\nTDA Records by Fiscal Year:")
print(fy_dist)

# Key Analysis 10: Brand Distribution
print("\n" + "="*80)
print("[10] BRANDS IN TDA PROGRAMS")
print("="*80)

brand_dist = df_tda_combined['BRAND'].value_counts()
print(f"\nTop 20 Brands in TDA:")
print(brand_dist.head(20))

# Summary for Task 3 Approach
print("\n" + "="*80)
print("TASK 3 APPROACH - TDA EFFECTIVENESS ANALYSIS")
print("="*80)

print("""
AVAILABLE METRICS FOR TDA EFFECTIVENESS:

1. **TDA Volume**: Count of TDA plans/activities by product group
2. **TDA Duration**: Average duration (TACTIC_START_DATE to TACTIC_END_DATE)
3. **Activity Types**: Which activities (Display, VGR, etc.) per product group
4. **Geographic Spread**: How widely each product group's TDA is deployed
5. **Channel Coverage**: Modern Trade, Traditional Trade, E-commerce mix

NEXT STEP - DATA LINKAGE:
Need to link TDA data with Contract_Claim_Dump to calculate:
  - Sales during TDA periods vs non-TDA periods
  - Sales lift % = (Sales_TDA - Baseline) / Baseline × 100
  - ROI = Sales Lift / TDA Budget
  - Cost per incremental sale

KEY QUESTION:
How to join TDA with sales data?
  Option 1: By PRODUCT_GROUP + MONTH + LOCATION
  Option 2: By PRODUCT_GROUP + CHANNEL + FISCAL_YEAR
  Option 3: By BRAND + PRODUCT_GROUP + TIME_PERIOD
""")

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)
print(f"Unique TDA Plans: {df_tda_combined['TDA_PLAN_ID'].nunique():,}")
print(f"Unique Products: {df_tda_combined['PRODUCT_GROUP'].nunique()}")
print(f"Unique Brands: {df_tda_combined['BRAND'].nunique()}")
print(f"Unique Channels: {df_tda_combined['CHANNEL'].nunique()}")
print(f"Locations: {df_tda_combined['SIP_LOCATION'].nunique()}")
print(f"Date Range: {df_tda_combined['TACTIC_START_DATE'].min()} to {df_tda_combined['TACTIC_END_DATE'].max()}")
