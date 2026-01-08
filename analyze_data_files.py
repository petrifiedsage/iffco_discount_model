import pandas as pd
import os
from typing import Dict, List

# Required features
required_features = {
    'TACTIC_TYPE': 'Predictor and Label',
    'PRODUCT_GROUP': 'Product',
    'PRODUCT_SUB_GROUP': 'Product',
    'BRAND': 'Product',
    'CHANNEL_GROUP': 'Channel',
    'DEAL_TYPE': 'Deal',
    'AMENDED_DEAL_AMOUNT': 'Financial',
    'ADJUSTMENT_VALUE_AMOUNT': 'Financial',
    'COUNTRY': 'Location',
    'TERRITORY': 'Location',
    'MONTH': 'Time',
    'PERIOD': 'Time',
    'CLAIM_STATUS': 'Outcome Label',
    'CLAIM_AMOUNT': 'Outcome Label'
}

base_path = r'd:\BlueRose\iffco_discount_model\data\csv'

# Files to analyze
files_to_check = {
    'TACTIC_PLAN': 'TACTIC_24_25_TACTIC_PLAN_DIMNS.csv',
    'TACTIC_CLAIMS': 'TACTIC_24_25_TACTIC_CM_CLAIMS_DIMNS.csv',
    'TACTIC_SKU': 'TACTIC_24_25_SKU1.csv',
    'TDA_SIP_PLAN': 'TDA_FY 24_25_SIP_PLAN_DIMNS1.csv',
    'TDA_SIP_CLAIM': 'TDA_FY 24_25_SIP_CLAIM_DIMNS1.csv',
    'TDA_SIP_SKU': 'TDA_FY 24_25_SIP_SKU_DIMNS1.csv',
    'CONTRACT': 'CY2024_Contracts_export_CY 2024.csv',
    'CONTRACT_CLAIM_DUMP': 'Contract_Claim_Dump-Jan24-Dec24-UAE_Jan 2024.csv',
    'CONTRACT_WITH_PG': 'Contract_Data_with_Product Group_IDS_OU_2025_Contracts.csv'
}

results = {}

for file_type, filename in files_to_check.items():
    filepath = os.path.join(base_path, filename)
    try:
        df = pd.read_csv(filepath, nrows=1)
        cols = df.columns.tolist()
        
        # Check which required features are present
        present = []
        missing = []
        for feature in required_features.keys():
            if feature in cols:
                present.append(feature)
            else:
                missing.append(feature)
        
        # Check for additional useful features
        additional = []
        useful_patterns = ['STATUS', 'APPROVAL', 'BUDGET', 'GL_CODE', 'COST', 'SALES', 'REGISTRY', 'PARTY_SITE']
        for col in cols:
            if any(pattern in col.upper() for pattern in useful_patterns):
                if col not in required_features:
                    additional.append(col)
        
        results[file_type] = {
            'total_columns': len(cols),
            'all_columns': cols,
            'present_required': present,
            'missing_required': missing,
            'coverage_pct': round(len(present) / len(required_features) * 100, 1),
            'additional_useful': additional[:15]
        }
    except Exception as e:
        results[file_type] = {'error': str(e)}

# Print results
print('='*80)
print('DATA FILE FEATURE COVERAGE ANALYSIS')
print('='*80)

for file_type, data in results.items():
    if 'error' in data:
        err_msg = data['error']
        print(f'\n{file_type}: ERROR - {err_msg}')
        continue
    
    coverage = data['coverage_pct']
    present_count = len(data['present_required'])
    total_required = len(required_features)
    
    print(f'\n{file_type}:')
    print(f'  Coverage: {coverage}% ({present_count}/{total_required})')
    print(f'  Total columns: {data["total_columns"]}')
    print(f'  Present: {", ".join(data["present_required"])}')
    if data['missing_required']:
        print(f'  Missing: {", ".join(data["missing_required"])}')
    if data['additional_useful']:
        print(f'  Useful extras: {", ".join(data["additional_useful"])}')

print('\n' + '='*80)
print('SUMMARY AND RECOMMENDATIONS')
print('='*80)

# Sort by coverage
sorted_results = sorted(
    [(k, v) for k, v in results.items() if 'error' not in v],
    key=lambda x: x[1]['coverage_pct'],
    reverse=True
)

print('\nFiles ranked by feature coverage:')
for i, (file_type, data) in enumerate(sorted_results, 1):
    print(f'{i}. {file_type}: {data["coverage_pct"]}% coverage')

# Find files with best label coverage
print('\nFiles with outcome labels (CLAIM_STATUS, CLAIM_AMOUNT):')
for file_type, data in results.items():
    if 'error' not in data:
        has_status = 'CLAIM_STATUS' in data['present_required']
        has_amount = 'CLAIM_AMOUNT' in data['present_required']
        if has_status or has_amount:
            labels = []
            if has_status:
                labels.append('CLAIM_STATUS')
            if has_amount:
                labels.append('CLAIM_AMOUNT')
            print(f'  {file_type}: {", ".join(labels)}')

# Check for TACTIC_TYPE presence
print('\nFiles with TACTIC_TYPE (target variable):')
for file_type, data in results.items():
    if 'error' not in data:
        if 'TACTIC_TYPE' in data['present_required']:
            print(f'  {file_type}: YES')

print('\n' + '='*80)
print('DETAILED RECOMMENDATIONS')
print('='*80)

print('\n1. BEST FILES FOR COMPLETE TRAINING DATA:')
best_files = [ft for ft, d in sorted_results[:3]]
for file_type in best_files:
    data = results[file_type]
    print(f'\n   {file_type}:')
    print(f'     - {data["coverage_pct"]}% feature coverage')
    print(f'     - Has: {", ".join(data["present_required"][:5])}...')
    print(f'     - Missing: {", ".join(data["missing_required"][:5])}')

print('\n2. FILES TO COMBINE FOR ENHANCED MODEL:')
print('   Recommended combination:')

# Check which combinations would give best coverage
tactic_claims_present = set(results['TACTIC_CLAIMS']['present_required'])
tactic_plan_present = set(results['TACTIC_PLAN']['present_required'])
combined = tactic_claims_present.union(tactic_plan_present)
combined_pct = round(len(combined) / len(required_features) * 100, 1)
print(f'   - TACTIC_CLAIMS + TACTIC_PLAN: {combined_pct}% coverage')
print(f'     Combined features: {len(combined)}/{len(required_features)}')

# Check TDA
tda_claim_present = set(results['TDA_SIP_CLAIM']['present_required'])
tda_plan_present = set(results['TDA_SIP_PLAN']['present_required'])
tda_combined = tda_claim_present.union(tda_plan_present)
tda_combined_pct = round(len(tda_combined) / len(required_features) * 100, 1)
print(f'   - TDA_SIP_CLAIM + TDA_SIP_PLAN: {tda_combined_pct}% coverage')
print(f'     Combined features: {len(tda_combined)}/{len(required_features)}')

# Check Contract Claim Dump
contract_dump_present = set(results['CONTRACT_CLAIM_DUMP']['present_required'])
contract_dump_pct = results['CONTRACT_CLAIM_DUMP']['coverage_pct']
print(f'   - CONTRACT_CLAIM_DUMP alone: {contract_dump_pct}% coverage')

print('\n3. NEW FEATURES DISCOVERED:')
all_additional = set()
for file_type, data in results.items():
    if 'error' not in data:
        all_additional.update(data['additional_useful'])

print(f'   Total new useful features found: {len(all_additional)}')
print(f'   Examples: {", ".join(list(all_additional)[:10])}')

print('\n4. MISSING FEATURES TO ADDRESS:')
all_missing = set()
for file_type, data in results.items():
    if 'error' not in data:
        all_missing.update(data['missing_required'])

if all_missing:
    print(f'   Features not found in any file: {", ".join(all_missing)}')
else:
    print('   All required features found across files!')

print('\n' + '='*80)
