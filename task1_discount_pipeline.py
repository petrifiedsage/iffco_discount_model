"""
import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# 1. Auto-Discovery Pattern (Matches your export files)
FILE_MATCH_PATTERN = "*Contracts_export*.xls*"

# 2. Specific Extra File (The one you requested)
EXTRA_FILE_PATH = os.path.join(DATA_DIR, "Contract_Data_with_Product Group_IDS_OU_2025.xls")

FEATURES = [
    'CUSTOMER', 'GLOBAL_ACCOUNT', 'CHANNEL', 'TYPE', 'TACTIC_TYPE',
    'PRODUCT_GROUP', 'START_MONTH_NUM', 'END_MONTH_NUM'
]

def clean_text(x):
    return str(x).strip().title()

def train_task1_final():
    print("--- Starting Task 1: Discount Optimizer (Combined Datasets) ---")
    
    # 1. FILE DISCOVERY
    search_path = os.path.join(DATA_DIR, FILE_MATCH_PATTERN)
    found_files = glob.glob(search_path)
    
    # Add the extra file explicitly if it exists
    if os.path.exists(EXTRA_FILE_PATH):
        print(f"   + Found Extra File: {os.path.basename(EXTRA_FILE_PATH)}")
        found_files.append(EXTRA_FILE_PATH)
    else:
        print(f"   ⚠️ Warning: Extra file not found at {EXTRA_FILE_PATH}")

    if not found_files:
        print(f"❌ Error: No files found in {DATA_DIR}")
        return

    print(f"Found {len(found_files)} total files. Loading...")
    
    df_list = []
    for f in found_files:
        try:
            print(f"   > Reading: {os.path.basename(f)}...")
            
            # Use xlrd for .xls files
            if f.lower().endswith('.xls'):
                df = pd.read_excel(f, engine='xlrd')
            else:
                df = pd.read_excel(f)
            
            # Standardize Headers
            df.columns = df.columns.str.strip().str.upper()
            
            # Normalize Column Names (Mapping different schemas to one standard)
            rename_map = {
                'CONTRACT#': 'CONTRACT_ID',
                'CREATED_DATE': 'DATE',
                'CUSTOMER_NAME': 'CUSTOMER',
                'PRODUCT_GROUP': 'PRODUCT_GROUP',
                'CHANNEL': 'CHANNEL',
                'TACTIC_TYPE': 'TACTIC_TYPE'
            }
            df = df.rename(columns=rename_map)

            df_list.append(df)
        except Exception as e:
            print(f"   ! Error reading {os.path.basename(f)}: {e}")

    if not df_list: return

    full_df = pd.concat(df_list, ignore_index=True, sort=False)
    print(f"📊 Total Raw Rows: {len(full_df):,}")

    # 2. DEDUPLICATION (Crucial when mixing overlapping datasets)
    before_dedup = len(full_df)
    # We drop duplicates based on key columns to avoid training on the exact same contract twice
    # If CONTRACT_ID exists, use it. Otherwise, use all columns.
    if 'CONTRACT_ID' in full_df.columns:
        full_df = full_df.drop_duplicates(subset=['CONTRACT_ID', 'CUSTOMER', 'PRODUCT_GROUP', 'ACTUAL_REBATE'])
    else:
        full_df = full_df.drop_duplicates()
        
    print(f"   ✂️ Removed {before_dedup - len(full_df):,} duplicate rows.")
    print(f"   ✅ Final Training Set: {len(full_df):,} rows.")

    # 3. DATA CLEANING & PREP
    
    # A. Target Variable (Actual Rebate %)
    if 'ACTUAL_REBATE' in full_df.columns:
        full_df['Target_Value'] = full_df['ACTUAL_REBATE'].fillna(0)
        # Fallback to Fixed Amount if Actual is missing
        if 'FIXED_REBATE_AMOUNT' in full_df.columns:
            full_df['Target_Value'] = full_df['Target_Value'].fillna(full_df['FIXED_REBATE_AMOUNT'])
    else:
        print("❌ Critical Error: 'ACTUAL_REBATE' column not found.")
        return

    # Filter: Train only on valid Percentages (e.g. 0.5% to 40%)
    df_model = full_df[(full_df['Target_Value'] > 0) & (full_df['Target_Value'] <= 40)].copy()

    # B. Date Features
    if 'START_MONTH' in df_model.columns:
        df_model['START_MONTH'] = pd.to_datetime(df_model['START_MONTH'], errors='coerce', dayfirst=True)
        df_model['START_MONTH_NUM'] = df_model['START_MONTH'].dt.month
    else:
        df_model['START_MONTH_NUM'] = 1 

    if 'END_MONTH' in df_model.columns:
        df_model['END_MONTH'] = pd.to_datetime(df_model['END_MONTH'], errors='coerce', dayfirst=True)
        df_model['END_MONTH_NUM'] = df_model['END_MONTH'].dt.month
    else:
        df_model['END_MONTH_NUM'] = 12

    df_model['START_MONTH_NUM'] = df_model['START_MONTH_NUM'].fillna(1)
    df_model['END_MONTH_NUM'] = df_model['END_MONTH_NUM'].fillna(12)

    # C. Text Cleaning
    cols_to_clean = ['CUSTOMER', 'PRODUCT_GROUP', 'CHANNEL', 'TACTIC_TYPE', 'TYPE', 'GLOBAL_ACCOUNT']
    for col in cols_to_clean:
        if col in df_model.columns:
            df_model[col] = df_model[col].apply(clean_text)
        else:
            df_model[col] = "Unknown"

    # 4. METADATA EXTRACTION
    metadata = {
        'customers': sorted(df_model['CUSTOMER'].unique().tolist()),
        'products': sorted(df_model['PRODUCT_GROUP'].unique().tolist()),
        'channels': sorted(df_model['CHANNEL'].unique().tolist()),
        'tactics': sorted(df_model['TACTIC_TYPE'].unique().tolist()),
        'types': sorted(df_model['TYPE'].unique().tolist()),
        'cust_global_map': {}
    }
    
    if 'GLOBAL_ACCOUNT' in df_model.columns:
        temp = df_model.groupby('CUSTOMER')['GLOBAL_ACCOUNT'].first()
        metadata['cust_global_map'] = temp.to_dict()

    # 5. TRAINING
    X = df_model[FEATURES]
    y = df_model['Target_Value']

    categorical_cols = X.select_dtypes(include=['object']).columns
    numeric_cols = X.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ])

    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
    ])

    print(f"Training on {len(df_model)} validated rows...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)

    df_model['YEAR'] = pd.to_datetime(df_model['DATE'], errors='coerce').dt.year

    train_df = df_model[df_model['YEAR'] <= 2023]
    test_df  = df_model[df_model['YEAR'] >= 2024]

    X_train = train_df[FEATURES]
    y_train = train_df['Target_Value']

    X_test  = test_df[FEATURES]
    y_test  = test_df['Target_Value']

    
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"✅ Model Trained! Average Error (MAE): {mae:.2f}%")

    # 6. SAVE
    if not os.path.exists(MODELS_DIR): os.makedirs(MODELS_DIR)
    
    joblib.dump(model, os.path.join(MODELS_DIR, 'task1_discount_regressor_v3.pkl'))
    joblib.dump(metadata, os.path.join(MODELS_DIR, 'task1_metadata_v3.pkl'))
    print("✅ Files saved. Restart your Streamlit App.")

if __name__ == "__main__":
    train_task1_final()

"""

import pandas as pd
import numpy as np
import os
import glob
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error

# --- CONFIGURATION ---
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

FILE_MATCH_PATTERN = "*Contracts_export*.xls*"
EXTRA_FILE_PATH = os.path.join(DATA_DIR, "Contract_Data_with_Product Group_IDS_OU_2025.xls")

# ✅ LEAK-FREE FEATURES
FEATURES = [
    'CUSTOMER',
    'GLOBAL_ACCOUNT',
    'CHANNEL',
    'TYPE',
    'TACTIC_TYPE',
    'PRODUCT_GROUP',
    'YEAR_CREATED'
]

def clean_text(x):
    return str(x).strip().title()

def train_task1_final():
    print("--- Starting Task 1: Discount Optimizer (Final, Correct) ---")

    # 1. LOAD FILES
    search_path = os.path.join(DATA_DIR, FILE_MATCH_PATTERN)
    found_files = glob.glob(search_path)

    if os.path.exists(EXTRA_FILE_PATH):
        found_files.append(EXTRA_FILE_PATH)

    if not found_files:
        print("❌ No files found")
        return

    df_list = []
    for f in found_files:
        try:
            df = pd.read_excel(f, engine='xlrd' if f.lower().endswith('.xls') else None)
            df.columns = df.columns.str.strip().str.upper()

            rename_map = {
                'CONTRACT#': 'CONTRACT_ID',
                'CREATED_DATE': 'DATE',
                'CUSTOMER_NAME': 'CUSTOMER'
            }
            df = df.rename(columns=rename_map)

            df_list.append(df)

        except Exception as e:
            print(f"Error reading {f}: {e}")

    full_df = pd.concat(df_list, ignore_index=True)
    print(f"📊 Raw rows loaded: {len(full_df):,}")

    # 2. ✅ CORRECT DEDUPLICATION (CONTRACT + PRODUCT)
    dedup_cols = ['CONTRACT_ID', 'CUSTOMER', 'PRODUCT_GROUP']
    dedup_cols = [c for c in dedup_cols if c in full_df.columns]

    before = len(full_df)
    full_df = full_df.drop_duplicates(subset=dedup_cols)
    print(f"✂️ Removed {before - len(full_df):,} duplicate rows")

    # 3. TARGET (STRICT)
    if 'ACTUAL_REBATE' not in full_df.columns:
        print("❌ ACTUAL_REBATE column missing")
        return

    df_model = full_df[
        (full_df['ACTUAL_REBATE'] > 0) &
        (full_df['ACTUAL_REBATE'] <= 40)
    ].copy()

    df_model['Target_Value'] = df_model['ACTUAL_REBATE']

    # 4. DATE HANDLING (PRE-CONTRACT ONLY)
    df_model['DATE'] = pd.to_datetime(df_model['DATE'], errors='coerce', dayfirst=True)
    df_model = df_model[df_model['DATE'].notna()]
    df_model['YEAR_CREATED'] = df_model['DATE'].dt.year

    # 5. TEXT CLEANING
    cols_to_clean = ['CUSTOMER', 'PRODUCT_GROUP', 'CHANNEL', 'TACTIC_TYPE', 'TYPE', 'GLOBAL_ACCOUNT']
    for col in cols_to_clean:
        if col in df_model.columns:
            df_model[col] = df_model[col].apply(clean_text)
        else:
            df_model[col] = "Unknown"

    # 6. TIME-BASED SPLIT
    train_df = df_model[df_model['YEAR_CREATED'] <= 2023]
    test_df  = df_model[df_model['YEAR_CREATED'] >= 2024]

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows : {len(test_df):,}")

    X_train = train_df[FEATURES]
    y_train = train_df['Target_Value']

    X_test = test_df[FEATURES]
    y_test = test_df['Target_Value']

    # 7. PIPELINE
    categorical_cols = X_train.select_dtypes(include=['object']).columns
    numeric_cols = X_train.select_dtypes(exclude=['object']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=15,
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print(f"✅ Final Leak-Free MAE: {mae:.2f}%")

    # 8. SAVE OUTPUTS
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model, os.path.join(MODELS_DIR, 'task1_discount_regressor_FINAL.pkl'))

    test_df = test_df.copy()
    test_df.loc[:, 'PREDICTED_REBATE'] = preds

    test_df[['CUSTOMER', 'PRODUCT_GROUP', 'Target_Value', 'PREDICTED_REBATE']] \
        .to_csv(os.path.join(MODELS_DIR, 'rebate_recommendations_FINAL.csv'), index=False)

    print("✅ Model & predictions saved successfully")

if __name__ == "__main__":
    train_task1_final()
