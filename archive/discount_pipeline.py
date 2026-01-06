import argparse
import os
import sys
from typing import List, Optional

REQUIRED_COLUMNS: List[str] = [
    'CUSTOMER',
    'GLOBAL_ACCOUNT',
    'CHANNEL',
    'TYPE',
    'TACTIC_TYPE',
    'PRODUCT_GROUP',
    'ORGANIZATION',
    'MFG_ORGANIZATION',
    'CREATED_DATE',
    'START_MONTH',
    'END_MONTH',
    'ACTUAL_REBATE'
]

FEATURES: List[str] = [
    'CUSTOMER',
    'GLOBAL_ACCOUNT',
    'CHANNEL',
    'TYPE',
    'TACTIC_TYPE',
    'PRODUCT_GROUP',
    'ORGANIZATION',
    'MFG_ORGANIZATION',
    'CREATED_YEAR',
    'CREATED_MONTH',
    'START_MONTH_NUM',
    'END_MONTH_NUM'
]

VALID_REBATES = {10.0, 12.5, 15.0}


def parse_args():
    parser = argparse.ArgumentParser(description='Discount Recommendation Engine (Task 1)')
    parser.add_argument('--data-path', default=os.path.join('iffco_discount_model', 'data', 'contracts.xls'),
                        help='Path to Excel dataset (.xls or .xlsx)')
    parser.add_argument('--outputs-dir', default=os.path.join('iffco_discount_model', 'outputs'),
                        help='Directory to write outputs')
    parser.add_argument('--models-dir', default=os.path.join('iffco_discount_model', 'models'),
                        help='Directory to write models')
    parser.add_argument('--check-only', action='store_true', help='Validate dataset and columns without training')
    parser.add_argument('--sheet-name', default=None, help='Excel sheet name (default: first sheet)')
    return parser.parse_args()


def ensure_dirs(outputs_dir: str, models_dir: str):
    os.makedirs(outputs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)


def check_dataset_header(data_path: str, sheet_name: Optional[str]) -> List[str]:
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset not found at {data_path}")
        return []
    try:
        import pandas as pd
        engine = 'xlrd' if data_path.lower().endswith('.xls') else None
        df_head = pd.read_excel(data_path, sheet_name=sheet_name, nrows=0, engine=engine)
        return list(df_head.columns)
    except Exception as e:
        # Fallback to pyexcel for legacy .xls support
        try:
            import pyexcel as pe
            if sheet_name:
                book = pe.get_book(file_name=data_path)
                if sheet_name in book.sheet_names():
                    sheet = book[sheet_name]
                else:
                    sheet = book[0]
            else:
                sheet = pe.get_sheet(file_name=data_path)
            data = sheet.get_array()
            if not data:
                print('ERROR: Excel sheet is empty.')
                return []
            header = [str(h).strip() for h in data[0]]
            return header
        except Exception as e2:
            print("ERROR: Failed to read Excel header via pandas and pyexcel.")
            print(f"pandas error: {e}")
            print(f"pyexcel error: {e2}")
            return []


def round_to_slab(x: float) -> float:
    if x <= 11:
        return 10.0
    elif x <= 13.75:
        return 12.5
    else:
        return 15.0


def apply_business_rules(row):
    rebate = row['Predicted_Rebate_Slab']
    if row.get('CHANNEL') == 'GT':
        rebate = min(rebate, 10.0)
    if row.get('CHANNEL') == 'LOC_MT' and row.get('PRODUCT_GROUP') == 'Ice creams':
        rebate = min(rebate, 15.0)
    if row.get('PRODUCT_GROUP') == 'Bakery':
        rebate = min(rebate, 12.5)
    return rebate


def _load_dataframe_excel(data_path: str, sheet_name: Optional[str]):
    import pandas as pd
    try:
        engine = 'xlrd' if data_path.lower().endswith('.xls') else None
        return pd.read_excel(data_path, sheet_name=sheet_name, engine=engine)
    except Exception:
        # Fallback to pyexcel for legacy .xls
        try:
            import pyexcel as pe
            if sheet_name:
                book = pe.get_book(file_name=data_path)
                if sheet_name in book.sheet_names():
                    sheet = book[sheet_name]
                else:
                    sheet = book[0]
            else:
                sheet = pe.get_sheet(file_name=data_path)
            data = sheet.get_array()
            if not data:
                raise ValueError('Excel sheet is empty.')
            header = [str(h).strip() for h in data[0]]
            rows = data[1:]
            return pd.DataFrame(rows, columns=header)
        except Exception as e2:
            raise RuntimeError('Failed to load Excel via pandas and pyexcel.') from e2


def train_and_recommend(data_path: str, outputs_dir: str, models_dir: str, sheet_name: Optional[str]):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error
    import joblib

    df = _load_dataframe_excel(data_path, sheet_name)
    # Dates
    df['CREATED_DATE'] = pd.to_datetime(df.get('CREATED_DATE'), errors='coerce', dayfirst=True, format='mixed')
    df['START_MONTH'] = pd.to_datetime(df.get('START_MONTH'), errors='coerce', dayfirst=True, format='mixed')
    df['END_MONTH'] = pd.to_datetime(df.get('END_MONTH'), errors='coerce', dayfirst=True, format='mixed')

    df['CREATED_YEAR'] = df['CREATED_DATE'].dt.year
    df['CREATED_MONTH'] = df['CREATED_DATE'].dt.month
    df['START_MONTH_NUM'] = df['START_MONTH'].dt.month
    df['END_MONTH_NUM'] = df['END_MONTH'].dt.month

    # Target cleanup
    if 'ACTUAL_REBATE' not in df.columns:
        raise ValueError('ACTUAL_REBATE column missing from dataset.')
    df['ACTUAL_REBATE'] = pd.to_numeric(df['ACTUAL_REBATE'], errors='coerce')
    df = df.dropna(subset=['ACTUAL_REBATE'])
    # Normalize to allowed slabs to ensure discrete classification labels
    df['ACTUAL_REBATE_SLAB'] = df['ACTUAL_REBATE'].apply(round_to_slab)

    # Feature availability check
    available_features = [c for c in FEATURES if c in df.columns]
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        print(f"WARNING: Missing feature columns: {missing}. Proceeding with available features: {available_features}")
    if not available_features:
        raise ValueError('No feature columns available for training.')

    X = df[available_features]
    y = df['ACTUAL_REBATE_SLAB'].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    categorical_cols = X.select_dtypes(include='object').columns
    numeric_cols = X.select_dtypes(exclude='object').columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols),
            ('num', 'passthrough', numeric_cols)
        ]
    )

    model = Pipeline(steps=[
        ('preprocess', preprocessor),
        ('classifier', RandomForestClassifier(
            n_estimators=300,
            max_depth=14,
            min_samples_leaf=20,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        ))
    ])

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print('Accuracy:', accuracy_score(y_test, y_pred))
    print('\nClassification Report:\n')
    print(classification_report(y_test, y_pred))

    # MAE on test set (in percentage points)
    y_test_num = pd.to_numeric(y_test, errors='coerce')
    y_pred_num = pd.to_numeric(y_pred, errors='coerce')
    mae_test = mean_absolute_error(y_test_num, y_pred_num)
    print(f"MAE (pp) on test (Predicted slab vs Actual): {mae_test:.3f}")

    # Predict for all
    df['Predicted_Rebate'] = pd.to_numeric(model.predict(X), errors='coerce')
    df['Predicted_Rebate'] = df['Predicted_Rebate'].fillna(10.0)
    df['Predicted_Rebate_Slab'] = df['Predicted_Rebate'].apply(round_to_slab)

    df['Recommended_Rebate'] = df.apply(apply_business_rules, axis=1)

    # MAE for Recommended Rebate on the same test rows
    if set(X_test.index).issubset(set(df.index)):
        rec_test = pd.to_numeric(df.loc[X_test.index, 'Recommended_Rebate'], errors='coerce')
        act_test = pd.to_numeric(df.loc[X_test.index, 'ACTUAL_REBATE'], errors='coerce')
        mae_rec = mean_absolute_error(act_test, rec_test)
        print(f"MAE (pp) on test (Recommended vs Actual): {mae_rec:.3f}")

    final_cols = [
        'CUSTOMER', 'PRODUCT_GROUP', 'CHANNEL', 'TACTIC_TYPE',
        'ACTUAL_REBATE', 'Recommended_Rebate'
    ]
    existing_final = [c for c in final_cols if c in df.columns]
    final_df = df[existing_final].copy()
    if 'ACTUAL_REBATE' in final_df.columns and 'Recommended_Rebate' in final_df.columns:
        final_df['Rebate_Change'] = final_df['Recommended_Rebate'] - final_df['ACTUAL_REBATE']

    # Writes
    ensure_dirs(outputs_dir, models_dir)
    out_path = os.path.join(outputs_dir, 'rebate_recommendations.csv')
    try:
        final_df.to_csv(out_path, index=False)
        print(f'Wrote recommendations to: {out_path}')
    except PermissionError:
        import datetime as _dt
        ts = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        alt_path = os.path.join(outputs_dir, f'rebate_recommendations_{ts}.csv')
        final_df.to_csv(alt_path, index=False)
        print(f'Permission denied for {out_path}. Wrote to: {alt_path}')

    model_path = os.path.join(models_dir, 'discount_model.pkl')
    joblib.dump(model, model_path)
    print(f'Saved model to: {model_path}')


if __name__ == '__main__':
    args = parse_args()
    ensure_dirs(args.outputs_dir, args.models_dir)

    if args.check_only:
        header = check_dataset_header(args.data_path, args.sheet_name)
        if not header:
            sys.exit(1)
        missing = [c for c in REQUIRED_COLUMNS if c not in header]
        if missing:
            print(f'MISSING COLUMNS: {missing}')
            sys.exit(2)
        print('Dataset and required columns look good.')
        sys.exit(0)

    train_and_recommend(args.data_path, args.outputs_dir, args.models_dir, args.sheet_name)
