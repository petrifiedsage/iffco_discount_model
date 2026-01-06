import pandas as pd
import numpy as np
import os

from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

import joblib

file_path = "D:/BlueRose/iffco_discount_model/data/Contract_Claim_Dump-Jan25-Nov25_IDS_OU.xlsx"

xls = pd.ExcelFile(file_path)
df_list = []

for sheet in xls.sheet_names:
    temp = pd.read_excel(xls, sheet_name=sheet)
    temp['SOURCE_SHEET'] = sheet
    df_list.append(temp)

df = pd.concat(df_list, ignore_index=True)

print("Raw shape:", df.shape)
df.head()

df = df[
    [
        'CLAIM_DATE',
        'MONTH',
        'GEOGRAPHICAL_CLUSTER',
        'CUSTOMER_NAME',
        'CUSTOMER_CHANNEL',
        'PRODUCT_GROUP',
        'TACTIC_TYPE',
        'AMOUNT'
    ]
]

df['CLAIM_DATE'] = pd.to_datetime(df['CLAIM_DATE'], errors='coerce', format='mixed')

df['YEAR'] = df['CLAIM_DATE'].dt.year
df['MONTH_NUM'] = df['CLAIM_DATE'].dt.month

def map_season(month):
    if month in [3, 4, 5]:
        return 'Summer'
    elif month in [6, 7, 8]:
        return 'Peak Summer'
    elif month in [9, 10]:
        return 'Monsoon'
    else:
        return 'Winter'

df['SEASON'] = df['MONTH_NUM'].apply(map_season)

df['AMOUNT'] = pd.to_numeric(df['AMOUNT'], errors='coerce')
df = df.dropna(subset=['AMOUNT'])

monthly_demand = (
    df
    .groupby(
        [
            'YEAR',
            'MONTH_NUM',
            'SEASON',
            'PRODUCT_GROUP',
            'CUSTOMER_CHANNEL'
        ],
        as_index=False
    )
    .agg(
        MONTHLY_CLAIM_AMOUNT=('AMOUNT', 'sum')
    )
)

monthly_demand = monthly_demand.sort_values(['YEAR', 'MONTH_NUM'])
# Scale target to Lakhs (1 Lakh = 100,000) before splitting
monthly_demand['MONTHLY_CLAIM_LAKH'] = monthly_demand['MONTHLY_CLAIM_AMOUNT'] / 1e5

print("Monthly demand shape:", monthly_demand.shape)
print("Year distribution:")
print(monthly_demand['YEAR'].value_counts().sort_index())

monthly_demand.head()

# Flexible train/test split: use last 2 months for test if all data is from 2025
unique_years = monthly_demand['YEAR'].unique()
if len(unique_years) == 1 or monthly_demand[monthly_demand['YEAR'] < 2025].shape[0] == 0:
    # All data from single year or no pre-2025 data: use last 2 months as test
    max_month = monthly_demand['MONTH_NUM'].max()
    train_df = monthly_demand[monthly_demand['MONTH_NUM'] < max_month - 1]
    test_df = monthly_demand[monthly_demand['MONTH_NUM'] >= max_month - 1]
    print(f"Single year detected. Using months < {max_month - 1} for train, >= {max_month - 1} for test.")
else:
    train_df = monthly_demand[monthly_demand['YEAR'] < 2025]
    test_df = monthly_demand[monthly_demand['YEAR'] == 2025]
    print(f"Multi-year split: train < 2025, test = 2025.")

print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

if train_df.shape[0] == 0:
    raise ValueError("Training set is empty. Check your data date range and filtering logic.")

FEATURES = [
    'MONTH_NUM',
    'SEASON',
    'PRODUCT_GROUP',
    'CUSTOMER_CHANNEL'
]

TARGET = 'MONTHLY_CLAIM_LAKH'

X_train = train_df[FEATURES]
y_train = train_df[TARGET]

X_test = test_df[FEATURES]
y_test = test_df[TARGET]

categorical_cols = ['SEASON', 'PRODUCT_GROUP', 'CUSTOMER_CHANNEL']
numeric_cols = ['MONTH_NUM']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=True), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

model = Pipeline(
    steps=[
        ('preprocess', preprocessor),
        ('regressor', RandomForestRegressor(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1
        ))
    ]
)

model.fit(X_train, y_train)

preds = model.predict(X_test)

mae_lakh = mean_absolute_error(y_test, preds)
mae_rupees = mae_lakh * 1e5
print(f"MAE (Lakhs): {mae_lakh:.3f}")
print(f"MAE (Rupees): {mae_rupees:,.0f}")
print("R2 :", r2_score(y_test, preds))

future_df = monthly_demand.copy()
future_df['YEAR'] = 2026

future_df['FORECASTED_CLAIM_LAKH'] = model.predict(future_df[FEATURES])
future_df['FORECASTED_CLAIM_AMOUNT'] = future_df['FORECASTED_CLAIM_LAKH'] * 1e5

future_df.head()

os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

future_df.to_csv(
    "outputs/task2_demand_forecast.csv",
    index=False
)

joblib.dump(model, "models/task2_demand_forecast_model.pkl")

