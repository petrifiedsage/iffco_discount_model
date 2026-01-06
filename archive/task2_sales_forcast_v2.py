import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import glob
# ----------------------------------
# CONFIG
# ----------------------------------
BASE_DIR = r"D:/BlueRose/iffco_discount_model"
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLAIMS_DIR = r"D:/BlueRose/iffco_discount_model/data/contracts_claims_extracted"

csv_files = glob.glob(os.path.join(CLAIMS_DIR, "*.csv"))

if not csv_files:
    raise FileNotFoundError("❌ No CSV files found in contracts_claims_extracted")

print(f"📂 Found {len(csv_files)} monthly claim files")

df_list = []
for f in csv_files:
    try:
        temp = pd.read_csv(f)
        temp.columns = temp.columns.str.strip().str.upper()
        df_list.append(temp)
    except Exception as e:
        print(f"⚠️ Skipped {os.path.basename(f)}: {e}")

df = pd.concat(df_list, ignore_index=True)
print(f"📊 Total raw claim rows loaded: {len(df):,}")

os.makedirs(MODELS_DIR, exist_ok=True)

# ----------------------------------
# 1. LOAD CLAIM DATA (UAE ONLY)
# ----------------------------------
print("Loading claim dump...")

df.columns = df.columns.str.strip().str.upper()

use_cols = [
    "CLAIM_DATE",
    "CUSTOMER_NAME",
    "PRODUCT_GROUP",
    "AMOUNT",
    "GEOGRAPHICAL_CLUSTER"
]

df = df[use_cols].copy()
df = df.rename(columns={
    "CUSTOMER_NAME": "CUSTOMER",
    "GEOGRAPHICAL_CLUSTER": "REGION"
})

df["CLAIM_DATE"] = pd.to_datetime(df["CLAIM_DATE"], errors="coerce")
df = df[df["CLAIM_DATE"].notna()]
df = df[df["AMOUNT"] > 0]

df["YEAR"] = df["CLAIM_DATE"].dt.year
df["MONTH"] = df["CLAIM_DATE"].dt.month
df["PLOT_DATE"] = df["CLAIM_DATE"].dt.to_period("M").dt.to_timestamp()

print(f"Clean rows: {len(df):,}")

# ----------------------------------
# 2. AGGREGATE (MONTHLY)
# ----------------------------------
monthly = (
    df.groupby(["CUSTOMER", "PRODUCT_GROUP", "REGION", "YEAR", "MONTH", "PLOT_DATE"])
      .agg(
          SALES=("AMOUNT", "sum"),
          CLAIM_COUNT=("AMOUNT", "count")
      )
      .reset_index()
)

# ----------------------------------
# 3. FEATURE ENGINEERING
# ----------------------------------
print("Engineering features...")

# Long-term averages
cust_avg = monthly.groupby("CUSTOMER")["SALES"].mean().to_dict()
prod_avg = monthly.groupby("PRODUCT_GROUP")["SALES"].mean().to_dict()
geo_avg  = monthly.groupby("REGION")["SALES"].mean().to_dict()

cust_prod_avg = (
    monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"]
    .mean()
    .to_dict()
)

# Recent 3-month rolling average
monthly = monthly.sort_values("PLOT_DATE")

monthly["RECENT_AVG"] = (
    monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["SALES"]
    .rolling(3, min_periods=1)
    .mean()
    .reset_index(level=[0, 1], drop=True)
)

monthly["Cust_Avg"] = monthly["CUSTOMER"].map(cust_avg)
monthly["Prod_Avg"] = monthly["PRODUCT_GROUP"].map(prod_avg)
monthly["Geo_Avg"]  = monthly["REGION"].map(geo_avg)
monthly["Cust_Prod_Avg"] = monthly.apply(
    lambda r: cust_prod_avg.get((r["CUSTOMER"], r["PRODUCT_GROUP"]), r["Prod_Avg"]),
    axis=1
)

monthly["Is_Summer"] = monthly["MONTH"].isin([6, 7, 8]).astype(int)

monthly = monthly.fillna(0)

# ----------------------------------
# 4. TRAIN / TEST SPLIT (TIME-BASED)
# ----------------------------------
train_df = monthly[monthly["YEAR"] <= 2023]
test_df  = monthly[monthly["YEAR"] >= 2024]

FEATURES = [
    "Cust_Avg",
    "Prod_Avg",
    "Geo_Avg",
    "Cust_Prod_Avg",
    "RECENT_AVG",
    "MONTH",
    "Is_Summer"
]

X_train = train_df[FEATURES]
y_train = np.log1p(train_df["SALES"])

X_test = test_df[FEATURES]
y_test = np.log1p(test_df["SALES"])

print(f"Train rows: {len(train_df):,}")
print(f"Test rows : {len(test_df):,}")

# ----------------------------------
# 5. TRAIN MODEL
# ----------------------------------
print("Training model...")

model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=4,
    random_state=42
)

model.fit(X_train, y_train)

preds = model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print(f"✅ Log-MAE: {mae:.4f}")

# ----------------------------------
# MAE IN AED (INVERSE TRANSFORM)
# ----------------------------------

# Convert log predictions back to AED
y_test_aed = np.expm1(y_test)
preds_aed = np.expm1(preds)

mae_aed = mean_absolute_error(y_test_aed, preds_aed)

print(f"✅ MAE (AED): {mae_aed:,.0f}")


# ----------------------------------
# 6. SAVE MODEL + METADATA
# ----------------------------------
joblib.dump(model, os.path.join(MODELS_DIR, "task2_sales_forecasting_model_v4.pkl"))

encoders = {
    "cust_means": cust_avg,
    "prod_means": prod_avg,
    "geo_means": geo_avg,
    "cust_prod_means": cust_prod_avg,
    "recent_means": (
        monthly.groupby(["CUSTOMER", "PRODUCT_GROUP"])["RECENT_AVG"]
        .last()
        .to_dict()
    )
}

joblib.dump(encoders, os.path.join(MODELS_DIR, "task2_encoders_v4.pkl"))

metadata = {
    "customers": sorted(monthly["CUSTOMER"].unique()),
    "products": sorted(monthly["PRODUCT_GROUP"].unique()),
    "regions": sorted(monthly["REGION"].unique())
}

joblib.dump(metadata, os.path.join(MODELS_DIR, "task2_metadata_v4.pkl"))

# ----------------------------------
# 7. SAVE HISTORY FOR STREAMLIT
# ----------------------------------
history_df = monthly[["CUSTOMER", "PRODUCT_GROUP", "PLOT_DATE", "SALES"]].copy()
history_df = history_df.rename(columns={"PLOT_DATE": "Plot_Date"})

joblib.dump(history_df, os.path.join(MODELS_DIR, "history_monthly.pkl"))

print("✅ All Model 2 artifacts saved successfully.")
