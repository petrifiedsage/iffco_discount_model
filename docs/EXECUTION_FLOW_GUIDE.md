# IFFCO Discount Model - Complete Execution Flow Guide

**Date:** January 9, 2026  
**Purpose:** Step-by-step guide to run the project on a new system and get desired outputs

---

## 📋 PREREQUISITES

### 1. **Folder Structure Setup**
Ensure this structure exists:
```
iffco_discount_model/
├── data/
│   ├── csv/                    (← Place Contract_Claim_Dump*.csv files here)
│   └── [Other data files]
├── models/                     (← Created by training scripts)
├── output/                     (← Created by inference scripts)
├── *.py                        (All Python scripts)
├── requirements.txt
└── venv/                       (Optional: virtual environment)
```

### 2. **Install Dependencies**
```bash
cd d:\BlueRose\iffco_discount_model
pip install -r requirements.txt
```

**Key packages needed:**
- pandas, numpy, scikit-learn
- lightgbm, xgboost
- streamlit (for dashboard)
- joblib (for model saving)
- plotly (for visualizations)

### 3. **Prepare Data Files**
Place these in `data/csv/`:
- **Contract_Claim_Dump*.csv** (247 CSV files, 2022-2025)
  - Source: Siebel exports
  - Required columns: CUSTOMER_NAME, PRODUCT_GROUP, AMOUNT, STATUS, CLAIM_DATE, MONTH

Place these in `data/`:
- **Contract_Data_with_Product Group_IDS_OU_2025.xls** (or .xlsx)
  - Used for Task 1 (Discount Pipeline)

---

## 🚀 EXECUTION FLOW - STEP BY STEP

### **PHASE 1: INITIAL DATA VALIDATION** (Optional but Recommended)
This helps you verify data is correct before training.

#### Step 1a: Analyze Data Files
```bash
python analyze_data_files.py
```
**Output:** Console report of available features  
**Check:** Verify required columns exist  
**Expected:** Lists which CSV files have which features

#### Step 1b: Data Quality Check
```bash
python analyze_data_quality.py
```
**Output:** Console report of data quality  
**Check:** Nulls, duplicates, data types  
**Expected:** Shows data distribution and quality metrics

#### Step 1c: Check Join Relationships (Optional)
```bash
python analyze_join_relationships.py
```
**Output:** Console report of join cardinality  
**Check:** How records relate across files  
**Expected:** Shows 1:1, 1:Many relationships

#### Step 1d: Check TDA Join Keys (Optional)
```bash
python analyze_tda_join_keys.py
```
**Output:** Console report of TDA join cardinality  
**Check:** TDA plan and claim join relationships  
**Expected:** Shows proper join cardinality for TDA files

---

### **PHASE 2: TRAINING (One-Time Setup)**
Run these once to create models in `/models/` folder.

#### Step 2a: Train Task 1 - Discount Optimizer
```bash
python task1_discount_pipeline.py
```

**Expected Output:**
- ✅ `models/task1_discount_regressor_v3.pkl`
- ✅ `models/task1_metadata_v3.pkl`

**What it does:**
- Loads contract data
- Trains RandomForest discount classifier
- Saves model + metadata

**Time:** ~30-60 seconds  
**Prerequisites:** Contract data file exists

---

#### Step 2b: Train Task 2 - Sales Forecasting (CRITICAL)
```bash
python train_task2_corrected.py
```

**Expected Output:**
- ✅ `models/task2_reboot_model.pkl` (main model)
- ✅ `models/task2_reboot_metadata.pkl`
- ✅ `models/task2_reboot_states.pkl` (customer-product states)
- ✅ `models/history_monthly.pkl` (monthly history)
- ✅ `models/product_seasonality_correct.csv`

**What it does:**
- Loads all Contract_Claim_Dump*.csv files
- Aggregates by Customer + Product + Month
- Trains HistGradientBoosting + LightGBM ensemble
- Extracts seasonality patterns
- Evaluates on 2025 holdout

**Time:** 2-5 minutes  
**Prerequisites:** All CSV files in `data/csv/`  
**Monitor:** Watch console for training progress and final metrics (R2, MAE, APE)

---

#### Step 2c: Train Task 3 - Best Tactic Recommendation per Product
```bash
python -m scripts.training.task3_best_tactic_classifier
```

**Expected Output:**
- ✅ `results/task3_best_tactic_per_product.csv` (one row per product with recommended tactic)
- ✅ `results/task3_tactic_scores_full.csv` (all tactic scores per product)
- ✅ `results/task3_best_tactic_per_product_model.csv` (model predictions)
- ✅ `results/task3_tactic_scores_model.csv` (model scores for all tactics)

**What it does:**
- Loads TACTIC_PLAN and TACTIC_CLAIMS files from data/csv/
- Joins claims to plans via TACTIC_PLAN
- For each PRODUCT_GROUP, scores each TACTIC_TYPE by:
  - Approval rate (% of claims approved)
  - Average effectiveness score
- Recommends the single best tactic per product
- Also generates full rankings for reference

**Data Sources:**
- `data/csv/TACTIC_*_TACTIC_PLAN_DIMNS*.csv` (plan metadata)
- `data/csv/TACTIC_*_TACTIC_CM_CLAIMS_DIMNS*.csv` (actual claims)
- Optional: TDA SIP files for enhanced coverage

**Time:** ~30-60 seconds  
**Prerequisites:** TACTIC CSV files must exist in data/csv/

---

### **PHASE 3: VALIDATION & ANALYSIS** (Verify Model Quality)
After training Task 1 & 2, validate that models work correctly. Task 3 is self-contained and doesn't require validation.

#### Step 3a: Validate Task 2 Model Performance
```bash
python validate_task2_2025.py
```

**Expected Output:**
- Console report: R2, MAE, RMSE, Median APE on 2025 data
- Shows: "Train 2022-2024, Predict 2025 vs Actual"

**Check:** 
- R2 > 0.60 (good), > 0.80 (excellent)
- Median APE < 25% (good)
- No errors in output

**Example output:**
```
TASK 2: VALIDATION - FORECAST 2025 vs ACTUAL 2025
========================================
Overall Metrics:
  R² Score:      0.762
  MAE:           1,234.56
  RMSE:          2,456.78
  Median APE:    18.3%
Product-wise performance:
  Ice Cream      R²=0.81, APE=16.2%
  Oils           R²=0.75, APE=19.4%
  ...
```

---

#### Step 3b: Detailed Evaluation
```bash
python evaluate_task2_2025.py
```

**Expected Output:**
- `output/task2_2025_pred_vs_actual.csv` (monthly predictions)
- `output/task2_2025_summary_by_product.csv` (product aggregates)
- Console: Detailed metrics by product

**Check:**
- Files created successfully
- Open CSV to verify predictions look reasonable
- Compare pred vs actual columns

---

### **PHASE 4: GENERATE FORECASTS & REPORTS** (Main Outputs)
Create the actual forecast and analysis outputs.

#### Step 4a: Forecast 2026 Sales
```bash
python task2_forecast_2026.py
```

**Expected Output:**
- ✅ `output/task2_forecast_2026_monthly.csv` (monthly granularity)
- ✅ `output/task2_forecast_2026_quarterly.csv` (quarterly aggregates)

**What it does:**
- Uses trained model to predict Jan-Dec 2026
- Generates for every customer-product combination
- Aggregates to quarterly (Q1-Q4)

**Check:** Open CSVs and verify:
- All products have forecasts
- Values are positive and reasonable
- Date range is Jan-Dec 2026

**Example output structure:**
```
CUSTOMER_NAME, PRODUCT_GROUP, MONTH, PREDICTED_SALES
Walmart,       Ice Cream,      2026-01, 15234.56
Walmart,       Ice Cream,      2026-02, 16789.12
...
```

---

#### Step 4b: Aggregate to Quarterly Claims
```bash
python task2_quarterly_claims.py
```

**Expected Output:**
- ✅ `output/quarterly_claims_by_product.csv` (all years combined)
- ✅ `output/quarterly_claims_by_product_year.csv` (per-year breakdown)

**What it does:**
- Reads Contract_Claim_Dump files
- Groups by Product + Quarter
- Sums amounts for each quarter

**Check:** Open CSV and verify:
- Columns: PRODUCT_GROUP, Q1, Q2, Q3, Q4
- Values for 2022-2025 years
- Reasonable quarterly sums

---

#### Step 4c: Compare 2025 Actual vs 2026 Forecast
```bash
python compare_2025_2026.py
```

**Expected Output:**
- Console report: Side-by-side 2025 vs 2026 comparison
- Shows: Quarterly trends and year-over-year changes

**Check:** 
- Forecast shows logical growth/decline
- All products have comparisons
- No NaN values

**Example output:**
```
PRODUCT_GROUP    2025-Q1  2025-Q2  2026-Q1  2026-Q2  Growth
Ice Cream        45000    78000    48000    82000    +6.7%
Oils             120000   118000   125000   123000   +4.2%
```

---

### **PHASE 5: LAUNCH DASHBOARD** (Final Product)
Run the interactive web application.

#### Step 5: Run Streamlit App
```bash
streamlit run app.py
```

**Expected Output:**
- Browser opens to: `http://localhost:8501`
- Dashboard shows 3 tabs:
  1. 💰 Task 1: Discount Optimizer
  2. 📈 Task 2: Sales Forecast
  3. 🧩 Task 3: TACTIC Recommendation

**Check:**
- All models load (✅ Models Loaded message)
- Tabs are interactive
- Can input data and get predictions
- No errors in console

---

## 📊 QUICK REFERENCE: What Each Phase Produces

| Phase | Scripts | Output | Purpose |
|-------|---------|--------|---------|
| **1** | analyze_*.py | Console reports | Validate data quality |
| **2** | train_task1, train_task2, task3_best_tactic | `models/*.pkl`, `results/*.csv` | Build ML models & find best tactics |
| **3** | validate_*.py, evaluate_*.py | CSV files + metrics | Verify model quality (Task 1 & 2 only) |
| **4** | task2_forecast_*.py, task2_quarterly_*.py, compare_*.py | `output/*.csv` | Generate forecasts |
| **5** | app.py | Interactive dashboard | View/use predictions |

---

## 🔴 TROUBLESHOOTING - If Something Goes Wrong

### Problem: "File not found" errors
**Solution:**
```bash
1. Check data/csv/ has Contract_Claim_Dump*.csv files
2. Check data/ has Contract_Data_with_Product Group_IDS_OU_2025.xls
3. Verify file paths in scripts match your system
```

### Problem: Training takes forever or crashes
**Solution:**
```bash
1. Reduce data: Delete some Contract_Claim_Dump*.csv files to test
2. Check RAM: Need ~4-8 GB for full dataset
3. Check pandas version: pip install --upgrade pandas
```

### Problem: Models not loading in app.py
**Solution:**
```bash
1. Run training scripts first (Phase 2)
2. Check models/ folder has all .pkl files
3. Run validation scripts to confirm models exist
```

### Problem: Different outputs than expected
**Solution:**
```bash
1. Verify data is exactly same format
2. Check Python/package versions match
3. Re-run train_task2_corrected.py to retrain
4. Check task2_forecast_2026.py uses correct model paths
```

### Problem: CSV outputs are empty or missing columns
**Solution:**
```bash
1. Check evaluate_task2_2025.py runs without errors
2. Verify data has required columns
3. Run debug_cols.py to check column existence
```

---

## ✅ SUCCESS CHECKLIST

After running all phases, verify:

- [ ] Phase 1: Data validation reports show no critical errors
- [ ] Phase 2: All model files created in `models/`
  - [ ] task1_discount_regressor_v3.pkl
  - [ ] task2_reboot_model.pkl
  - [ ] task2_reboot_states.pkl
- [ ] Phase 2: Task 3 outputs created in `results/`
  - [ ] task3_best_tactic_per_product.csv
  - [ ] task3_tactic_scores_full.csv
- [ ] Phase 3: Validation shows good metrics (R² > 0.60, APE < 25%)
- [ ] Phase 4: Output CSVs exist and contain forecasts
  - [ ] task2_forecast_2026_monthly.csv
  - [ ] task2_forecast_2026_quarterly.csv
  - [ ] quarterly_claims_by_product.csv
- [ ] Phase 5: Dashboard loads without errors
  - [ ] "✅ Models Loaded" appears in sidebar
  - [ ] All 3 tabs are interactive

---

## 🎯 MINIMAL QUICK START (If You're in a Hurry)

If you want to skip validation and just get to outputs:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train models (one time)
python task1_discount_pipeline.py
python train_task2_corrected.py
python -m scripts.training.task3_best_tactic_classifier

# 3. Generate outputs
python task2_forecast_2026.py
python task2_quarterly_claims.py

# 4. Launch dashboard
streamlit run app.py
```

**Total time:** 5-10 minutes (mostly Task 2 training time)

---

## 🔄 REGULAR OPERATION (After Initial Setup)

Once models are trained, subsequent operations:

```bash
# Just run validation/analysis
python validate_task2_2025.py
python evaluate_task2_2025.py
python compare_2025_2026.py

# Or launch dashboard directly
streamlit run app.py
```

**Time:** < 1 minute

---

## 📝 EXPECTED FINAL OUTPUTS

After successful execution:

```
results/
├── task3_best_tactic_per_product.csv         ← Recommended tactic per product
├── task3_tactic_scores_full.csv              ← All tactic scores per product
├── task3_best_tactic_per_product_model.csv   ← Model predictions
└── task3_tactic_scores_model.csv             ← Model scores for all tactics

output/
├── task2_forecast_2026_monthly.csv           ← Monthly predictions for 2026
├── task2_forecast_2026_quarterly.csv         ← Quarterly summary 2026
├── quarterly_claims_by_product.csv           ← Historical by quarter
├── quarterly_claims_by_product_year.csv      ← By quarter + year
├── task2_2025_pred_vs_actual.csv             ← Validation results
└── task2_2025_summary_by_product.csv         ← Product-level validation

models/
├── task1_discount_regressor_v3.pkl           ← Discount model
├── task1_metadata_v3.pkl
├── task2_reboot_model.pkl                    ← Sales forecast model
├── task2_reboot_metadata.pkl
├── task2_reboot_states.pkl
├── history_monthly.pkl
└── product_seasonality_correct.csv
```

---

## 🆘 NEED HELP?

If you get stuck:

1. **Check data format:** Run `analyze_data_quality.py`
2. **Check model existence:** Verify `models/` folder
3. **Check paths:** Edit script paths to match your system
4. **Check versions:** Ensure Python 3.8+ and package versions match
5. **Run one script at a time:** Don't run multiple scripts simultaneously

---

**Last Updated:** January 9, 2026
