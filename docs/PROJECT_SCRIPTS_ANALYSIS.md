# IFFCO Discount Model - Complete Project Scripts Analysis

**Date:** January 9, 2026  
**Project:** IFFCO Intelligent Trade Planner  
**Purpose:** Comprehensive mapping of all scripts and their usage

---

## 📊 Executive Overview

The project implements **3 integrated ML tasks**:
- **Task 1**: Discount Optimization (Rebate Prediction)
- **Task 2**: Sales Forecasting (Claims Prediction)
- **Task 3**: Customer Segmentation (Tactic Recommendation)

**Total Scripts:** 26 active + 7+ archived  
**Key Output:** Streamlit web application (`app.py`)

---

## 🎯 PRIMARY PRODUCTION SCRIPTS

### Main Application
| Script | Purpose | Status |
|--------|---------|--------|
| **app.py** | Streamlit UI dashboard integrating all 3 tasks | ✅ PRODUCTION |
| **requirements.txt** | Python dependencies | ✅ ACTIVE |

---

## 🔧 TASK 1: Discount Optimization Pipeline

**Objective:** Predict and recommend discounts with business guardrails

### Training & Inference
| Script | Purpose | Status |
|--------|---------|--------|
| **task1_discount_pipeline.py** | Train discount classifier on contract data; save model | ✅ PRODUCTION |

**Details:**
- Loads: `data/Contract_Data_with_Product Group_IDS_OU_2025.xls`
- Features: CUSTOMER, GLOBAL_ACCOUNT, CHANNEL, TYPE, TACTIC_TYPE, PRODUCT_GROUP, START_MONTH_NUM, END_MONTH_NUM
- Model: RandomForestRegressor
- Output: `models/task1_discount_regressor_v3.pkl`, `models/task1_metadata_v3.pkl`
- Loaded by: `app.py`

---

## 📈 TASK 2: Sales Forecasting Pipeline

**Objective:** Forecast claims/sales by customer-product-month

### Training & Model Variants

#### Base/Corrected Model (PRODUCTION)
| Script | Purpose | Status |
|--------|---------|--------|
| **train_task2_corrected.py** | Main training pipeline (2022-2025 data) | ✅ PRODUCTION |

**Details:**
- Loads: `data/csv/Contract_Claim_Dump*.csv` (247 files)
- Data Aggregation: Monthly by Customer + PRODUCT_GROUP
- Model: HistGradientBoostingRegressor + LightGBM ensemble
- Features: Lags, moving averages, seasonality indices
- Output: `models/task2_reboot_model.pkl`, `metadata.pkl`, `states.pkl`, `history_monthly.pkl`
- Loaded by: `app.py`

#### Market-Seasonal Model
| Script | Purpose | Status |
|--------|---------|--------|
| **train_task2_market_seasonal.py** | Alternative model with Market×Season granularity | ⚠️ EXPERIMENTAL |

**Details:**
- Splits data by: MARKET + SEASON (summer, winter, spring, fall)
- Separate models per market-product combination
- Seasonality features engineered per market
- Output: `models/task2_market_seasonal_model.pkl`, `market_product_seasonality.csv`

#### Ensemble Model (Latest)
| Script | Purpose | Status |
|--------|---------|--------|
| **train_task2_improved_v4_ensemble.py** | Advanced ensemble (LightGBM + XGBoost + HistGradientBoosting) | ⚠️ LATEST |

**Details:**
- Combines 3 algorithms with weighted averaging
- Handles bad products/outliers
- Output: `output/task2_improved_v4_ensemble_predictions.csv`

#### Earlier Variants (Development)
| Script | Purpose | Status |
|--------|---------|--------|
| train_task2_improved_v1_exclude_bad_products.py | v1: Remove outlier products | 🔴 ARCHIVED |
| train_task2_improved_v2_market_split.py | v2: Market-based splitting | 🔴 ARCHIVED |
| train_task2_improved_v3_enhanced_lags.py | v3: Enhanced lag features | 🔴 ARCHIVED |

---

### Inference & Validation

| Script | Purpose | Status |
|--------|---------|--------|
| **app_task2_inference_corrected.py** | Inference module for Task 2 predictions | ✅ PRODUCTION |
| **task2_forecast_2026.py** | Generate 2026 monthly/quarterly forecasts (all customer-products) | ✅ PRODUCTION |
| **forecast_2026_market_seasonal.py** | 2026 forecast at Market×Product×Season granularity | ⚠️ EXPERIMENTAL |
| **task2_quarterly_claims.py** | Aggregate claims by product & quarter (Q1-Q4) | ✅ PRODUCTION |
| **validate_task2_2025.py** | Validate model: train 2022-2024, predict 2025 vs actual | ✅ VALIDATION |
| **evaluate_task2_2025.py** | Detailed evaluation on 2025 holdout data | ✅ VALIDATION |
| **compare_2025_2026.py** | Side-by-side comparison of 2025 actual vs 2026 forecast | ✅ ANALYSIS |

---

## 🧩 TASK 3: Best Tactic Recommendation per Product

**Objective:** Identify the most effective promotional tactic for each product group

### Training

| Script | Purpose | Status |
|--------|---------|--------|
| **scripts/training/task3_best_tactic_classifier.py** | Find best tactic per product group | ✅ PRODUCTION |

**Details:**
- Data Sources: TACTIC_PLAN and TACTIC_CLAIMS files from data/csv/
- Joins: Claims to plans via TACTIC_PLAN key
- Scoring: Approval rate × Effectiveness score per tactic per product
- Output: 
  - `results/task3_best_tactic_per_product.csv` (recommended tactic per product)
  - `results/task3_tactic_scores_full.csv` (all tactics ranked per product)
  - `results/task3_best_tactic_per_product_model.csv` (model predictions)
  - `results/task3_tactic_scores_model.csv` (model-based scores)
- Loaded by: `app.py`
- Run: `python -m scripts.training.task3_best_tactic_classifier`

---

## 🔬 DATA ANALYSIS & EXPLORATION SCRIPTS

**Purpose:** Understand data structure, quality, and relationships

| Script | Purpose | Status |
|--------|---------|--------|
| **analyze_data_files.py** | Check data files for required features (TACTIC_TYPE, PRODUCT_GROUP, etc.) | ⚠️ EXPLORATION |
| **analyze_data_quality.py** | Data quality checks: nulls, duplicates, value distributions | ⚠️ EXPLORATION |
| **analyze_join_relationships.py** | Analyze how TACTIC_PLAN, TDA_SIP_PLAN, etc. join with claims | ⚠️ EXPLORATION |
| **analyze_tda_join_keys.py** | Deep dive into TDA join keys and cardinality | ⚠️ EXPLORATION |
| **debug_cols.py** | Column existence and type debugging | ⚠️ DEBUGGING |
| **improvement_summary_report.py** | Summarize improvements across model versions | ⚠️ REPORTING |

---

## 📑 SUPPORTING DOCUMENTATION

| File | Purpose |
|------|---------|
| **README.md** | Quick start guide for Task 1 |
| **BRIEFING.md** | Business context, data sources, organizational structure |
| **STRUCTURE.md** | Project folder and file organization |
| **EXECUTIVE_SUMMARY.md** | Analysis of tactic type prediction features |
| **DATA_FILE_ANALYSIS_REPORT.md** | Detailed data file analysis report |
| **docs/ACTION_PLAN.md** | Implementation roadmap |
| **docs/IMPLEMENTATION_COMPLETE.md** | Completion status |
| **docs/TASK2_REBUILD_SUMMARY.md** | Task 2 rebuild details |
| **docs/training_log.txt** | Training session logs |

---

## 📊 MODEL ARTIFACTS STRUCTURE

### `/models/` Directory
```
├── Task 1 (Discount)
│   ├── task1_discount_regressor_v3.pkl
│   └── task1_metadata_v3.pkl
│
├── Task 2 (Sales Forecasting - PRIMARY)
│   ├── task2_reboot_model.pkl (main model)
│   ├── task2_reboot_metadata.pkl
│   ├── task2_reboot_states.pkl
│   ├── history_monthly.pkl
│   └── product_seasonality_correct.csv
│
├── Task 2 (Market-Seasonal - Alternative)
│   ├── task2_market_seasonal_model.pkl
│   ├── task2_market_seasonal_metadata.pkl
│   ├── history_market_seasonal.pkl
│   └── market_product_seasonality.csv
│
└── Task 3 (Segmentation)
    └── customer_segments.csv
```

---

## 📁 DATA SOURCES

| Location | Contents | Records |
|----------|----------|---------|
| `data/csv/` | Contract_Claim_Dump*.csv (2022-2025) | 247 files, 42,946+ records |
| `data/` | Contract exports, TDA data | Various formats (.xls, .xlsx, .csv) |
| `output/` | Generated predictions and analyses | CSV outputs |

---

## 🔄 SCRIPT DEPENDENCY FLOW

```
app.py (Main Dashboard)
├── Loads: task1_discount_regressor_v3.pkl (Task 1)
├── Loads: task2_reboot_model.pkl (Task 2)
│   └── Generated by: train_task2_corrected.py
│   └── Validation: validate_task2_2025.py, evaluate_task2_2025.py
├── Loads: customer_segments.csv (Task 3)
│   └── Generated by: task3_segmentation.py
└── Loads: Ensemble outputs
    └── Generated by: train_task2_improved_v4_ensemble.py

Forecast & Analysis (Standalone)
├── task2_forecast_2026.py → output/task2_forecast_2026_*.csv
├── task2_quarterly_claims.py → output/quarterly_claims_by_product*.csv
├── forecast_2026_market_seasonal.py → output/forecast_2026_market_seasonal.csv
└── compare_2025_2026.py → Comparison analysis
```

---

## 🎯 SCRIPT USAGE RECOMMENDATIONS

### Always Needed (Core Pipeline)
1. ✅ **app.py** - Main application entry point
2. ✅ **train_task2_corrected.py** - Main model training (Task 2)
3. ✅ **task1_discount_pipeline.py** - Task 1 training
4. ✅ **scripts/training/task3_best_tactic_classifier.py** - Task 3 tactic recommendation
5. ✅ **app_task2_inference_corrected.py** - Inference support

### Validation & Analysis
6. ✅ **validate_task2_2025.py** - Model validation
7. ✅ **evaluate_task2_2025.py** - Detailed evaluation
8. ✅ **task2_forecast_2026.py** - Generate 2026 forecasts
9. ✅ **task2_quarterly_claims.py** - Quarterly aggregations
10. ✅ **compare_2025_2026.py** - Year-over-year analysis

### Experimental/Alternative
11. ⚠️ **train_task2_market_seasonal.py** - Alternative seasonal model
12. ⚠️ **train_task2_improved_v4_ensemble.py** - Latest ensemble approach
13. ⚠️ **forecast_2026_market_seasonal.py** - Granular market forecasts

### Development & Debugging
14. 🔬 **analyze_data_files.py** - Feature validation
15. 🔬 **analyze_data_quality.py** - Data quality assessment
16. 🔬 **analyze_join_relationships.py** - Join logic validation
17. 🔬 **analyze_tda_join_keys.py** - TDA key analysis
18. 🔬 **debug_cols.py** - Column debugging
19. 🔬 **improvement_summary_report.py** - Version comparison

---

## 🚀 QUICK START EXECUTION ORDER

### First Time Setup
```bash
1. Install: pip install -r requirements.txt
2. Data: Place Contract_Claim_Dump*.csv in data/csv/
3. Data: Place TACTIC_*.csv files in data/csv/ (for Task 3)
4. Train Task 1: python task1_discount_pipeline.py
5. Train Task 2: python train_task2_corrected.py
6. Train Task 3: python -m scripts.training.task3_best_tactic_classifier
7. Run App: streamlit run app.py
```

### Regular Operations
```bash
1. Validate: python validate_task2_2025.py
2. Forecast 2026: python task2_forecast_2026.py
3. Generate Reports: python evaluate_task2_2025.py
4. Compare: python compare_2025_2026.py
5. Dashboard: streamlit run app.py
```

### Task 3 Only (Best Tactics)
```bash
1. Update TACTIC files in data/csv/
2. Run: python -m scripts.training.task3_best_tactic_classifier
3. Check: results/task3_best_tactic_per_product.csv
```

---

## 📝 Summary Statistics

| Category | Count |
|----------|-------|
| **Production Scripts** | 5 |
| **Validation/Analysis Scripts** | 7 |
| **Data Exploration Scripts** | 6 |
| **Experimental/Alternative** | 3 |
| **Archived Variants** | 7+ |
| **Documentation Files** | 8 |
| **Total Active Scripts** | 26 |

---

## 🔑 Key Files by Functionality

### If you need to...
- **Build the dashboard**: Use `app.py`
- **Train a new model**: Use `train_task2_corrected.py` (Task 2) or `task1_discount_pipeline.py` (Task 1)
- **Find best tactic per product**: Use `python -m scripts.training.task3_best_tactic_classifier` (Task 3)
- **Make 2026 forecasts**: Use `task2_forecast_2026.py`
- **Validate model accuracy**: Use `validate_task2_2025.py` or `evaluate_task2_2025.py`
- **Check data quality**: Use `analyze_data_*.py` scripts
- **Compare forecasts**: Use `compare_2025_2026.py`
- **Understand project structure**: Read `STRUCTURE.md`, `BRIEFING.md`
- **Check implementation status**: Read `IMPLEMENTATION_COMPLETE.md`

---

**Last Updated:** January 9, 2026
