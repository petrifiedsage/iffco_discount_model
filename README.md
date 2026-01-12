# IFFCO Trade Strategy Engine

A comprehensive predictive analytics platform for discount optimization, sales forecasting, and tactic recommendation. This system leverages historical TACTIC data to provide actionable insights for trade strategy decisions.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Tasks](#tasks)
- [Project Structure](#project-structure)
- [Dependencies](#dependencies)
- [Datasets](#datasets)
- [Installation](#installation)
- [Usage](#usage)
- [Model Artifacts](#model-artifacts)
- [Outputs](#outputs)

---

## 🎯 Overview

This platform consists of three core analytics tasks:

1. **Task 1**: Discount/Rebate Optimization
2. **Task 2**: Sales Forecasting (2025 validation + 2026 predictions)
3. **Task 3**: Tactic Effectiveness Ranking & Recommendation

The entire system is accessible via a **Streamlit web application** (`app.py`) that provides interactive dashboards for each task.

---

## 🔧 Tasks

### Task 1: Discount Optimizer
**Goal**: Predict optimal discount percentages for customer-product combinations with business guardrails.

**Approach**:
- Train RandomForest regressor on historical contract/rebate data
- Apply business rules (channel/product caps)
- Output safe, explainable discount recommendations

**Business Guardrails**:
- GT channel: capped at 10%
- LOC_MT + Ice creams: capped at 15%
- Bakery: capped at 12.5%

**Models Used**:
- `task1_discount_regressor_v3.pkl`
- `task1_metadata_v3.pkl`

---

### Task 2: Sales Forecasting
**Goal**: Predict monthly sales per customer-product using time series modeling with seasonality.

**Approach**:
- LightGBM Tweedie regressor with lag features (1, 2, 3, 6, 12 months)
- Product-level seasonality adjustments
- Customer-product state tracking for inference
- Retrained on 2022-2024 data, validated on 2025, forecasting 2026

**Features**:
- On-demand forecast (1-12 months ahead)
- 2025 actual vs predicted validation (Median APE: ~12.35%)
- 2026 seasonal forecasts (Winter, Spring, Summer, Fall, Festival)

**Models Used**:
- `task2_reboot_model.pkl` (LightGBM model)
- `task2_reboot_metadata.pkl` (feature metadata)
- `task2_reboot_states.pkl` (time series states)
- `history_monthly.pkl` (historical data for plotting)
- `product_seasonality_correct.csv` (seasonality factors)

---

### Task 3: Tactic Recommendation
**Goal**: Rank and recommend the best `TACTIC_TYPE` per `PRODUCT_GROUP` for 2026.

**Approach**:
1. **Heuristic Scoring**: `score = approval_rate × log1p(claim_count) × log1p(total_claim_amount)`
2. **ML Model**: Logistic classifier predicts approval probability; computes `pred_expected_score`
3. **2026 Weighting**: Optionally blends score with 2026 sales forecast volume

**Data Sources** (TACTIC files only):
- TACTIC plans: `TACTIC_*_TACTIC_PLAN_DIMNS*.csv`
- TACTIC claims: `TACTIC_*_CLAIMS_DIMNS*.csv`, `TACTIC_*_TACTIC_CM_CLAIMS_DIMNS*.csv`

**Models/Outputs**:
- `task3_best_tactic_per_product_model.csv` (top-1 per product, model-ranked)
- `task3_tactic_scores_model.csv` (all tactics scored)
- `task3_2026_tactic_recommendations.csv` (2026 predictions)
- `task3_model_metrics.json` (classifier metrics)

---

## 📁 Project Structure

```
iffco_discount_model/
├── app.py                              # Streamlit web application (main entry point)
├── app_task2_inference_corrected.py    # Task 2 inference helper
├── requirements.txt                    # Python dependencies
├── README.md                           # This file
│
├── data/                               # Raw datasets (ignored by git)
│   └── csv/
│       ├── TACTIC_*_TACTIC_PLAN_DIMNS*.csv
│       ├── TACTIC_*_CLAIMS_DIMNS*.csv
│       └── (other raw TACTIC/TDA files)
│
├── models/                             # Trained models & artifacts
│   ├── task1_discount_regressor_v3.pkl
│   ├── task1_metadata_v3.pkl
│   ├── task2_reboot_model.pkl
│   ├── task2_reboot_metadata.pkl
│   ├── task2_reboot_states.pkl
│   ├── history_monthly.pkl
│   ├── product_seasonality_correct.csv
│   └── customer_segments.csv
│
├── output/                             # Task 2 forecasts & validation
│   ├── task2_2025_validation_retrained.csv
│   ├── task2_2026_forecast_retrained.csv
│   └── task2_improved_v4_ensemble_predictions.csv
│
├── results/                            # Task 3 outputs
│   ├── task3_best_tactic_per_product_model.csv
│   ├── task3_tactic_scores_model.csv
│   ├── task3_2026_tactic_recommendations.csv
│   ├── task3_2026_tactic_scores.csv
│   └── task3_model_metrics.json
│
├── scripts/                            # Training & prediction scripts
│   ├── __init__.py
│   ├── predict_task3_2026.py          # Task 3: 2026 tactic predictor
│   └── training/
│       ├── __init__.py
│       └── task3_best_tactic_classifier.py
│
└── docs/                               # Documentation & analysis
    ├── BRIEFING.md
    ├── EXECUTION_FLOW_GUIDE.md
    └── (other analysis docs)
```

---

## 📦 Dependencies

### Core Packages
```
pandas>=2.1.0
numpy>=1.24.0
scikit-learn>=1.2.0
joblib>=1.3.0
streamlit>=1.30.0
```

### Visualization
```
plotly>=5.0.0
matplotlib>=3.0.0
seaborn>=0.11.0
```

### Data I/O
```
openpyxl>=3.1.0
xlrd==1.2.0
```

### Installation
```bash
pip install -r requirements.txt
```

---

## 📊 Datasets

### Task 1: Discount Optimizer
- **Source**: Contract/rebate historical data
- **Format**: Excel (`.xls`, `.xlsx`) or CSV
- **Location**: `data/contracts.xls`

### Task 2: Sales Forecasting
- **Source**: Contract claim dump with sales transactions
- **Features**: Customer, Product Group, Date, Sales Amount
- **Training**: 2022-2024
- **Validation**: 2025
- **Forecast**: 2026

### Task 3: Tactic Recommendation
**TACTIC Files** (placed in `data/csv/`):
- **Plans**: `TACTIC_*_TACTIC_PLAN_DIMNS*.csv`
  - Columns: `TACTIC_PLAN`, `PRODUCT_GROUP`, `TACTIC_TYPE`, `CHANNEL`, `COUNTRY`, `DEAL_STATUS`, etc.
- **Claims**: `TACTIC_*_CLAIMS_DIMNS*.csv`, `TACTIC_*_TACTIC_CM_CLAIMS_DIMNS*.csv`
  - Columns: `TACTIC_PLAN`, `CLAIM_AMOUNT`, `CLAIM_STATUS`, `TACTIC_NAME`

**Note**: Raw `data/` folder is excluded from git. Only processed models/outputs are tracked.

---

## 🚀 Usage

### 1. Run Streamlit App (Recommended)

Launch the interactive web application:

```powershell
streamlit run app.py
```

**Features**:
- **Tab 1**: Discount optimizer with business guardrails
- **Tab 2**: On-demand sales forecast, 2025 validation, 2026 forecast by season
- **Tab 3**: Top tactic recommendations per product

**Access**: Open browser at `http://localhost:8501`

---

### 2. Train Task 3 Model (TACTIC Classifier)

Train the approval classifier and generate tactic scores:

```powershell
python scripts/training/task3_best_tactic_classifier.py
```

**Outputs**:
- `results/task3_best_tactic_per_product_model.csv`
- `results/task3_tactic_scores_model.csv`
- `results/task3_model_metrics.json`

---

### 3. Generate 2026 Tactic Predictions

Predict best tactics for 2026 with optional sales forecast weighting:

```powershell
python scripts/predict_task3_2026.py
```

**Outputs**:
- `results/task3_2026_tactic_recommendations.csv` (top-1 per product)
- `results/task3_2026_tactic_scores.csv` (all tactics scored)

**Configuration**:
- Set `WEIGHT_BY_FORECAST = True` in script to blend with 2026 sales volume
- Requires `output/task2_2026_forecast_retrained.csv`

---

## 📦 Model Artifacts

All models are pre-trained and stored in `models/` and `output/` directories:

| File | Task | Description |
|------|------|-------------|
| `task1_discount_regressor_v3.pkl` | 1 | Discount prediction model |
| `task1_metadata_v3.pkl` | 1 | Feature metadata |
| `task2_reboot_model.pkl` | 2 | Sales forecast model (LightGBM) |
| `task2_reboot_metadata.pkl` | 2 | Feature names, products, customers |
| `task2_reboot_states.pkl` | 2 | Time series states per customer-product |
| `history_monthly.pkl` | 2 | Historical sales for plotting |
| `product_seasonality_correct.csv` | 2 | Seasonality factors |
| `customer_segments.csv` | 3 | Customer segmentation (optional) |

---

## 📄 Outputs

### Task 1
- `outputs/rebate_recommendations.csv`: Predicted discount % per customer-product with guardrails applied

### Task 2
- `output/task2_2025_validation_retrained.csv`: 2025 actual vs predicted sales (APE metrics)
- `output/task2_2026_forecast_retrained.csv`: 2026 forecast by product & season (Winter, Spring, Summer, Fall, Festival)

### Task 3
- `results/task3_best_tactic_per_product_model.csv`: Top-1 tactic per product (model-ranked)
- `results/task3_tactic_scores_model.csv`: All tactics with approval probability & expected score
- `results/task3_2026_tactic_recommendations.csv`: 2026 predictions (optionally volume-weighted)
- `results/task3_model_metrics.json`: Logistic classifier performance (accuracy, F1, ROC-AUC)

---

## 🔍 Key Features

- **Interactive UI**: Streamlit dashboards for all tasks
- **Business Rules**: Discount caps enforced per channel/product
- **Time Series**: Lag features + seasonality for accurate forecasting
- **Approval Prediction**: ML model estimates tactic approval probability
- **Volume Weighting**: 2026 recommendations prioritize high-volume products
- **AED Currency**: All amounts displayed in AED format

---

## 📝 Notes

- **Data Privacy**: Raw `data/` folder is gitignored; only models/outputs are tracked
- **Python Version**: Tested on Python 3.11+
- **Memory**: Large datasets use sparse encoding and chunked processing
- **Extensibility**: Modular scripts allow easy customization of features, models, and business rules

---

## 📧 Support

For issues or questions, refer to the documentation in `docs/` or review the analysis notebooks.

---

**Last Updated**: January 2026
