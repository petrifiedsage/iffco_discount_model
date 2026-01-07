# IFFCO Trade Strategy Engine - Folder Structure

## 📁 Root Files (Production)
- `app.py` - Main Streamlit application
- `task1_discount_pipeline.py` - Task 1: Discount optimization training
- `train_task2_corrected.py` - Task 2: Sales forecasting training (PRODUCTION)
- `task3_segmentation.py` - Task 3: Customer segmentation training
- `app_task2_inference_corrected.py` - Task 2 inference module
- `requirements.txt` - Python dependencies
- `README.md` - Project documentation

## 📂 Directory Structure

### `/models/`
Trained model files (*.pkl, *.csv)
- `task2_reboot_model.pkl` - Main forecasting model
- `task2_reboot_metada0ta.pkl` - Model metadata
- `task2_reboot_states.pkl` - Customer-product 0
- `history_monthly.pkl` - Historical sales data
- `product_seasonality_correct.csv` - Seasonality patterns
- Task 1 and Task 3 models

### `/data/`
- `/csv/` - Raw Contract_Claim_Dump files (247 CSVs, 2022-2025)
- Training data sources

### `/scripts/`
- `/testing/` - Model testing and validation scripts
  - `test_model_performance.py` - Comprehensive model evaluation
  - `check_*.py` - Data validation scripts
  
- `/exploration/` - Data analysis and exploration
  - `explore_*.py` - Data structure analysis
  - `analyze_*.py` - Data quality checks
  - `extract_*.py` - Feature extraction
  - `clean_*.py` - Data cleaning utilities
  
- `/training/` - Future training scripts

### `/docs/`
- Project documentation and analysis reports
- Training logs and summaries
- Technical documentation

### `/archive/`
- `/old_training/` - Deprecated training scripts (v3, v7, etc.)
- `/notebooks/` - Jupyter notebooks
- `/data/`, `/models/`, `/outputs/` - Old artifacts
- Archived CSV results

### `/output/`
Generated forecast outputs

### `/results/`
Model evaluation results

## 🚀 Quick Start

**Run the app:**
```bash
streamlit run app.py
```

**Retrain models:**
```bash
# Task 1: Discount Optimization
python task1_discount_pipeline.py

# Task 2: Sales Forecasting (RECOMMENDED)
python train_task2_corrected.py

# Task 3: Customer Segmentation
python task3_segmentation.py
```

**Test model performance:**
```bash
python scripts\testing\test_model_performance.py
```

## 📊 Current Model Performance (Task 2)

- **Test Median APE:** 47.5%
- **R² Score:** 0.839
- **MAE:** $836 per transaction
- **Revenue Accuracy:** 62.7%
- **Data:** 1.97M transactions, 3,939 customers, 28 products
- **Source:** Contract_Claim_Dump files (2022-2025)

## 📝 Notes

- All old/experimental scripts moved to `/archive/old_training/`
- Production code kept in root for easy access
- Test scripts organized in `/scripts/testing/`
- Data exploration tools in `/scripts/exploration/`
