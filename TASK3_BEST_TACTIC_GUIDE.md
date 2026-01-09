# Task 3: Best Tactic Recommendation - Quick Reference

**Updated:** January 9, 2026  
**Status:** ✅ NEW TASK (Replaces Customer Segmentation)

---

## 📋 What is Task 3?

**Goal:** Find the single **best promotional tactic** for each product group based on historical effectiveness

**Input:** Historical TACTIC plans and claims data  
**Output:** Ranked list of tactics per product, with recommended #1 tactic

**Example Output:**
```
PRODUCT_GROUP     BEST_TACTIC          APPROVAL_RATE   EFFECTIVENESS_SCORE
Ice Cream         BOGO                 92.5%           8.7/10
Oils              Trade Display        87.3%           8.2/10
Biscuits          Fixed Rebate         89.1%           8.4/10
```

---

## 🚀 HOW TO RUN

### Command
```bash
python -m scripts.training.task3_best_tactic_classifier
```

### From Different Locations
```bash
# If in project root
python -m scripts.training.task3_best_tactic_classifier

# If in project directory
cd d:\BlueRose\iffco_discount_model
python -m scripts.training.task3_best_tactic_classifier
```

---

## 📊 INPUTS REQUIRED

Place these files in `data/csv/`:

| File Pattern | Description | Required? |
|--------------|-------------|-----------|
| `TACTIC_*_TACTIC_PLAN_DIMNS*.csv` | Plan metadata (what tactics were planned) | ✅ YES |
| `TACTIC_*_TACTIC_CM_CLAIMS_DIMNS*.csv` | Claim results (what was approved/amount) | ✅ YES |
| `TACTIC_*_CLAIMS_DIMNS*.csv` | Alternative claims file naming | ✅ YES (if above not found) |
| `TDA_*_SIP_PLAN*.csv` | TDA plan data (enhanced coverage) | ⚠️ OPTIONAL |
| `TDA_*_SIP_CLAIM*.csv` | TDA claim data (enhanced coverage) | ⚠️ OPTIONAL |

**Example filenames that work:**
- `TACTIC_24_25_TACTIC_PLAN_DIMNS.csv`
- `TACTIC_24_25_TACTIC_CM_CLAIMS_DIMNS.csv`
- `TACTIC_23_24_CLAIMS_DIMNS.csv`

---

## 📤 OUTPUTS GENERATED

In `results/` folder:

| File | Purpose | Use Case |
|------|---------|----------|
| **task3_best_tactic_per_product.csv** | One row per product with recommended tactic | Business recommendations |
| **task3_tactic_scores_full.csv** | All tactics ranked per product | Detailed analysis |
| **task3_best_tactic_per_product_model.csv** | Model-based predictions | ML-driven insights |
| **task3_tactic_scores_model.csv** | Model scores for all tactics | Full model output |

### Sample Output: task3_best_tactic_per_product.csv
```
PRODUCT_GROUP,BEST_TACTIC,APPROVAL_RATE,AVG_SCORE,SAMPLE_SIZE,CONFIDENCE
Ice Cream,BOGO,0.925,8.7,245,HIGH
Oils,Trade Display Allowance,0.873,8.2,189,HIGH
Biscuits,Fixed Rebate,0.891,8.4,156,MEDIUM
Soaps,OSA Rental,0.856,7.9,98,MEDIUM
```

---

## ⚙️ HOW IT WORKS

### Algorithm Steps
1. **Load Plans**: Read TACTIC plan files
   - Extract PRODUCT_GROUP, TACTIC_TYPE, plan dates
   
2. **Load Claims**: Read TACTIC claim files
   - Extract actual approved amounts, status
   
3. **Join**: Match claims to plans via TACTIC_PLAN key
   - Handles 1:Many relationships (multiple claims per plan)
   - Deduplicates to claim-level
   
4. **Score Each Tactic**: For every (PRODUCT_GROUP, TACTIC_TYPE) pair
   - **Approval Rate** = % of plans that resulted in approved claims
   - **Effectiveness Score** = Average normalized claim amount
   - **Count** = Number of historical records
   
5. **Rank**: Sort tactics by combined score for each product
   
6. **Output**: Export top-1 (best) + full rankings

---

## 🔍 WHAT TO CHECK IN OUTPUTS

### task3_best_tactic_per_product.csv

✅ **Good indicators:**
- All major products have a recommended tactic
- APPROVAL_RATE > 70% (tactics are generally successful)
- Variety of tactics across products (BOGO, TDA, Fixed Rebate, etc.)
- SAMPLE_SIZE > 50 (enough historical data)

❌ **Red flags:**
- Empty file or very few rows
- All products recommend same tactic (indicates data quality issue)
- APPROVAL_RATE < 50% (tactics aren't effective)
- SAMPLE_SIZE < 10 (insufficient historical data)

### task3_tactic_scores_full.csv

✅ **Good indicators:**
- Each product has multiple tactics listed (3-5 per product)
- Scores vary between tactics (not all the same)
- Relative ranking makes business sense

❌ **Red flags:**
- Only 1 tactic per product
- All scores identical
- NaN or null values

---

## 🆘 TROUBLESHOOTING

### Problem: "No plan files matched"
**Cause:** TACTIC plan files not found  
**Solution:**
```bash
# Check files exist:
ls data/csv/TACTIC_*PLAN*.csv

# If not found, you may need:
- Copy TACTIC_*_TACTIC_PLAN_DIMNS*.csv to data/csv/
```

### Problem: Output files empty
**Cause:** No matching claims data or join failed  
**Solution:**
```bash
# Verify claims files exist:
ls data/csv/TACTIC_*CLAIMS*.csv

# Check column names match (TACTIC_PLAN, PRODUCT_GROUP, TACTIC_TYPE)
python analyze_data_files.py
```

### Problem: "Merge did not produce any rows"
**Cause:** Plans and claims don't join properly  
**Solution:**
```bash
1. Check join key exists in both files: TACTIC_PLAN
2. Check column names: python debug_cols.py
3. Run: python analyze_tda_join_keys.py (to understand join logic)
```

### Problem: File not found error when running command
**Cause:** Script path incorrect  
**Solution:**
```bash
# Make sure you're in project root:
cd d:\BlueRose\iffco_discount_model

# Then run:
python -m scripts.training.task3_best_tactic_classifier

# Or use direct path:
python scripts/training/task3_best_tactic_classifier.py
```

---

## 📈 INTEGRATION WITH DASHBOARD

The dashboard (`app.py`) displays Task 3 results:

1. **Tab 3: TACTIC Recommendation**
2. **Shows:**
   - Best tactic for selected product
   - Top N tactics by effectiveness
   - Approval rates and scores
   - Download options for full data

**Files loaded by app.py:**
- `results/task3_best_tactic_per_product.csv`
- `results/task3_tactic_scores_full.csv`
- `results/task3_best_tactic_per_product_model.csv`
- `results/task3_tactic_scores_model.csv`

---

## 🔄 WORKFLOW INTEGRATION

### Complete Project Flow

```
1. Train Task 1: python task1_discount_pipeline.py
2. Train Task 2: python train_task2_corrected.py
3. Train Task 3: python -m scripts.training.task3_best_tactic_classifier ← NEW
4. Validate: python validate_task2_2025.py
5. Forecast: python task2_forecast_2026.py
6. Dashboard: streamlit run app.py
```

### Task 3 Standalone

If you only need Task 3 results (without Tasks 1 & 2):

```bash
# Just run Task 3
python -m scripts.training.task3_best_tactic_classifier

# Check results
cat results/task3_best_tactic_per_product.csv

# View in dashboard (if Tasks 1 & 2 also completed):
streamlit run app.py
```

---

## 📊 DATA REQUIREMENTS

### Minimum Data
- At least 10 historical TACTIC plans
- At least 1 product group
- Matching claims for those plans

### Recommended Data
- 100+ historical TACTIC plans (FY 2023-2025)
- 5+ product groups
- 50%+ of plans have matching claims
- Multiple tactics per product group

### Current Data
- Plans: TACTIC_24_25_TACTIC_PLAN_DIMNS.csv (~1,000+ rows)
- Claims: TACTIC_24_25_TACTIC_CM_CLAIMS_DIMNS.csv (~2,200 rows)
- Optional: TDA_SIP files (~57,000 rows)

---

## 🎯 EXPECTED RESULTS

### For Ice Cream Product
```
Ranking:
  1. BOGO              [92.5% approval, $8.7M avg]  ← RECOMMENDED
  2. Trade Display     [87.3% approval, $6.2M avg]
  3. OSA Rental        [76.4% approval, $4.5M avg]
  4. Fixed Rebate      [69.8% approval, $3.1M avg]
```

### Business Interpretation
- **BOGO (Buy-One-Get-One)** is most effective for Ice Cream
- Has highest approval rate (92.5%)
- Generates largest average claim amount
- Recommended for future Ice Cream campaigns

---

## 📝 SUMMARY

| Item | Details |
|------|---------|
| **What** | Find best tactic per product group |
| **How** | Score all tactics by approval rate & effectiveness |
| **Input** | TACTIC plan and claim CSV files |
| **Output** | Best tactics ranked per product |
| **Run** | `python -m scripts.training.task3_best_tactic_classifier` |
| **Results** | `results/task3_*.csv` files |
| **Time** | ~30-60 seconds |
| **Status** | ✅ Production ready |

---

**Last Updated:** January 9, 2026
