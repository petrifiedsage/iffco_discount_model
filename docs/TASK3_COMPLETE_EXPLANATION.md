# Task 3: Best Tactic per Product Group - Complete Workflow Explanation

## 🎯 High-Level Purpose

**Task 3** recommends the **best promotional tactic per product group** based on historical claim effectiveness from TACTIC and TDA (Store In-Promotion) data.

**Input:** TACTIC plans + claims + TDA SIP plans + claims  
**Output:** Best tactic per product + full scored rankings  
**Score:** Approval rate × Log(claim volume) × Log(claim amount)

---

## 📊 Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                               │
├─────────────────────────────────────────────────────────────────┤
│  TACTIC Plans              │  TACTIC Claims                      │
│  (TACTIC_PLAN_DIMNS)       │  (TACTIC_CM_CLAIMS_DIMNS)          │
│  - PRODUCT_GROUP           │  - TACTIC_PLAN (FK)                │
│  - TACTIC_TYPE             │  - CLAIM_STATUS                    │
│  - BRAND, CHANNEL, etc.    │  - CLAIM_AMOUNT                    │
│                            │                                    │
│  TDA SIP Plans             │  TDA SIP Claims                     │
│  (TDA_*_SIP_PLAN_DIMNS)    │  (TDA_*_SIP_CLAIM_DIMNS)           │
│  - PRODUCT_GROUP           │  - TACTIC_NUMBER (FK)              │
│  - TACTIC_TYPE             │  - CLAIM_STATUS                    │
│  - BRAND, CHANNEL, etc.    │  - CLAIM_AMOUNT                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │  LOAD & MERGE       │
                    │  (join via keys)    │
                    └─────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│           COMBINED ENRICHED CLAIMS DATASET                      │
│  (merged claims + plan details for each claim)                  │
├─────────────────────────────────────────────────────────────────┤
│  PRODUCT_GROUP | TACTIC_TYPE | CLAIM_STATUS | CLAIM_AMOUNT     │
│  BRAND | CHANNEL | COUNTRY | TERRITORY | DEAL_TYPE | etc.      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │  AGGREGATE & SCORE  │
                    │  (per Product×Tactic)
                    └─────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              HEURISTIC SCORES (ALWAYS GENERATED)               │
│  - approval_rate = (Approved + 0.5×Pending) / Total            │
│  - score = approval_rate × log(claim_count) × log(amount)      │
│  - Pick top-1 per product group                                │
├─────────────────────────────────────────────────────────────────┤
│  Outputs:                                                       │
│  ✓ task3_best_tactic_per_product.csv                           │
│  ✓ task3_tactic_scores_full.csv                                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────────┐
                    │  OPTIONAL: TRAIN    │
                    │  ML MODEL           │
                    │  (if label variety) │
                    └─────────────────────┘
                              ↓
         ┌────────────────────────────────────────┐
         │ MODEL-BASED SCORES (if sufficient data)
         │ - pred_approval_prob (logistic output) │
         │ - pred_expected_score = prob × logs    │
         │ - Rank by pred_expected_score          │
         ├────────────────────────────────────────┤
         │ Outputs:                               │
         │ ✓ task3_best_tactic_per_product_model  │
         │ ✓ task3_tactic_scores_model.csv        │
         │ ✓ task3_model_metrics.json             │
         └────────────────────────────────────────┘
```

---

## 🔧 Key Functions Explained

### 1. **Load Functions** (Data Ingestion)

#### `load_plans() → (DataFrame, int)`
```python
# Loads all TACTIC_*_TACTIC_PLAN_DIMNS.csv files
# Keeps: TACTIC_PLAN, PRODUCT_GROUP, TACTIC_TYPE, BRAND, CHANNEL, etc.
# Filters out: rows missing TACTIC_PLAN, PRODUCT_GROUP, or TACTIC_TYPE
# Returns: deduplicated plans DataFrame
```

**Example row:**
```
TACTIC_PLAN: TP_2024_001
PRODUCT_GROUP: Oils
TACTIC_TYPE: BB_TC_ FLYERS & INSTORE ADVT.
BRAND: Rahma
COUNTRY: KW
```

#### `load_claims() → (DataFrame, int)`
```python
# Loads all TACTIC_*_TACTIC_CM_CLAIMS_DIMNS.csv files
# Keeps: TACTIC_PLAN (foreign key), CLAIM_STATUS, CLAIM_AMOUNT
# Converts CLAIM_AMOUNT to numeric, handles missing values
# Returns: merged claims DataFrame
```

**Example row:**
```
TACTIC_PLAN: TP_2024_001
CLAIM_STATUS: Open
CLAIM_AMOUNT: 5000
```

#### `load_tda_plans()`, `load_tda_claims()`
```python
# Same logic for TDA SIP data
# Uses TACTIC_NUMBER as foreign key instead of TACTIC_PLAN
# Harmonizes column names (CHANNEL_TYPE → CHANNEL, etc.)
```

---

### 2. **Data Enrichment** (Join Plans + Claims)

#### `prepare_enriched_claims(plans, claims) → DataFrame`
```python
# MERGE: LEFT join claims to plans via TACTIC_PLAN
# RECOVER MISSING PRODUCT_GROUP:
#   - Try extracting from TACTIC_NAME
#   - If still missing, label as "UNKNOWN_PRODUCT"
# FILL MISSING TACTIC_TYPE similarly
# FILL categorical columns: PRODUCT_SUB_GROUP, BRAND, etc. → "UNKNOWN"
# FILL numeric columns: AMENDED_DEAL_AMOUNT, ADJUSTMENT_VALUE_AMOUNT → 0.0
# DROP rows still missing PRODUCT_GROUP or TACTIC_TYPE
```

**Result:** Each claim row now has PRODUCT_GROUP and TACTIC_TYPE attached.

#### `combine_data_sources(tactic_merged, tda_merged) → DataFrame`
```python
# Stack TACTIC and TDA data (union)
# Add DATA_SOURCE column: "TACTIC" or "TDA"
# Ensures both datasets contribute to scoring
```

---

### 3. **Scoring Engine** (The Core Logic)

#### `build_scores(plans, merged) → DataFrame`
```
Purpose: Aggregate claims per PRODUCT_GROUP × TACTIC_TYPE and compute effectiveness scores

STEP 1: AGGREGATE claims per product×tactic
────────────────────────────────────────────
  claim_count           = count of claims
  total_claim_amount    = sum of CLAIM_AMOUNT
  approved_count        = count where CLAIM_STATUS = "APPROVED"
  pending_count         = count where CLAIM_STATUS in ("SENT", "OPEN", "SUBMITTED")
  rejected_count        = count where CLAIM_STATUS in ("REJECTED", "CANCELLED")
  
  mode fields           = most frequent value (brand, channel, country, etc.)

STEP 2: CALCULATE APPROVAL PROXY
─────────────────────────────────
  approval_proxy = approved_count + 0.5 × pending_count
  
  WHY 0.5?
  - Fully approved claims = 100% credit (weight = 1.0)
  - Pending claims = uncertain (weight = 0.5) — might be approved later
  - Rejected claims = 0% credit (weight = 0.0)
  
  approval_rate = approval_proxy / claim_count
                (clipped to 0.0 if no claims)

STEP 3: CALCULATE EFFECTIVENESS SCORE
──────────────────────────────────────
  score = approval_rate × log1p(claim_count) × log1p(total_claim_amount)
  
  Intuition:
  ✓ Higher approval_rate → higher score
  ✓ More claims (volume signal) → higher score
  ✓ Larger total amount (financial impact) → higher score
  
  log1p() = log(1 + x), handles zero gracefully

STEP 4: ENSURE FULL COVERAGE
────────────────────────────
  - Take all unique (PRODUCT_GROUP, TACTIC_TYPE) pairs from plans
  - Join with aggregated claims
  - Fill missing: claim_count=0, approval_rate=0, score=0
  - This ensures tactics with zero claims still appear in output

STEP 5: RANK WITHIN PRODUCT
──────────────────────────
  score_rank = rank(score) per PRODUCT_GROUP (descending)
  - score_rank == 1 → best tactic for that product
  - score_rank == 2 → second best, etc.

RETURN: Full DataFrame sorted by (PRODUCT_GROUP, score_rank)
```

**Example output for Oils:**
```
PRODUCT_GROUP | TACTIC_TYPE                      | claim_count | approval_rate | score      | score_rank
──────────────┼────────────────────────────────────┼─────────────┼───────────────┼────────────┼────────────
Oils          | BB_TC_ FLYERS & INSTORE ADVT.    | 1985        | 0.50          | 51.82      | 1.0
Oils          | BB_OSA_RENTALS_EXISTING PRDS     | 6           | 0.50          | 4.35       | 2.0
Oils          | NC_CST_FXD CONTRACTUAL REBATES   | 14          | 1.00          | 16.22      | 3.0
```

---

#### `pick_best(agg) → DataFrame`
```python
# Filter: keep only rows where score_rank == 1
# Return: one best tactic per product group
# Sort: descending by score
```

---

### 4. **Model-Based Scoring** (Optional)

#### `build_training_frame(merged) → (X, y)`
```python
# Extract features: TACTIC_TYPE, PRODUCT_GROUP, BRAND, CHANNEL, etc.
# Extract labels: map CLAIM_STATUS → binary (1=approved, 0=pending/rejected/unknown)
# Filter: keep only rows with valid labels
# Return: feature matrix X, label series y
```

#### `train_approval_model(X, y) → (Pipeline, dict)`
```
PURPOSE: Predict approval probability for any product×tactic combination

ARCHITECTURE: Sklearn Pipeline
┌─────────────────────────────────────────┐
│ 1. Preprocessing (ColumnTransformer)    │
│    ├─ Categorical: OneHotEncoding       │
│    └─ Numeric: StandardScaling          │
│                                          │
│ 2. Classification (LogisticRegression)  │
│    ├─ max_iter=400                      │
│    ├─ class_weight="balanced"           │
│    └─ n_jobs=-1 (parallel)              │
└─────────────────────────────────────────┘

TRAINING:
  - Train/val split: 80/20 stratified
  - Output: probabilities for each product×tactic candidate

METRICS RETURNED:
  accuracy      = (TP + TN) / Total
  f1            = harmonic mean of precision & recall
  roc_auc       = area under ROC curve
  positive_rate_val = % of approved claims in validation set
```

#### `score_with_model(model, agg) → DataFrame`
```python
# For each (PRODUCT_GROUP, TACTIC_TYPE) row in agg:
#   1. Create feature vector (extract modes, means)
#   2. Call model.predict_proba() → pred_approval_prob (0.0 to 1.0)
#   3. Calculate pred_expected_score = pred_approval_prob × log(claim_count) × log(amount)
#   4. Rank per product by pred_expected_score
#
# Return: agg DataFrame with new columns:
#   - pred_approval_prob
#   - pred_expected_score
#   - pred_rank (1 = best predicted, model-based)
```

---

### 5. **Output Generation**

#### **Heuristic Outputs (Always)**
```
task3_best_tactic_per_product.csv
├─ PRODUCT_GROUP
├─ TACTIC_TYPE (best heuristic tactic)
├─ score (heuristic score)
├─ approval_rate
├─ claim_count
├─ total_claim_amount
└─ approved_count, pending_count, rejected_count

task3_tactic_scores_full.csv
├─ All columns from heuristic scoring
├─ All columns from aggregation
├─ score_rank (heuristic ranking)
└─ Can have multiple rows per product (all tactics ranked)
```

#### **Model-Based Outputs (Conditional)**
```
Only generated if label variety is sufficient (y.nunique() >= 2 and min_count >= 2)

task3_best_tactic_per_product_model.csv
├─ PRODUCT_GROUP
├─ TACTIC_TYPE (best model-predicted tactic)
├─ pred_approval_prob (model confidence)
├─ pred_expected_score (model score)
├─ score (heuristic score for comparison)
└─ approval_rate

task3_tactic_scores_model.csv
├─ All aggregated metrics
├─ Heuristic scores + rankings
├─ Model scores + rankings (pred_rank, pred_rank_2026)
└─ pred_expected_score_2026 (if 2026-specific data available)

task3_model_metrics.json
├─ accuracy
├─ f1
├─ roc_auc
├─ train_samples
├─ val_samples
└─ positive_rate_val
```

---

## 📈 Complete Example: Oils Product

### Raw Data
```
1,985 claims for "BB_TC_ FLYERS & INSTORE ADVT."
├─ Status: Open (1,985) → Pending
├─ Status: Approved (0)
├─ Status: Rejected (0)
└─ Total amount: $845,906
```

### Aggregation Step
```
claim_count = 1,985
approved_count = 0
pending_count = 1,985
total_claim_amount = 845,906

approval_proxy = 0 + 0.5 × 1,985 = 992.5
approval_rate = 992.5 / 1,985 = 0.50 (50%)
```

### Scoring Step
```
score = approval_rate × log1p(claim_count) × log1p(amount)
      = 0.50 × log1p(1,985) × log1p(845,906)
      = 0.50 × 7.600 × 13.648
      = 51.82
```

### Output
```
✓ Tactic: BB_TC_ FLYERS & INSTORE ADVT.
✓ Score: 51.82 (highest for Oils)
✓ Approval rate: 50% (all pending)
✓ Claim count: 1,985
✓ Total amount: $845,906
✓ Rank: 1 (best tactic for Oils)
```

---

## 🎯 Key Decision Points

| Decision | Logic |
|----------|-------|
| **Why weight pending 0.5?** | Conservative: they might not be approved. Equal weight to uncertainty. |
| **Why use log scores?** | Non-linear scaling: 100 claims → 5.6 score boost; 1M amount → 14 boost. Prevents extreme outliers. |
| **Why train ML model?** | Goes beyond heuristic: learns patterns like "certain channels/brands/countries → higher approval" |
| **When skip model training?** | When all claims have same status (e.g., all pending). Can't learn binary classification without variety. |
| **Why pick top-1 per product?** | Business requirement: one recommended tactic per product group for simplicity. |

---

## 🚀 Execution Flow

```
main()
  │
  ├─ load_plans() + load_tda_plans()
  ├─ load_claims() + load_tda_claims()
  │
  ├─ prepare_enriched_claims() + prepare_tda_enriched_claims()
  ├─ combine_data_sources()  # Union TACTIC + TDA
  │
  ├─ build_scores()          # Heuristic aggregation
  ├─ pick_best()             # Top-1 per product
  ├─ Save: task3_best_tactic_per_product.csv
  ├─ Save: task3_tactic_scores_full.csv
  │
  ├─ build_training_frame()
  ├─ Check label variety
  │
  ├─ IF sufficient variety:
  │   ├─ train_approval_model()     # Logistic regression
  │   ├─ score_with_model()         # Get predictions
  │   ├─ pick_best_model()          # Top-1 by model score
  │   ├─ Save: task3_best_tactic_per_product_model.csv
  │   ├─ Save: task3_tactic_scores_model.csv
  │   └─ Save: task3_model_metrics.json
  │
  └─ END
```

---

## 📊 Why This Approach Works

| Problem | Solution |
|---------|----------|
| No direct sales/volume data | Use claim count + amount as proxy for business impact |
| Uncertain (pending) claims | Weight 0.5: conservative but not zero |
| Need to compare tactics | Score = approval × volume × impact = composite metric |
| Want ML insight | Optional logistic model learns feature interactions |
| Multiple data sources | Combine TACTIC + TDA for richer signal |

---

## 🔍 Quick Debug Tips

**Check if model trained:**
```bash
ls results/task3_*_model.csv  # If exist = model ran
cat results/task3_model_metrics.json  # See performance
```

**Check if heuristic scores are correct:**
```bash
# Find product, get its tactics
head -50 results/task3_tactic_scores_full.csv | grep "Oils"
# Verify: score_rank=1 has highest score for that product
```

**Check approval rate calculation:**
```
approval_rate should be ≈ 0.5 if mostly pending claims
approval_rate should be close to 1.0 if mostly approved
approval_rate should be < 0.1 if mostly rejected
```

---

## Summary

**Task 3 = Find Best Tactic per Product**
- **Input:** TACTIC plans + claims + TDA SIP data
- **Heuristic:** Score = approval_rate × log(volume) × log(amount)
- **Optional ML:** Logistic regression predicts approval probability
- **Output:** Best tactic per product + full rankings
- **Use Case:** Recommend promotional strategy for each product group
