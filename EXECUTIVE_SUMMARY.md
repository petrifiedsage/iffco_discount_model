# Executive Summary: Data File Analysis for Tactic Type Prediction Model

**Date:** January 8, 2026  
**Analysis:** Data sources for enhanced tactic type prediction model  
**Objective:** Identify files with required features and maximize training data

---

## Key Findings

### 🎯 Best Data Sources Identified

| Rank | Data Source | Features | Samples | Match Rate | Recommendation |
|------|-------------|----------|---------|------------|----------------|
| 1 | **TACTIC_CLAIMS + TACTIC_PLAN** | 100% | 2,201 | 96.5% | ✅ PRIMARY |
| 2 | **TDA_SIP_CLAIM + TDA_SIP_PLAN** | 92.9% | 57,496 | 55.7% | ✅ ADD |
| 3 | **CONTRACT_CLAIM_DUMP** | 28.6% | 42,946 | N/A | ⚠️ OPTIONAL |

### 📊 Impact Summary

**Current State:**
- Training samples: 2,201
- Features: 14
- Tactic types: 32
- Historical coverage: FY 24-25 only

**Recommended Enhanced State:**
- Training samples: **~34,000** (15x increase)
- Features: **23+** (9 new features)
- Tactic types: **40+** (broader coverage)
- Historical coverage: **FY 21-26** (5 years)

### 💡 Critical Insights

1. **TACTIC files** (currently used):
   - ✅ 100% feature coverage when joined
   - ✅ High-quality outcome labels
   - ⚠️ Limited training samples (2,201)
   - ⚠️ One-to-many join (avg 5 plan rows per claim)

2. **TDA SIP files** (recommended addition):
   - ✅ 26x more samples than TACTIC
   - ✅ 92.9% feature coverage
   - ✅ 99.9% approval rate (successful tactics)
   - ⚠️ Join on TACTIC_NUMBER (not composite key!)
   - ⚠️ 55.7% of claims have matching plan data

3. **CONTRACT_CLAIM_DUMP** (optional):
   - ✅ Large volume (42,946 records)
   - ✅ Alternative outcome labels
   - ❌ Only 3 tactic types
   - ❌ Missing key features (BRAND, DEAL_TYPE, etc.)

---

## New Features Discovered

### High-Value Features (14 new)

**Categorization & Classification:**
- `GL_CODE` - General ledger categorization
- `BUDGET_SPENT_TYPE` - Spending classification
- `TCAT` - Tactic category

**Status & Approval:**
- `DEAL_STATUS` - Deal approval predictor
- `PLAN_STATUS` - Planning quality indicator
- `DETAILS_STATUS` - Activity approval status
- `SUB_STATUS` - Detailed approval stage

**Customer & Location:**
- `REGISTRY_ID` - Customer registry ID
- `EBS_PARTY_SITE_ID` - Customer site identifier
- `GEO_CLUST` - Geographic cluster
- `AREA_GROUP` - Area grouping

**Financial & Planning:**
- `BUDGET_ID` - Budget allocation tracking
- `COST_CENTER` - Cost center code

**Sales Type:**
- `SALES_TYPE` - Sell-In vs. other (TDA only)

---

## Join Strategy (CRITICAL)

### ✅ CORRECT APPROACH

**TACTIC Files:**
```python
merged = tactic_claims.merge(
    tactic_plan, 
    on='TACTIC_PLAN',
    how='left'
)
# Handle 1:Many relationship (avg 5 rows per claim)
merged = merged.drop_duplicates(subset=['CLAIM_ID'])
```

**TDA Files:**
```python
merged = tda_claim.merge(
    tda_plan,
    on='TACTIC_NUMBER',  # ← Use TACTIC_NUMBER
    how='left'
)
# Results in 1:1 relationship, 55.7% match
```

### ❌ AVOID

**TDA Composite Key (causes explosion):**
```python
# DON'T DO THIS - creates 23M rows from 57K!
merged = tda_claim.merge(
    tda_plan,
    on=['TDA_PLAN_ID', 'TDA_PLAN_ROW_ID'],  # ← WRONG
    how='left'
)
```

---

## Implementation Roadmap

### Phase 1: Enhance TACTIC Model (Immediate)
**Timeline:** 1-2 days  
**Impact:** +5-8% accuracy improvement

Actions:
- [ ] Add 14 new features to TACTIC training
- [ ] Engineer customer behavior features
- [ ] Add financial ratio features
- [ ] Load historical TACTIC data (FY 21-26)

### Phase 2: Integrate TDA Data (High Priority)
**Timeline:** 2-3 days  
**Impact:** +10-15% accuracy improvement

Actions:
- [ ] Load all TDA SIP files (13 plan + 13 claim files)
- [ ] Join using TACTIC_NUMBER
- [ ] Standardize column names
- [ ] Combine with TACTIC data
- [ ] Retrain model with ~34K samples

### Phase 3: Add Contract Data (Optional)
**Timeline:** 1 day  
**Impact:** +2-3% for specific tactic types

Actions:
- [ ] Filter Contract Claim Dump for relevant types
- [ ] Augment contractual rebate training
- [ ] Ensemble with main model

---

## Data Quality Summary

### TACTIC Files
| Metric | Value |
|--------|-------|
| Total Claims | 2,201 |
| Total Plan Rows | 43,242 |
| Unique Tactics | 2,298 |
| Tactic Types | 32 |
| Claim Status Types | 5 |
| Join Success | 96.5% |
| Avg Rows per Claim | 5.0 |

**Top Tactic Types:**
1. NC_GEN TERM_TRADE SCHEMES (66.4%)
2. NC_GEN TERM_PRICE CORRECTION (26.3%)
3. NC_CST_FXD CONTRACTUAL REBATES (1.0%)

**Claim Status Distribution:**
- Sent: 72.7%
- Open: 25.1%
- Approved: 1.9%
- Rejected: 0.2%

### TDA SIP Files
| Metric | Value |
|--------|-------|
| Total Claims | 57,496 |
| Total Plan Rows | 64,950 |
| Unique Tactics | ~57,000 |
| Tactic Types | 13 |
| Claim Status Types | 3 |
| Join Success (TACTIC_NUMBER) | 55.7% |

**Top Tactic Types:**
1. BB_OFFSA_RENTALS_EXISTING PRDS (68.0%)
2. CP_PRICE OFF TO CONSUMER (21.1%)
3. NC_GEN TERM_PRICE CORRECTION (6.0%)

**Claim Status Distribution:**
- Approved: 99.9%
- Cancelled: 0.1%
- Submitted: 0.0%

---

## Feature Coverage Matrix

| Feature | Required? | TACTIC | TDA | CONTRACT |
|---------|-----------|--------|-----|----------|
| **TACTIC_TYPE** | ✅ Target | ✅ | ✅ | ✅ |
| PRODUCT_GROUP | ✅ | ✅ | ✅ | ✅ |
| PRODUCT_SUB_GROUP | ✅ | ✅ | ✅ | ❌ |
| BRAND | ✅ | ✅ | ✅ | ❌ |
| CHANNEL_GROUP | ✅ | ⚠️ | ⚠️ | ❌ |
| DEAL_TYPE | ✅ | ✅ | ✅ | ❌ |
| AMENDED_DEAL_AMOUNT | ✅ | ✅ | ✅ | ❌ |
| ADJUSTMENT_VALUE_AMOUNT | ✅ | ✅ | ✅ | ❌ |
| COUNTRY | ✅ | ✅ | ✅ | ❌ |
| TERRITORY | ✅ | ✅ | ❌ | ❌ |
| MONTH | ✅ | ✅ | ✅ | ✅ |
| PERIOD | ✅ | ✅ | ✅ | ✅ |
| **CLAIM_STATUS** | ✅ Label | ✅ | ✅ | ~ |
| **CLAIM_AMOUNT** | ✅ Label | ✅ | ✅ | ~ |

**Legend:**  
✅ Available | ❌ Missing | ⚠️ Partial (use proxy) | ~ Alternative available

---

## Recommendations Priority

### 🔴 HIGH PRIORITY (Do First)

1. **Add new features to TACTIC model**
   - Implement: GL_CODE, DEAL_STATUS, REGISTRY_ID, etc.
   - Expected impact: +5-8% accuracy
   - Effort: Low (1-2 days)

2. **Integrate TDA SIP data**
   - Use TACTIC_NUMBER join key
   - Adds 32,000 training samples
   - Expected impact: +10-15% accuracy
   - Effort: Medium (2-3 days)

3. **Load historical TACTIC data**
   - FY 21-22 to FY 25-26
   - Improves temporal patterns
   - Expected impact: +3-5% accuracy
   - Effort: Low (1 day)

### 🟡 MEDIUM PRIORITY (Do Later)

4. **Engineer derived features**
   - Customer behavior patterns
   - Financial ratios
   - Expected impact: +2-4% accuracy
   - Effort: Medium (2 days)

5. **Handle CHANNEL_GROUP missing data**
   - Map from CHANNEL field
   - Create proxy from customer data
   - Expected impact: +1-2% accuracy
   - Effort: Low (1 day)

### 🟢 LOW PRIORITY (Optional)

6. **Add CONTRACT_CLAIM_DUMP for specific types**
   - Only for contractual rebates
   - Expected impact: +2-3% for those types
   - Effort: Low (1 day)

---

## Risk Assessment

| Risk | Severity | Mitigation |
|------|----------|------------|
| TDA join explosion | 🔴 HIGH | Use TACTIC_NUMBER, not composite key |
| CHANNEL_GROUP missing | 🟡 MEDIUM | Use CHANNEL as proxy |
| TDA 44% unmatched claims | 🟡 MEDIUM | Use claims-only features for unmatched |
| Contract data limited types | 🟢 LOW | Use only for specific tactic types |
| TACTIC 1:Many joins | 🟡 MEDIUM | Deduplicate by CLAIM_ID |

---

## Expected Outcomes

### Model Performance
- **Current Accuracy:** Baseline
- **Phase 1 (New Features):** +5-8%
- **Phase 2 (TDA Data):** +10-15%
- **Total Expected Improvement:** +15-23%

### Model Robustness
- **15x more training data** → Better generalization
- **5 years historical** → Temporal patterns
- **40+ tactic types** → Broader coverage
- **23+ features** → Richer representations

### Business Impact
- Better tactic type recommendations
- Improved approval rate predictions
- More accurate budget forecasting
- Enhanced strategic planning

---

## Next Steps

1. ✅ Review this analysis with stakeholders
2. ⬜ Approve Phase 1 & 2 implementation
3. ⬜ Allocate development resources (3-5 days)
4. ⬜ Implement enhanced data pipeline
5. ⬜ Retrain and validate model
6. ⬜ Compare performance metrics
7. ⬜ Deploy if improvements confirmed

---

**Files Generated:**
- `DATA_FILE_ANALYSIS_REPORT.md` - Detailed technical report
- `analyze_data_files.py` - Feature coverage analysis script
- `analyze_data_quality.py` - Data quality assessment script
- `analyze_join_relationships.py` - Join strategy analysis script
- `analyze_tda_join_keys.py` - TDA join key investigation
- `EXECUTIVE_SUMMARY.md` - This document
