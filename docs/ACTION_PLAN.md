# TASK 2 MODEL ENHANCEMENT ACTION PLAN
## Analysis Complete → Ready for Implementation
**Date**: January 5, 2026 | **Status**: ✅ READY

---

## ANALYSIS RESULTS SUMMARY

### Data Analyzed
- ✅ **1,043,293 records** from TDA/SIP Claims (FY21-26)
- ✅ **7,816 records** from TACTIC Claims (FY21-26)
- ✅ **142 CSV files** processed and validated
- ✅ **6 fiscal years** of historical data

### New Seasonality File Created
**File**: `models/product_seasonality_enhanced.csv`
- **Size**: 25,632 rows | **Products**: 1,749 unique
- **Old file**: 72 rows | **Products**: 6 unique
- **Improvement**: **+29,050% coverage** | **4.7x higher seasonality range**

### Key Findings
1. **Peak months**: August (1.15x), October (1.13x), July (1.10x), September (1.09x)
2. **Low months**: April (0.82x), May (0.83x), January (0.90x), June (0.91x)
3. **Most seasonal products**: Range from 0.0x - 11.73x (previously max 2.50x)
4. **Current gap**: 1,743 products missing from old seasonality file

---

## IMPLEMENTATION PLAN

### Phase 1: Update Model Code (30 minutes)

#### 1.1 Update Training Script
**File**: `train_task2_reboot.py`

**Change 1** - Load new seasonality (find this section):
```python
# OLD (around line 150):
seasonality_df = pd.read_csv(f'{artifacts_dir}/product_seasonality_claims.csv')

# NEW:
seasonality_df = pd.read_csv(f'{artifacts_dir}/product_seasonality_enhanced.csv')
```

**Change 2** - In the forecast function, update how seasonality is applied:
```python
# If currently using fixed SEASONALITY_MULTIPLIER, replace with:
seasonal_idx = seasonality_df[
    (seasonality_df['PRODUCT_GROUP'] == product_group) & 
    (seasonality_df['MONTH'] == month)
]['SEASONALITY_INDEX'].values

seasonal_multiplier = seasonal_idx[0] if len(seasonal_idx) > 0 else 1.0
```

---

#### 1.2 Update Inference Script
**File**: `app_task2_inference_snippet.py` (integrated in app.py)

**Change** - Replace volume-based boost with seasonality index:
```python
# OLD (around line 200):
# volume_boost = 2.0 if customer_product_avg < 200 else ...
# prediction *= volume_boost

# NEW:
# Get seasonality index
seasonal_idx = seasonality_df[
    (seasonality_df['PRODUCT_GROUP'] == product_code) &
    (seasonality_df['MONTH'] == forecast_month)
]['SEASONALITY_INDEX'].values

if len(seasonal_idx) > 0:
    seasonal_multiplier = seasonal_idx[0]
    confidence = seasonality_df[
        (seasonality_df['PRODUCT_GROUP'] == product_code) &
        (seasonality_df['MONTH'] == forecast_month)
    ]['CONFIDENCE'].values[0]
    
    # Apply with confidence weighting
    adjusted_multiplier = 1.0 + (seasonal_multiplier - 1.0) * confidence
    prediction *= adjusted_multiplier
else:
    # Fallback for unknown products
    prediction *= 1.0
```

---

### Phase 2: Retrain Model (2 hours)

#### 2.1 Execute Training
```powershell
# In venv terminal:
cd d:\BlueRose\iffco_discount_model
.\venv\Scripts\Activate.ps1
python train_task2_reboot.py
```

**Expected Output**:
- New training logs with enhanced seasonality loaded
- New model artifacts: `task2_reboot_model.pkl`
- Report showing metrics (Train R², Test R², APE)

**Expected Improvement**:
- Previous Median APE: 67.7%
- Target Median APE: 45-55%

---

#### 2.2 Validate on Test Set
```powershell
python test_task2_model.py
```

**Expected Results**:
- Accuracy within ±20%: 35-40% (vs current 21%)
- Accuracy within ±50%: 65-70% (vs current 52.4%)

---

### Phase 3: Deploy to Streamlit (1 hour)

#### 3.1 Update Streamlit App
**File**: `app.py`

Ensure Tab 2 (Sales Forecasting) loads the new seasonality file:
```python
# At top of Tab 2 section:
seasonality_df = pd.read_csv('models/product_seasonality_enhanced.csv')
```

#### 3.2 Test Deployment
```powershell
# Terminal 1: Start Streamlit
.\venv\Scripts\Activate.ps1
streamlit run app.py

# Terminal 2: Test the app
Start-Process "http://localhost:8501"
```

**Test Cases**:
1. Select a highly seasonal product (e.g., product 316600001630)
2. Verify forecast shows peaks in Aug/Sep (not May)
3. Select a low-seasonal product
4. Verify relatively flat forecast

---

## DETAILED CHANGE LOCATIONS

### File: `train_task2_reboot.py`

**Location 1**: Seasonality Loading
```
Line ~150: seasonality_df = pd.read_csv(...)
```
**Action**: Change filename from `product_seasonality_claims.csv` to `product_seasonality_enhanced.csv`

**Location 2**: Seasonal Multiplier Application
```
Search for: "SEASONALITY_MULTIPLIER"
```
**Action**: If using fixed value, replace with dynamic lookup from seasonality_df

---

### File: `app_task2_inference_snippet.py`

**Location**: Volume Boost Logic
```
Line ~200-250: forecast_recursive() function
```
**Action**: Replace volume_boost = 2.0 logic with seasonal_idx lookup

**Key Fields to Use**:
- `seasonality_df['SEASONALITY_INDEX']` - The multiplier to apply
- `seasonality_df['CONFIDENCE']` - How much to trust this pattern (0.5-1.0)

---

## QUALITY ASSURANCE CHECKLIST

- [ ] Updated `train_task2_reboot.py` to load enhanced seasonality
- [ ] Updated `app_task2_inference_snippet.py` with new inference logic
- [ ] Ran training script successfully without errors
- [ ] Generated new model artifacts (task2_reboot_model.pkl)
- [ ] Tested on validation set (2025 data)
- [ ] Validated improvements (APE < 60%)
- [ ] Deployed to Streamlit
- [ ] Tested seasonal products show correct peaks (Aug-Oct)
- [ ] Tested low-seasonal products show flat predictions
- [ ] App runs without errors

---

## RISK MITIGATION

### Risk 1: Model overfits to new seasonality
**Mitigation**: Keep min_samples_leaf=50, monitor train/test R² gap

### Risk 2: Forecast spikes too high for extreme products
**Mitigation**: Use confidence weighting (only apply 60-80% of seasonal adjustment for low-confidence products)

### Risk 3: Unknown products crash forecast
**Mitigation**: Use fallback (seasonal_idx[0] if len(...) > 0 else 1.0)

### Risk 4: Changes introduce bugs
**Mitigation**: Test on sample predictions before deploying to Streamlit

---

## EXPECTED METRICS AFTER IMPLEMENTATION

| Metric | Current | Target | Improvement |
|--------|---------|--------|------------|
| Median APE | 67.7% | 45-55% | **20-25%** |
| Accuracy ±20% | 21.0% | 35-40% | **+67-90%** |
| Accuracy ±50% | 52.4% | 65-70% | **+24-34%** |
| Products covered | 6 | 1,749 | **+29,050%** |
| Seasonality range | 0.20-2.50x | 0.00-11.73x | **4.7x** |

---

## FILES CREATED DURING ANALYSIS

| File | Purpose | Size |
|------|---------|------|
| `models/product_seasonality_enhanced.csv` | New seasonality patterns (1,749 products) | 1.1 MB |
| `DATA_ANALYSIS_REPORT.md` | Detailed findings & recommendations | 8 KB |
| `ANALYSIS_SUMMARY.md` | Quick reference & action items | 5 KB |
| `analyze_new_tda_tactic_data.py` | Data exploration script | 7 KB |
| `extract_enhanced_seasonality.py` | Seasonality extraction script | 8 KB |
| `show_seasonality_comparison.py` | Comparison visualization | 2 KB |

---

## NEXT IMMEDIATE STEPS

1. **RIGHT NOW**: Review this action plan
2. **TODAY**: 
   - Make changes to train_task2_reboot.py (5 min)
   - Make changes to app_task2_inference_snippet.py (15 min)
   - Run training (2 hours)
   - Test predictions (30 min)
3. **TODAY EVENING**: 
   - Deploy to Streamlit
   - Validate in browser
   - Monitor for errors

---

## SUCCESS CRITERIA

✅ **Success** = New model with enhanced seasonality deployed and showing:
- [ ] Median APE < 60% (improvement from 67.7%)
- [ ] Seasonal products (Aug-Oct) show 1.2-2.0x peaks
- [ ] No forecast errors on unknown products
- [ ] Streamlit app runs smoothly

---

## CONTACT & NOTES

- **Data Source**: `data/csv/` folder (142 new files)
- **Enhanced Seasonality**: `models/product_seasonality_enhanced.csv`
- **Current Model**: `models/task2_reboot_model.pkl`
- **All work done in venv**: ✓ Confirmed

---

**Status**: ✅ ANALYSIS COMPLETE | ⏳ AWAITING IMPLEMENTATION

*Ready to proceed with code changes when you confirm.*
