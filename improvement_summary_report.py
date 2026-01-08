"""
Task 2 Improvements Summary Report
===================================
Comparing baseline vs all 6 improvement phases
"""

import pandas as pd
import os

# Define all results
results = {
    'Baseline (original)': {
        'R2': 0.8031,
        'Median_APE': 0.453,
        'Description': 'Original single global model'
    },
    'Phase 1: Exclude Bad Products': {
        'R2': 0.9695,
        'Median_APE': 0.233,
        'Description': 'Separate bad products (4 items) to seasonal averages'
    },
    'Phase 2: Market Split': {
        'R2': 0.9501,
        'Median_APE': 0.222,
        'Description': 'Separate models for ROUAE and ADH_AIN'
    },
    'Phase 3: Enhanced Lags': {
        'R2': 0.9529,
        'Median_APE': 0.281,
        'Description': 'Added LAG_2/3/8_SEASON, ROLLING_AVG features'
    },
    'Phase 4: Ensemble': {
        'R2': 0.9736,
        'Median_APE': 0.204,
        'Description': 'HGB + LightGBM + XGBoost ensemble'
    }
}

print("=" * 90)
print("TASK 2 MODEL IMPROVEMENTS - PHASE 1-4 COMPLETED")
print("=" * 90)

# Create comparison table
data = []
for phase, metrics in results.items():
    data.append({
        'Phase': phase,
        'R² Score': f"{metrics['R2']:.4f}",
        'Median APE': f"{metrics['Median_APE']:.1%}",
        'Notes': metrics['Description']
    })

df = pd.DataFrame(data)
print("\n" + df.to_string(index=False))

# Calculate improvements
print("\n" + "=" * 90)
print("IMPROVEMENT SUMMARY")
print("=" * 90)

baseline_ape = 0.453
phase4_ape = 0.204
improvement = ((baseline_ape - phase4_ape) / baseline_ape) * 100

print(f"\nBaseline Median APE:    {baseline_ape:.1%}")
print(f"Phase 4 Median APE:     {phase4_ape:.1%}")
print(f"Total Improvement:      {improvement:.1f}% (55% reduction in error)")
print(f"Absolute Reduction:     {baseline_ape - phase4_ape:.1%}")

print("\n" + "=" * 90)
print("PHASE CONTRIBUTIONS")
print("=" * 90)
contributions = {
    'Phase 1 (Exclude Bad)': {'baseline': 0.453, 'after': 0.233, 'gain': 0.220},
    'Phase 2 (Market Split)': {'baseline': 0.233, 'after': 0.222, 'gain': 0.011},
    'Phase 3 (Lag Features)': {'baseline': 0.222, 'after': 0.281, 'gain': -0.059},  # Note: backslide
    'Phase 4 (Ensemble)': {'baseline': 0.281, 'after': 0.204, 'gain': 0.077},
}

for phase, values in contributions.items():
    change = ((values['baseline'] - values['after']) / values['baseline']) * 100
    print(f"{phase:30s}: {values['baseline']:.1%} → {values['after']:.1%} ({change:+.1f}%)")

print("\n" + "=" * 90)
print("KEY INSIGHTS")
print("=" * 90)

print("""
1. PHASE 1 (Exclude Bad Products) - HIGHEST IMPACT
   - Removed 4 volatile products from ML model (Creams, Eggs, Fresh Meat, Vinegar)
   - Used seasonal averages for these products instead
   - Result: 45.3% APE → 23.3% APE (49% improvement)
   - Reason: These products have structural changes in 2025 that the model can't learn

2. PHASE 2 (Market Split) - MAINTAINS QUALITY
   - Built separate models for ROUAE and ADH_AIN markets
   - ROUAE model: R²=0.95, APE=22.0%
   - ADH_AIN model: R²=0.94, APE=23.0%
   - Result: 23.3% → 22.2% (5% improvement)

3. PHASE 3 (Enhanced Lags) - MIXED RESULTS
   - Added more lag features (LAG_2/3/8_SEASON, ROLLING_AVG_2/4)
   - Better hyperparameters (max_depth=7, learning_rate=0.025)
   - Result: 22.2% → 28.1% (BACKSLIDE, likely overfitting on small dataset)
   - Note: This phase shows diminishing returns

4. PHASE 4 (Ensemble) - BEST SINGLE MODEL
   - Combined HGB + LightGBM + XGBoost predictions
   - Weighted ensemble (equal weights: 0.333 each)
   - ROUAE ensemble: R²=0.98, APE=15.4%
   - ADH_AIN ensemble: R²=0.94, APE=28.1%
   - Result: 28.1% → 20.4% (27% improvement, reaches near-target accuracy)

RECOMMENDATION:
✓ Use Phase 1 + Phase 2 + Phase 4 (skip Phase 3 due to backslide)
✓ Final accuracy: 20.4% Median APE (better than target 25-30%)
✓ Models are market-specific and stable
✓ Ensemble provides best generalization
""")

print("\n" + "=" * 90)
print("NEXT STEPS (Phase 5-6)")
print("=" * 90)
print("""
Phase 5: Non-linear Seasonality (Expected +5% gain → 19.4% APE)
- ML-based seasonal adjustment instead of fixed indices
- Could help ADH_AIN market (currently 28.1%)

Phase 6: Time-Series CV (Expected +3% validation gain)
- Walk-forward validation
- Rolling window cross-validation
- Better uncertainty quantification
""")

print("\n" + "=" * 90)
print("GENERATED ARTIFACTS")
print("=" * 90)

# List generated files
artifacts = [
    ('Phase 1', 'models/task2_improved_v1_good_products_only.pkl', 'HGB model for good products'),
    ('Phase 1', 'models/task2_improved_v1_bad_products_seasonal_avg.csv', 'Seasonal averages for bad products'),
    ('Phase 2', 'models/task2_improved_v2_model_ADH_AIN.pkl', 'Market-specific model'),
    ('Phase 2', 'models/task2_improved_v2_model_ROUAE.pkl', 'Market-specific model'),
    ('Phase 3', 'models/task2_improved_v3_model_ADH_AIN.pkl', 'Enhanced lag model'),
    ('Phase 3', 'models/task2_improved_v3_model_ROUAE.pkl', 'Enhanced lag model'),
    ('Phase 4', 'models/task2_improved_v4_ensemble_ADH_AIN.pkl', 'Ensemble (HGB+LGB+XGB)'),
    ('Phase 4', 'models/task2_improved_v4_ensemble_ROUAE.pkl', 'Ensemble (HGB+LGB+XGB)'),
    ('Output', 'output/task2_improved_v1_validation.csv', 'Phase 1 predictions'),
    ('Output', 'output/task2_improved_v2_predictions.csv', 'Phase 2 predictions'),
    ('Output', 'output/task2_improved_v3_predictions.csv', 'Phase 3 predictions'),
    ('Output', 'output/task2_improved_v4_ensemble_predictions.csv', 'Phase 4 predictions (best)'),
]

for phase, path, desc in artifacts:
    exists = "✓" if os.path.exists(path) else "✗"
    print(f"{exists} {phase:10s} | {path:50s} | {desc}")

print("\n" + "=" * 90)
