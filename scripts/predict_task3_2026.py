"""
Predict best tactic type per product group for 2026.

Approach:
- Reuse Task3 loaders and model to estimate approval probability.
- Build aggregates per PRODUCT_GROUP × TACTIC_TYPE.
- Score with model to get `pred_expected_score`.
- Optionally weight by 2026 sales forecast per product if available.
- Output top tactic per product and full scored list.

Run:
  python -m scripts.predict_task3_2026
"""
from pathlib import Path
import sys
import pandas as pd
import numpy as np

# Ensure scripts package is importable when run directly
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.training.task3_best_tactic_classifier import (
    load_plans,
    load_claims,
    prepare_enriched_claims,
    build_scores,
    build_training_frame,
    train_approval_model,
    score_with_model,
)

DATA_DIR = ROOT / "data/csv"
OUTPUT_DIR = ROOT / "output"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FORECAST_2026_PATH = OUTPUT_DIR / "task2_2026_forecast_retrained.csv"

WEIGHT_BY_FORECAST = True  # set False to ignore sales forecast weighting


def load_forecast_2026() -> pd.DataFrame | None:
    if FORECAST_2026_PATH.exists():
        df = pd.read_csv(FORECAST_2026_PATH)
        # Expect columns like PRODUCT_GROUP and TOTAL_2026 (or seasonal columns)
        total_cols = [c for c in ["TOTAL_2026", "Winter", "Spring", "Summer", "Fall", "Festival"] if c in df.columns]
        if not total_cols:
            return None
        df[total_cols] = df[total_cols].apply(pd.to_numeric, errors="coerce")
        df["TOTAL_2026"] = df[total_cols].sum(axis=1)
        return df[["PRODUCT_GROUP", "TOTAL_2026"]].dropna()
    return None


def main():
    print("=" * 80)
    print("TASK 3: Predict Best Tactic per Product for 2026")
    print("=" * 80)

    print("\n📥 Loading historical TACTIC data (TACTIC files only)...")
    plans, _ = load_plans()
    claims, _ = load_claims()
    tactic_merged = prepare_enriched_claims(plans, claims)
    # Use only TACTIC data for training and scoring
    merged = tactic_merged
    agg = build_scores(plans, merged)

    # Train approval model
    X, y = build_training_frame(merged)
    if y.nunique() < 2 or y.value_counts().min() < 2:
        print("\n⚠️  Not enough label variety to train model; falling back to heuristic scores.")
        agg_scored = agg.copy()
        agg_scored["pred_approval_prob"] = np.nan
        agg_scored["pred_expected_score"] = agg_scored["score"].astype(float)
    else:
        model, metrics = train_approval_model(X, y)
        print(f"\n✅ Model trained: accuracy={metrics['accuracy']:.3f}, f1={metrics['f1']:.3f}, roc_auc={metrics['roc_auc']:.3f}")
        agg_scored = score_with_model(model, agg)

    # Optionally weight by 2026 sales forecast volume per product
    if WEIGHT_BY_FORECAST:
        forecast_df = load_forecast_2026()
        if forecast_df is not None and not forecast_df.empty:
            vol = forecast_df.set_index("PRODUCT_GROUP")["TOTAL_2026"].to_dict()
            # Normalize volume to [0,1] for stable scaling
            vals = np.array(list(vol.values()), dtype=float)
            if len(vals) > 0 and np.nanmax(vals) > 0:
                vmax = np.nanmax(vals)
                vol_norm = {k: (float(v) / float(vmax)) if vmax > 0 else 0.0 for k, v in vol.items()}
            else:
                vol_norm = {k: 0.0 for k in vol.keys()}
            agg_scored["vol_weight"] = agg_scored["PRODUCT_GROUP"].map(vol_norm).fillna(0.0)
            # Blend model score with volume weight (alpha controls influence)
            alpha = 0.5
            base = agg_scored["pred_expected_score"].astype(float)
            agg_scored["pred_expected_score_2026"] = (1 - alpha) * base + alpha * (base * (1 + agg_scored["vol_weight"]))
        else:
            print("\nℹ️  2026 forecast file not found or missing expected columns; skipping volume weighting.")
            agg_scored["pred_expected_score_2026"] = agg_scored["pred_expected_score"].astype(float)
    else:
        agg_scored["pred_expected_score_2026"] = agg_scored["pred_expected_score"].astype(float)

    # Pick best per product for 2026
    agg_scored["pred_rank_2026"] = agg_scored.groupby("PRODUCT_GROUP")["pred_expected_score_2026"].rank(ascending=False, method="first")
    cols = [
        "PRODUCT_GROUP",
        "TACTIC_TYPE",
        "pred_approval_prob",
        "pred_expected_score",
        "pred_expected_score_2026",
        "score",
        "approval_rate",
        "claim_count",
        "total_claim_amount",
    ]
    best_2026 = agg_scored[agg_scored["pred_rank_2026"] == 1][cols].sort_values("pred_expected_score_2026", ascending=False)

    out_best = RESULTS_DIR / "task3_2026_tactic_recommendations.csv"
    out_full = RESULTS_DIR / "task3_2026_tactic_scores.csv"

    best_2026.to_csv(out_best, index=False)
    agg_scored.to_csv(out_full, index=False)

    print(f"\nSaved 2026 recommendations: {out_best} (rows={len(best_2026)})")
    print(f"Saved 2026 full scores:     {out_full} (rows={len(agg_scored)})")


if __name__ == "__main__":
    main()
