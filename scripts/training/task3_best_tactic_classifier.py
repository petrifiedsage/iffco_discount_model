"""
Task 3 (Option A): Recommend single best tactic per product group

Approach (no sales/volume data available):
- Use TACTIC plan + claims files to proxy effectiveness.
- Join claims to plans via TACTIC_PLAN to recover PRODUCT_GROUP and TACTIC_TYPE.
- Effectiveness score = approval_rate * log1p(total_claim_amount) * log1p(claim_count).
- For each product group, pick the tactic with highest score.

Outputs (in results/):
- task3_best_tactic_per_product.csv  : one row per product with recommended tactic + key metrics
- task3_tactic_scores_full.csv       : all tactic scores per product for inspection

Run: python -m scripts.training.task3_best_tactic_classifier
"""

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_DIR = Path("data/csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

PLAN_GLOBS = [
    "TACTIC_*_TACTIC_PLAN_DIMNS*.csv",
]
CLAIM_GLOBS = [
    "TACTIC_*_TACTIC_CM_CLAIMS_DIMNS*.csv",
    "TACTIC_*_CLAIMS_DIMNS*.csv",  # handles files named TACTIC_23_24_CLAIMS_DIMNS.csv
]
TDA_PLAN_GLOBS = [
    "TDA_FY*_SIP_PLAN_DIMNS*.csv",
]
TDA_CLAIM_GLOBS = [
    "TDA_FY*_SIP_CLAIM_DIMNS*.csv",
]

# --------------------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------------------


def load_plans() -> tuple[pd.DataFrame, int]:
    """Load plans and return (dataframe, raw_row_count_before_filtering)"""
    frames = []
    raw_count = 0
    for pattern in PLAN_GLOBS:
        for fpath in sorted(DATA_DIR.glob(pattern)):
            df = pd.read_csv(fpath, low_memory=False)
            raw_count += len(df)
            df.columns = df.columns.str.strip()
            keep_cols = [c for c in [
                "TACTIC_PLAN",
                "PRODUCT_GROUP",
                "PRODUCT_SUB_GROUP",
                "BRAND",
                "TACTIC_TYPE",
                "CHANNEL",
                "CHANNEL_GROUP",
                "COUNTRY",
                "TERRITORY",
                "GL_CODE",
                "DEAL_STATUS",
                "REGISTRY_ID",
                "START_DATE",
                "END_DATE",
                "DEAL_TYPE",
                "AMENDED_DEAL_AMOUNT",
                "ADJUSTMENT_VALUE_AMOUNT",
            ] if c in df.columns]
            if not keep_cols:
                continue
            df = df[keep_cols].dropna(subset=["TACTIC_PLAN", "PRODUCT_GROUP", "TACTIC_TYPE"], how="any")
            frames.append(df)
    if not frames:
        raise FileNotFoundError("No plan files matched; expected TACTIC_*_TACTIC_PLAN_DIMNS*.csv in data/csv")
    plans = pd.concat(frames, ignore_index=True).drop_duplicates()
    return plans, raw_count


def load_tda_plans() -> tuple[pd.DataFrame, int]:
    """Load TDA SIP plans and return (dataframe, raw_row_count_before_filtering)"""
    frames = []
    raw_count = 0
    for pattern in TDA_PLAN_GLOBS:
        for fpath in sorted(DATA_DIR.glob(pattern)):
            df = pd.read_csv(fpath, low_memory=False)
            raw_count += len(df)
            df.columns = df.columns.str.strip()
            keep_cols = [c for c in [
                "TACTIC_NUMBER",
                "PRODUCT_GROUP",
                "PRODUCT_SUB_GROUP",
                "BRAND",
                "TACTIC_TYPE",
                "CHANNEL",
                "CHANNEL_TYPE",
                "COUNTRY",
                "GL_CODE",
                "DEAL_STATUS",
                "DETAILS_STATUS",
                "REGISTRY_ID",
                "TACTIC_START_DATE",
                "TACTIC_END_DATE",
                "DETAILS_ACTIVITY_TYPE",
                "DEAL_TYPE",
                "SIP_LOCATION",
            ] if c in df.columns]
            if not keep_cols or "TACTIC_NUMBER" not in df.columns:
                continue
            df = df[keep_cols].dropna(subset=["TACTIC_NUMBER"], how="any")
            # Harmonize column names
            if "CHANNEL_TYPE" in df.columns and "CHANNEL" not in df.columns:
                df["CHANNEL"] = df["CHANNEL_TYPE"]
            if "DETAILS_STATUS" in df.columns and "DEAL_STATUS" not in df.columns:
                df["DEAL_STATUS"] = df["DETAILS_STATUS"]
            if "TACTIC_START_DATE" in df.columns:
                df["START_DATE"] = df["TACTIC_START_DATE"]
            if "TACTIC_END_DATE" in df.columns:
                df["END_DATE"] = df["TACTIC_END_DATE"]
            if "DETAILS_ACTIVITY_TYPE" in df.columns:
                df["ACTIVITY_TYPE"] = df["DETAILS_ACTIVITY_TYPE"]
            frames.append(df)
    if not frames:
        return pd.DataFrame(), 0
    plans = pd.concat(frames, ignore_index=True).drop_duplicates()
    return plans, raw_count


APPROVED_LABELS = {"APPROVED"}
PENDING_LABELS = {"SENT", "OPEN", "SUBMITTED"}
REJECTED_LABELS = {"REJECTED", "CANCELLED"}


def label_claim_status(status: str) -> float | None:
    """Map CLAIM_STATUS to binary label (1=approved, 0=rejected/pending). Return None if unknown."""
    if pd.isna(status):
        return None
    status = str(status).strip().upper()
    if status in APPROVED_LABELS:
        return 1.0
    if status in REJECTED_LABELS or status in PENDING_LABELS:
        return 0.0
    return None


def most_frequent(series: pd.Series, unknown: str = "UNKNOWN") -> str:
    """Return most frequent non-null value, else unknown."""
    if series is None:
        return unknown
    s = series.dropna().astype(str)
    if s.empty:
        return unknown
    return s.value_counts().idxmax()


def extract_product_from_tactic_name(name: str) -> str | None:
    """Extract product group name from TACTIC_NAME (fallback for orphan claims)."""
    if pd.isna(name):
        return None
    
    name_upper = str(name).upper()
    
    # Map common keywords to product groups (matching our PRODUCT_GROUP values)
    product_map = {
        'CHOCOLATE': 'Chocolate',
        'BISCUIT': 'Biscuits',
        'WAFER': 'Wafers',
        'CONFECTION': 'Confectionary',
        'SNACK': 'Snacks',
        'BEVERAGE': 'Beverages',
        'ICE CREAM': 'Ice creams',
        'OIL': 'Oils',
        'FLOUR': 'Flour',
        'CREAM': 'Creams',
        'EGG': 'Eggs',
        'MEAT': 'Fresh Meat Products',
        'BAKERY': 'Bakery Ingredients',
        'CONDIMENT': 'Condiments - Wet',
        'SOUP': 'Soups',
    }
    
    for keyword, product in product_map.items():
        if keyword in name_upper:
            return product
    
    return None


def extract_tactic_type_from_tactic_name(name: str) -> str | None:
    """Extract tactic type from TACTIC_NAME (fallback for orphan claims)."""
    if pd.isna(name):
        return None
    
    name_upper = str(name).upper()
    
    # Map common tactic keywords to standard tactic types
    tactic_map = {
        'TRADE SCHEME': 'NC_GEN TERM_TRADE SCHEMES',
        'REBATE': 'NC_CST_FXD CONTRACTUAL REBATES',
        'FXD CONTRACTUAL': 'NC_CST_FXD CONTRACTUAL REBATES',
        'PROGRESSIVE REBATE': 'NC_CST_PROGRESSIVE REBATES',
        'SALES INCENTIVE': 'NC_SI_SALES INCENTIVES',
        'OSA': 'OSA Rentals',
        'INSTORE PROMO': 'INSTORE PROMOTIONS',
        'LISTING FEE': 'LISTING FEES',
    }
    
    for keyword, tactic in tactic_map.items():
        if keyword in name_upper:
            return tactic
    
    return None


def prepare_enriched_claims(plans: pd.DataFrame, claims: pd.DataFrame) -> pd.DataFrame:
    """Merge claims with plans, recover missing PRODUCT_GROUP/TACTIC_TYPE, and fill model features."""
    merged = claims.merge(plans, on="TACTIC_PLAN", how="left")

    for col in [
        "PRODUCT_SUB_GROUP",
        "BRAND",
        "CHANNEL_GROUP",
        "COUNTRY",
        "TERRITORY",
        "DEAL_TYPE",
        "MONTH",
        "PERIOD",
        "AMENDED_DEAL_AMOUNT",
        "ADJUSTMENT_VALUE_AMOUNT",
    ]:
        if col not in merged.columns:
            merged[col] = np.nan

    missing_pg = merged["PRODUCT_GROUP"].isna().sum()
    if missing_pg > 0:
        print(f"  ⚠️  {missing_pg:,} orphan TACTIC_PLAN records; attempting TACTIC_NAME extraction...")
        if "TACTIC_NAME" in merged.columns:
            mask_orphan = merged["PRODUCT_GROUP"].isna()
            extracted_prod = merged.loc[mask_orphan, "TACTIC_NAME"].apply(extract_product_from_tactic_name)
            recovered_prod = extracted_prod.notna().sum()
            merged.loc[mask_orphan & extracted_prod.notna(), "PRODUCT_GROUP"] = extracted_prod[extracted_prod.notna()]
            print(f"     → Recovered {recovered_prod:,} products from TACTIC_NAME")

        still_missing = merged["PRODUCT_GROUP"].isna().sum()
        if still_missing > 0:
            print(f"     → {still_missing:,} records still missing PRODUCT_GROUP (labeled as UNKNOWN_PRODUCT)")
            merged["PRODUCT_GROUP"] = merged["PRODUCT_GROUP"].fillna("UNKNOWN_PRODUCT")

    missing_tt = merged["TACTIC_TYPE"].isna().sum()
    if missing_tt > 0:
        print(f"  ⚠️  {missing_tt:,} orphan TACTIC_PLAN records missing TACTIC_TYPE; attempting extraction...")
        if "TACTIC_NAME" in merged.columns:
            mask_orphan = merged["TACTIC_TYPE"].isna()
            extracted_tactic = merged.loc[mask_orphan, "TACTIC_NAME"].apply(extract_tactic_type_from_tactic_name)
            recovered_tactic = extracted_tactic.notna().sum()
            merged.loc[mask_orphan & extracted_tactic.notna(), "TACTIC_TYPE"] = extracted_tactic[extracted_tactic.notna()]
            print(f"     → Recovered {recovered_tactic:,} tactic types from TACTIC_NAME")

        still_missing = merged["TACTIC_TYPE"].isna().sum()
        if still_missing > 0:
            print(f"     → {still_missing:,} records still missing TACTIC_TYPE (labeled as UNKNOWN_TACTIC)")
            merged["TACTIC_TYPE"] = merged["TACTIC_TYPE"].fillna("UNKNOWN_TACTIC")

    for col in ["PRODUCT_SUB_GROUP", "BRAND", "CHANNEL_GROUP", "COUNTRY", "TERRITORY", "DEAL_TYPE", "MONTH", "PERIOD"]:
        merged[col] = merged[col].fillna("UNKNOWN")
    for col in ["AMENDED_DEAL_AMOUNT", "ADJUSTMENT_VALUE_AMOUNT"]:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0.0)

    merged = merged.dropna(subset=["PRODUCT_GROUP", "TACTIC_TYPE"])
    return merged


def prepare_tda_enriched_claims(tda_plans: pd.DataFrame, tda_claims: pd.DataFrame) -> pd.DataFrame:
    """Merge TDA claims with plans using TACTIC_NUMBER, harmonize features."""
    if tda_plans.empty or tda_claims.empty:
        return pd.DataFrame()
    
    merged = tda_claims.merge(tda_plans, on="TACTIC_NUMBER", how="left")
    
    # Ensure feature columns exist
    for col in [
        "PRODUCT_SUB_GROUP",
        "BRAND",
        "CHANNEL",
        "COUNTRY",
        "DEAL_TYPE",
        "GL_CODE",
        "DEAL_STATUS",
        "ACTIVITY_TYPE",
        "START_DATE",
        "END_DATE",
    ]:
        if col not in merged.columns:
            merged[col] = np.nan
    
    # Add dummy columns for consistency with TACTIC data
    merged["CHANNEL_GROUP"] = merged.get("CHANNEL", np.nan)
    merged["TERRITORY"] = "UNKNOWN"  # TDA doesn't have TERRITORY
    merged["MONTH"] = "UNKNOWN"
    merged["PERIOD"] = "UNKNOWN"
    merged["AMENDED_DEAL_AMOUNT"] = 0.0
    merged["ADJUSTMENT_VALUE_AMOUNT"] = 0.0
    
    # Fill missing values
    for col in ["PRODUCT_SUB_GROUP", "BRAND", "CHANNEL", "CHANNEL_GROUP", "COUNTRY", "TERRITORY", 
                "DEAL_TYPE", "MONTH", "PERIOD", "GL_CODE", "DEAL_STATUS", "ACTIVITY_TYPE"]:
        merged[col] = merged[col].fillna("UNKNOWN")
    
    merged = merged.dropna(subset=["PRODUCT_GROUP", "TACTIC_TYPE"])
    return merged


def combine_data_sources(tactic_merged: pd.DataFrame, tda_merged: pd.DataFrame) -> pd.DataFrame:
    """Combine TACTIC and TDA data sources, harmonizing columns."""
    if tda_merged.empty:
        print("  ℹ️  No TDA data available; using TACTIC data only")
        return tactic_merged
    
    # Ensure both have same columns
    common_cols = list(set(tactic_merged.columns) & set(tda_merged.columns))
    
    # Add source identifier
    tactic_merged = tactic_merged.copy()
    tda_merged = tda_merged.copy()
    tactic_merged["DATA_SOURCE"] = "TACTIC"
    tda_merged["DATA_SOURCE"] = "TDA"
    
    # Select common columns plus source
    if "DATA_SOURCE" not in common_cols:
        common_cols.append("DATA_SOURCE")
    
    combined = pd.concat([
        tactic_merged[common_cols],
        tda_merged[common_cols]
    ], ignore_index=True)
    
    print(f"  ✅ Combined TACTIC ({len(tactic_merged):,}) + TDA ({len(tda_merged):,}) = {len(combined):,} rows")
    return combined


def load_claims() -> tuple[pd.DataFrame, int]:
    """Load claims and return (dataframe, raw_row_count_before_filtering)"""
    frames = []
    raw_count = 0
    for pattern in CLAIM_GLOBS:
        for fpath in sorted(DATA_DIR.glob(pattern)):
            df = pd.read_csv(fpath, low_memory=False)
            raw_count += len(df)
            df.columns = df.columns.str.strip()
            if "CLAIM_AMOUNT" not in df.columns or "CLAIM_STATUS" not in df.columns or "TACTIC_PLAN" not in df.columns:
                continue
            keep = ["TACTIC_PLAN", "CLAIM_AMOUNT", "CLAIM_STATUS"]
            if "TACTIC_NAME" in df.columns:
                keep.append("TACTIC_NAME")
            df_sub = df[keep].copy()
            df_sub["CLAIM_AMOUNT"] = pd.to_numeric(df_sub["CLAIM_AMOUNT"], errors="coerce")
            df_sub["CLAIM_STATUS"] = df_sub["CLAIM_STATUS"].astype(str).str.strip().str.upper()
            frames.append(df_sub)
    if not frames:
        raise FileNotFoundError("No claims files matched; expected TACTIC_*_CLAIMS*.csv in data/csv")
    claims = pd.concat(frames, ignore_index=True)
    claims = claims.dropna(subset=["TACTIC_PLAN", "CLAIM_AMOUNT"])
    claims["CLAIM_STATUS"] = claims["CLAIM_STATUS"].fillna("UNKNOWN")
    return claims, raw_count


def load_tda_claims() -> tuple[pd.DataFrame, int]:
    """Load TDA SIP claims and return (dataframe, raw_row_count_before_filtering)"""
    frames = []
    raw_count = 0
    for pattern in TDA_CLAIM_GLOBS:
        for fpath in sorted(DATA_DIR.glob(pattern)):
            df = pd.read_csv(fpath, low_memory=False)
            raw_count += len(df)
            df.columns = df.columns.str.strip()
            if "CLAIM_AMOUNT" not in df.columns or "CLAIM_STATUS" not in df.columns or "TACTIC_NUMBER" not in df.columns:
                continue
            keep = ["TACTIC_NUMBER", "CLAIM_AMOUNT", "CLAIM_STATUS", "CLAIM_ID"]
            if "TACTIC_NAME" in df.columns:
                keep.append("TACTIC_NAME")
            df_sub = df[[c for c in keep if c in df.columns]].copy()
            df_sub["CLAIM_AMOUNT"] = pd.to_numeric(df_sub["CLAIM_AMOUNT"], errors="coerce")
            df_sub["CLAIM_STATUS"] = df_sub["CLAIM_STATUS"].astype(str).str.strip().str.upper()
            # Deduplicate on CLAIM_ID if available
            if "CLAIM_ID" in df_sub.columns:
                df_sub = df_sub.drop_duplicates(subset=["CLAIM_ID"])
            frames.append(df_sub)
    if not frames:
        return pd.DataFrame(), 0
    claims = pd.concat(frames, ignore_index=True)
    claims = claims.dropna(subset=["TACTIC_NUMBER", "CLAIM_AMOUNT"])
    claims["CLAIM_STATUS"] = claims["CLAIM_STATUS"].fillna("UNKNOWN")
    return claims, raw_count


# --------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------


def build_scores(plans: pd.DataFrame, merged: pd.DataFrame) -> pd.DataFrame:
    """Aggregate claim stats per PRODUCT_GROUP × TACTIC_TYPE and compute heuristic score."""
    # Aggregate claim stats per product × tactic (only where claims exist)
    agg = (
        merged.groupby(["PRODUCT_GROUP", "TACTIC_TYPE"], as_index=False)
        .agg(
            claim_count=("CLAIM_AMOUNT", "count"),
            total_claim_amount=("CLAIM_AMOUNT", "sum"),
            approved_count=("CLAIM_STATUS", lambda s: s.isin(APPROVED_LABELS).sum()),
            pending_count=("CLAIM_STATUS", lambda s: s.isin(PENDING_LABELS).sum()),
            rejected_count=("CLAIM_STATUS", lambda s: s.isin(REJECTED_LABELS).sum()),
            mean_amended_deal_amount=("AMENDED_DEAL_AMOUNT", "mean"),
            mean_adjustment_value_amount=("ADJUSTMENT_VALUE_AMOUNT", "mean"),
            product_sub_group_mode=("PRODUCT_SUB_GROUP", most_frequent),
            brand_mode=("BRAND", most_frequent),
            channel_group_mode=("CHANNEL_GROUP", most_frequent),
            deal_type_mode=("DEAL_TYPE", most_frequent),
            country_mode=("COUNTRY", most_frequent),
            territory_mode=("TERRITORY", most_frequent),
            month_mode=("MONTH", most_frequent),
            period_mode=("PERIOD", most_frequent),
        )
    )

    # Approval proxy: approved + 0.5 * pending (avoid div by zero)
    agg["approval_proxy"] = agg["approved_count"] + 0.5 * agg["pending_count"]
    agg["approval_rate"] = agg["approval_proxy"] / agg["claim_count"].replace({0: np.nan})
    agg["approval_rate"] = agg["approval_rate"].fillna(0.0)

    # Effectiveness proxy: approval × log volume × log amount
    agg["score"] = (
        agg["approval_rate"]
        * np.log1p(agg["claim_count"])
        * np.log1p(agg["total_claim_amount"].clip(lower=0))
    )

    # Ensure every PRODUCT_GROUP × TACTIC_TYPE seen in plans appears (even with zero claims)
    all_pairs = plans[["PRODUCT_GROUP", "TACTIC_TYPE"]].drop_duplicates()

    # Plan-side modes to backfill categorical features when claims are missing
    plan_modes = (
        plans.groupby(["PRODUCT_GROUP", "TACTIC_TYPE"], as_index=False)
        .agg(
            product_sub_group_mode=("PRODUCT_SUB_GROUP", most_frequent),
            brand_mode=("BRAND", most_frequent),
            channel_group_mode=("CHANNEL_GROUP", most_frequent),
            country_mode=("COUNTRY", most_frequent),
            territory_mode=("TERRITORY", most_frequent),
        )
    )

    base = all_pairs.merge(plan_modes, on=["PRODUCT_GROUP", "TACTIC_TYPE"], how="left")
    full = base.merge(agg, on=["PRODUCT_GROUP", "TACTIC_TYPE"], how="left", suffixes=("_plan", ""))

    # Coalesce categorical feature modes (prefer claims-derived, fallback to plan-derived, else UNKNOWN)
    mode_cols = [
        "product_sub_group_mode",
        "brand_mode",
        "channel_group_mode",
        "deal_type_mode",
        "country_mode",
        "territory_mode",
        "month_mode",
        "period_mode",
    ]
    for col in mode_cols:
        plan_col = f"{col}_plan"
        if plan_col in full.columns:
            full[col] = full[col].fillna(full[plan_col])
            full = full.drop(columns=[plan_col])
        full[col] = full[col].fillna("UNKNOWN")

    fill_zero_cols = [
        "claim_count",
        "total_claim_amount",
        "approved_count",
        "pending_count",
        "rejected_count",
        "approval_rate",
        "score",
        "mean_amended_deal_amount",
        "mean_adjustment_value_amount",
    ]
    for c in fill_zero_cols:
        full[c] = full[c].fillna(0.0)

    # Rank tactics per product (score==0 will still be ranked)
    full["score_rank"] = full.groupby("PRODUCT_GROUP")["score"].rank(ascending=False, method="first")
    return full.sort_values(["PRODUCT_GROUP", "score_rank"])


def pick_best(agg: pd.DataFrame) -> pd.DataFrame:
    top = agg[agg["score_rank"] == 1].copy()
    cols = [
        "PRODUCT_GROUP",
        "TACTIC_TYPE",
        "score",
        "approval_rate",
        "claim_count",
        "total_claim_amount",
        "approved_count",
        "pending_count",
        "rejected_count",
    ]
    return top[cols].sort_values("score", ascending=False)


def build_training_frame(merged: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select model features + labels from enriched claim rows."""
    feature_cols = [
        "TACTIC_TYPE",
        "PRODUCT_GROUP",
        "PRODUCT_SUB_GROUP",
        "BRAND",
        "CHANNEL_GROUP",
        "COUNTRY",
        "TERRITORY",
        "DEAL_TYPE",
        "MONTH",
        "PERIOD",
        "AMENDED_DEAL_AMOUNT",
        "ADJUSTMENT_VALUE_AMOUNT",
    ]
    # Add new features if available
    optional_cols = ["GL_CODE", "DEAL_STATUS", "ACTIVITY_TYPE", "DATA_SOURCE"]
    for col in optional_cols:
        if col in merged.columns:
            feature_cols.append(col)
    
    merged = merged.copy()
    merged["label"] = merged["CLAIM_STATUS"].apply(label_claim_status)
    merged = merged.dropna(subset=["label"])
    y = merged["label"].astype(int)
    X = merged[[c for c in feature_cols if c in merged.columns]]
    return X, y


def train_approval_model(X: pd.DataFrame, y: pd.Series) -> tuple[Pipeline, dict]:
    """Train logistic classifier with one-hot encoding to estimate approval probability."""
    # Identify categorical vs numeric columns dynamically
    num_cols = ["AMENDED_DEAL_AMOUNT", "ADJUSTMENT_VALUE_AMOUNT"]
    num_cols = [c for c in num_cols if c in X.columns]
    
    cat_cols = [c for c in X.columns if c not in num_cols]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    transformers = []
    if cat_cols:
        transformers.append(("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols))
    if num_cols:
        transformers.append(("num", StandardScaler(with_mean=False), num_cols))
    
    preprocessor = ColumnTransformer(transformers)

    clf = LogisticRegression(
        max_iter=400,
        class_weight="balanced",
        n_jobs=-1,
    )

    model = Pipeline([
        ("preprocess", preprocessor),
        ("clf", clf),
    ])

    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    try:
        roc_auc = float(roc_auc_score(y_val, y_proba))
    except ValueError:
        roc_auc = float("nan")

    metrics = {
        "accuracy": float(accuracy_score(y_val, y_pred)),
        "f1": float(f1_score(y_val, y_pred)),
        "roc_auc": roc_auc,
        "train_samples": int(len(X_train)),
        "val_samples": int(len(X_val)),
        "positive_rate_val": float(y_val.mean()),
        "features_used": list(X.columns),
    }
    return model, metrics


def make_candidate_features(agg: pd.DataFrame) -> pd.DataFrame:
    """Create feature rows for each PRODUCT_GROUP × TACTIC_TYPE aggregate."""
    base_features = {
        "TACTIC_TYPE": agg["TACTIC_TYPE"],
        "PRODUCT_GROUP": agg["PRODUCT_GROUP"],
        "PRODUCT_SUB_GROUP": agg.get("product_sub_group_mode", pd.Series(["UNKNOWN"] * len(agg))),
        "BRAND": agg.get("brand_mode", pd.Series(["UNKNOWN"] * len(agg))),
        "CHANNEL_GROUP": agg.get("channel_group_mode", pd.Series(["UNKNOWN"] * len(agg))),
        "COUNTRY": agg.get("country_mode", pd.Series(["UNKNOWN"] * len(agg))),
        "TERRITORY": agg.get("territory_mode", pd.Series(["UNKNOWN"] * len(agg))),
        "DEAL_TYPE": agg.get("deal_type_mode", pd.Series(["UNKNOWN"] * len(agg))),
        "MONTH": agg.get("month_mode", pd.Series(["UNKNOWN"] * len(agg))),
        "PERIOD": agg.get("period_mode", pd.Series(["UNKNOWN"] * len(agg))),
        "AMENDED_DEAL_AMOUNT": agg.get("mean_amended_deal_amount", pd.Series([0.0] * len(agg))),
        "ADJUSTMENT_VALUE_AMOUNT": agg.get("mean_adjustment_value_amount", pd.Series([0.0] * len(agg))),
    }
    
    # Add optional features if they exist in agg (from TDA data)
    optional_features = ["GL_CODE", "DEAL_STATUS", "ACTIVITY_TYPE", "DATA_SOURCE"]
    for feat in optional_features:
        mode_col = f"{feat.lower()}_mode"
        if mode_col in agg.columns:
            base_features[feat] = agg[mode_col]
        else:
            base_features[feat] = pd.Series(["UNKNOWN"] * len(agg))
    
    return pd.DataFrame(base_features)


def score_with_model(model: Pipeline, agg: pd.DataFrame) -> pd.DataFrame:
    """Add model-predicted approval probability and expected score to aggregates."""
    candidates = make_candidate_features(agg)
    proba = model.predict_proba(candidates)[:, 1]
    scored = agg.copy()
    scored["pred_approval_prob"] = proba
    scored["pred_expected_score"] = scored["pred_approval_prob"] * np.log1p(1 + scored["claim_count"]) * np.log1p(1 + scored["total_claim_amount"].clip(lower=0))
    scored["pred_rank"] = scored.groupby("PRODUCT_GROUP")["pred_expected_score"].rank(ascending=False, method="first")
    return scored


def pick_best_model(scored: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "PRODUCT_GROUP",
        "TACTIC_TYPE",
        "pred_approval_prob",
        "pred_expected_score",
        "score",
        "approval_rate",
        "claim_count",
        "total_claim_amount",
    ]
    top = scored[scored["pred_rank"] == 1].copy()
    return top[cols].sort_values("pred_expected_score", ascending=False)


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    print("=" * 80)
    print("TASK 3: Best Tactic per Product (ENHANCED: TACTIC + TDA data)")
    print("=" * 80)

    # Load TACTIC data
    print("\n📥 Loading TACTIC data...")
    plans, plans_raw = load_plans()
    claims, claims_raw = load_claims()
    tactic_merged = prepare_enriched_claims(plans, claims)
    
    # Load TDA data
    print("\n📥 Loading TDA SIP data...")
    tda_plans, tda_plans_raw = load_tda_plans()
    tda_claims, tda_claims_raw = load_tda_claims()
    tda_merged = prepare_tda_enriched_claims(tda_plans, tda_claims)
    
    # Combine data sources
    print("\n🔄 Combining data sources...")
    merged = combine_data_sources(tactic_merged, tda_merged)
    
    print(f"\n📊 DATA SUMMARY:")
    print(f"  TACTIC Plans (raw):       {plans_raw:,} rows")
    print(f"  TACTIC Plans (filtered):  {len(plans):,} rows")
    print(f"  TACTIC Claims (raw):      {claims_raw:,} rows")
    print(f"  TACTIC Claims (merged):   {len(tactic_merged):,} rows")
    print(f"  TDA Plans (raw):          {tda_plans_raw:,} rows")
    print(f"  TDA Plans (filtered):     {len(tda_plans):,} rows")
    print(f"  TDA Claims (raw):         {tda_claims_raw:,} rows")
    print(f"  TDA Claims (merged):      {len(tda_merged):,} rows")
    print(f"  COMBINED TOTAL:           {len(merged):,} rows")

    # Heuristic scoring (existing logic) - use combined plans
    all_plans = pd.concat([plans, tda_plans], ignore_index=True) if not tda_plans.empty else plans
    agg = build_scores(all_plans, merged)
    best = pick_best(agg)

    out_best = RESULTS_DIR / "task3_best_tactic_per_product.csv"
    out_full = RESULTS_DIR / "task3_tactic_scores_full.csv"

    best.to_csv(out_best, index=False)
    agg.to_csv(out_full, index=False)

    print(f"\nSaved heuristic best-per-product: {out_best} (rows={len(best)})")
    print(f"Saved heuristic full scores:     {out_full} (rows={len(agg)})")

    # Model-based approval probability
    X, y = build_training_frame(merged)
    label_counts = y.value_counts()
    if y.nunique() < 2 or label_counts.min() < 2:
        print("\n⚠️  Not enough label variety to train model; skipping model-based ranking.")
        model = None
    else:
        model, metrics = train_approval_model(X, y)
        metrics_path = RESULTS_DIR / "task3_model_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2))
        print("\n✅ Trained approval model")
        print(f"  accuracy={metrics['accuracy']:.3f} | f1={metrics['f1']:.3f} | roc_auc={metrics['roc_auc']:.3f} | pos_rate_val={metrics['positive_rate_val']:.3f}")
        print(f"  Metrics saved: {metrics_path}")

        agg_scored = score_with_model(model, agg)
        best_model = pick_best_model(agg_scored)

        out_best_model = RESULTS_DIR / "task3_best_tactic_per_product_model.csv"
        out_full_model = RESULTS_DIR / "task3_tactic_scores_model.csv"

        best_model.to_csv(out_best_model, index=False)
        agg_scored.to_csv(out_full_model, index=False)

        print(f"\nSaved model-ranked best-per-product: {out_best_model} (rows={len(best_model)})")
        print(f"Saved model-ranked full scores:     {out_full_model} (rows={len(agg_scored)})")

        print("\nTop 10 products by model expected score:")
        display_cols = [
            "PRODUCT_GROUP",
            "TACTIC_TYPE",
            "pred_approval_prob",
            "pred_expected_score",
            "score",
            "approval_rate",
        ]
        print(best_model.head(10)[display_cols].to_string(index=False))

    print("\nTop 10 products by heuristic score:")
    display_cols = [
        "PRODUCT_GROUP",
        "TACTIC_TYPE",
        "score",
        "approval_rate",
        "claim_count",
        "total_claim_amount",
    ]
    print(best.head(10)[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()
