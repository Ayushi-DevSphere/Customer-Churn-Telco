"""
feature_engineering.py
=======================
Creates domain-driven features that add predictive signal beyond the raw
Telco columns.  All transforms are deterministic and reproducible.

New features created:
  ┌─────────────────────────────────┬───────────────────────────────────────────────────┐
  │ Feature                         │ Description                                       │
  ├─────────────────────────────────┼───────────────────────────────────────────────────┤
  │ tenure_bucket                   │ Ordinal tenure group (0-12 mo, 13-24 mo, ...)       │
  │ avg_monthly_spend               │ TotalCharges / max(tenure,1)                       │
  │ contract_group                  │ Month-to-month vs Long-term grouping              │
  │ engagement_score                │ Weighted count of adopted add-on services         │
  │ charges_per_service             │ MonthlyCharges / (num_services + 1)               │
  │ high_value_flag                 │ 1 if MonthlyCharges > 75th percentile             │
  │ long_tenure_flag                │ 1 if tenure > 24 months                          │
  │ is_month_to_month               │ Binary: Contract is Month-to-month               │
  └─────────────────────────────────┴───────────────────────────────────────────────────┘
"""

from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import yaml

from src.logger import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Individual feature creators
# ---------------------------------------------------------------------------

def add_tenure_bucket(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Bin tenure into labeled ordinal groups.

    Bins and labels are configurable via config.yaml ▸ feature_engineering.
    """
    bins = config["feature_engineering"]["tenure_bins"]
    labels = config["feature_engineering"]["tenure_labels"]

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        right=True,
        include_lowest=True,
    )
    # Ordinal encoding for the bucket
    bucket_order = {lbl: i for i, lbl in enumerate(labels)}
    df["tenure_bucket_code"] = df["tenure_bucket"].map(bucket_order).astype(int)
    df.drop(columns=["tenure_bucket"], inplace=True)
    logger.debug("tenure_bucket_code distribution:\n%s", df["tenure_bucket_code"].value_counts())
    return df


def add_avg_monthly_spend(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average monthly spend = TotalCharges / max(tenure, 1).
    Captures lifetime spend velocity independently of tenure length.
    """
    df["avg_monthly_spend"] = df["TotalCharges"] / df["tenure"].clip(lower=1)
    return df


def add_contract_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group contracts: Month-to-month (high churn risk) vs Long-term.
    Also creates a binary is_month_to_month flag.
    """
    contract_map = {
        "Month-to-month": 0,
        "One year": 1,
        "Two year": 2,
    }
    df["contract_group"] = df["Contract"].map(contract_map).fillna(0).astype(int)
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    return df


def add_engagement_score(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Composite feature: weighted count of adopted add-on services.

    A customer with more active services is presumed to be more engaged
    and (generally) less likely to churn.  Weights are configurable.
    """
    weights = config["feature_engineering"]["engagement_weights"]
    service_col_map = {
        "online_security": "OnlineSecurity",
        "online_backup": "OnlineBackup",
        "device_protection": "DeviceProtection",
        "tech_support": "TechSupport",
        "streaming_tv": "StreamingTV",
        "streaming_movies": "StreamingMovies",
    }

    score = pd.Series(0.0, index=df.index)
    for key, col in service_col_map.items():
        if col in df.columns:
            active = (df[col] == "Yes").astype(float)
            weight = weights.get(key, 1.0)
            score += active * weight
            logger.debug("Engagement: %s = weight %.1f", col, weight)

    df["engagement_score"] = score
    return df


def add_charges_per_service(df: pd.DataFrame) -> pd.DataFrame:
    """
    MonthlyCharges normalized by number of add-on services subscribed.
    Reveals whether a customer is paying proportionally for what they use.
    """
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    present_cols = [c for c in service_cols if c in df.columns]
    num_services = df[present_cols].apply(lambda col: (col == "Yes").astype(int)).sum(axis=1)
    df["num_services"] = num_services
    df["charges_per_service"] = df["MonthlyCharges"] / (num_services + 1)
    return df


def add_flag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create binary flag features for high-value and long-tenure customers."""
    charge_thresh = df["MonthlyCharges"].quantile(0.75)
    df["high_value_flag"] = (df["MonthlyCharges"] > charge_thresh).astype(int)
    df["long_tenure_flag"] = (df["tenure"] > 24).astype(int)
    logger.debug("high_value_flag threshold: %.2f", charge_thresh)
    return df


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_feature_engineering(
    df: pd.DataFrame,
    config: Optional[dict] = None,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Apply all feature engineering steps in sequence.

    Args:
        df: Cleaned, imputed DataFrame (includes raw columns + target).
        config: Pre-loaded config dict (optional, avoids re-reading file).
        config_path: Path to config YAML.

    Returns:
        DataFrame with additional engineered columns.
    """
    if config is None:
        config = load_config(config_path)

    df = df.copy()
    n_before = df.shape[1]

    logger.info("Starting feature engineering ...")

    df = add_tenure_bucket(df, config)
    df = add_avg_monthly_spend(df)
    df = add_contract_group(df)
    df = add_engagement_score(df, config)
    df = add_charges_per_service(df)
    df = add_flag_features(df)

    n_after = df.shape[1]
    logger.info(
        "Feature engineering complete. Columns added: %d (total: %d)",
        n_after - n_before, n_after
    )
    new_cols = [c for c in df.columns if c not in df.columns[:n_before]]
    logger.debug("New columns: %s", df.columns.tolist()[n_before:])

    return df


# ---------------------------------------------------------------------------
# Feature list helpers (for downstream ColumnTransformer updates)
# ---------------------------------------------------------------------------

ENGINEERED_NUMERIC = [
    "tenure_bucket_code",
    "avg_monthly_spend",
    "contract_group",
    "engagement_score",
    "num_services",
    "charges_per_service",
    "high_value_flag",
    "long_tenure_flag",
    "is_month_to_month",
]


def update_config_with_engineered(config: dict) -> dict:
    """
    Append engineered numeric features to the preprocessing numeric list
    so the ColumnTransformer scales them along with raw numerics.

    Args:
        config: Configuration dict (modified in place and returned).

    Returns:
        Updated configuration dict.
    """
    existing = set(config["preprocessing"]["numeric_features"])
    to_add = [f for f in ENGINEERED_NUMERIC if f not in existing]
    config["preprocessing"]["numeric_features"].extend(to_add)
    logger.debug("Added engineered features to numeric pipeline: %s", to_add)
    return config


if __name__ == "__main__":
    from src.preprocessing import load_config, load_raw_data, clean_data, impute_missing

    cfg = load_config()
    df_raw = load_raw_data(cfg["data"]["raw_path"])
    df_clean = clean_data(df_raw, cfg)
    df_imputed = impute_missing(df_clean, cfg)
    df_eng = run_feature_engineering(df_imputed, config=cfg)

    print("\nEngineered sample:")
    print(df_eng[ENGINEERED_NUMERIC].head())
    print("\nNaN check:", df_eng.isnull().sum().sum())
