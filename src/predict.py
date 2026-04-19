"""
predict.py
==========
Batch and single-record inference module.

Responsibilities:
  - Load a persisted model pipeline
  - Accept raw customer data (dict or DataFrame)
  - Apply the same preprocessing + feature engineering
  - Return churn probability and label
  - Support high-risk customer segmentation on a batch

This module is used by both the CLI and the FastAPI endpoint.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd
import yaml

from src.feature_engineering import run_feature_engineering, update_config_with_engineered
from src.logger import setup_logger
from src.preprocessing import clean_data, impute_missing, load_config

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_path: Optional[str] = None, config: Optional[dict] = None) -> Any:
    """
    Load a fitted sklearn pipeline from disk.

    Args:
        model_path: Override model path. Falls back to config value.
        config: Pre-loaded config dict.

    Returns:
        Loaded pipeline object.

    Raises:
        FileNotFoundError: If the model file is missing.
    """
    if config is None:
        config = load_config()
    if model_path is None:
        model_path = config["api"]["model_path"]

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}\n"
            "Run `python -m src.train` to train and save a model."
        )
    model = joblib.load(path)
    logger.info("Model loaded from %s", path)
    return model


# ---------------------------------------------------------------------------
# Input normalization
# ---------------------------------------------------------------------------

def normalize_input(
    data: Union[Dict[str, Any], pd.DataFrame],
    config: dict,
) -> pd.DataFrame:
    """
    Convert raw user input into a DataFrame ready for the pipeline.

    The pipeline's internal ColumnTransformer handles further preprocessing,
    but we still need to apply feature engineering first (since those columns
    must exist before the transformer sees them).

    Args:
        data: Single record as dict or a batch DataFrame.
        config: Project configuration dict.

    Returns:
        Preprocessed DataFrame (feature-engineered, not yet scaled).
    """
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    elif isinstance(data, pd.DataFrame):
        df = data.copy()
    else:
        raise TypeError(f"Unsupported input type: {type(data)}")

    # Coerce numeric types that may arrive as strings
    numeric_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Impute missing values that may exist in live input
    df = impute_missing(df, config)

    # Derive engineered features
    df = run_feature_engineering(df, config=config)

    # Update config so ColumnTransformer knows about engineered columns
    config = update_config_with_engineered(config)

    return df


# ---------------------------------------------------------------------------
# Single prediction
# ---------------------------------------------------------------------------

def predict_single(
    data: Dict[str, Any],
    model: Optional[Any] = None,
    config: Optional[dict] = None,
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Predict churn probability for a single customer record.

    Args:
        data: Customer feature dict (raw, matching Telco column names).
        model: Pre-loaded pipeline (loaded if None).
        config: Pre-loaded config dict.
        config_path: Path to config file.

    Returns:
        {
            "churn_probability": float,
            "churn_label": bool,
            "risk_tier": str,       # "High" | "Medium" | "Low"
            "confidence": float,
        }
    """
    if config is None:
        config = load_config(config_path)
    if model is None:
        model = load_model(config=config)

    # Prepare input
    df = normalize_input(data, config)

    # Predict
    prob = float(model.predict_proba(df)[0, 1])
    threshold = config["api"]["churn_threshold"]
    label = prob >= threshold

    # Risk tier
    if prob >= 0.70:
        risk_tier = "High"
    elif prob >= 0.40:
        risk_tier = "Medium"
    else:
        risk_tier = "Low"

    confidence = prob if label else (1 - prob)

    result = {
        "churn_probability": round(prob, 4),
        "churn_label": bool(label),
        "churn_label_text": "Churn" if label else "No Churn",
        "risk_tier": risk_tier,
        "confidence": round(confidence, 4),
    }

    logger.info(
        "Prediction -> prob=%.4f | label=%s | risk=%s",
        prob, result["churn_label_text"], risk_tier
    )
    return result


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    df: pd.DataFrame,
    model: Optional[Any] = None,
    config: Optional[dict] = None,
    config_path: str = "config/config.yaml",
) -> pd.DataFrame:
    """
    Predict churn probabilities for a batch of customers.

    Args:
        df: DataFrame of raw customer records (matching Telco column names).
        model: Pre-loaded pipeline.
        config: Pre-loaded config dict.
        config_path: Fallback config path.

    Returns:
        Original DataFrame with added columns:
          churn_probability, churn_label, risk_tier
    """
    if config is None:
        config = load_config(config_path)
    if model is None:
        model = load_model(config=config)

    out_df = df.copy()
    proc_df = normalize_input(df, config)
    probs = model.predict_proba(proc_df)[:, 1]
    threshold = config["api"]["churn_threshold"]

    out_df["churn_probability"] = np.round(probs, 4)
    out_df["churn_label"] = (probs >= threshold).astype(int)
    out_df["risk_tier"] = pd.cut(
        probs,
        bins=[-np.inf, 0.40, 0.70, np.inf],
        labels=["Low", "Medium", "High"],
    )

    n_churn = out_df["churn_label"].sum()
    logger.info(
        "Batch prediction complete | n=%d | predicted_churn=%d (%.1f%%)",
        len(out_df), n_churn, n_churn / len(out_df) * 100
    )
    return out_df


# ---------------------------------------------------------------------------
# High-risk segmentation
# ---------------------------------------------------------------------------

def segment_high_risk(
    results_df: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Return the top-N highest-risk customers from batch results.

    Args:
        results_df: Output of predict_batch().
        top_n: Number of records to return.

    Returns:
        Subset DataFrame sorted by churn_probability descending.
    """
    high_risk = (
        results_df[results_df["risk_tier"] == "High"]
        .sort_values("churn_probability", ascending=False)
        .head(top_n)
    )
    logger.info(
        "High-risk segment: %d customers (top %d returned)",
        len(results_df[results_df["risk_tier"] == "High"]), len(high_risk)
    )
    return high_risk


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    # Example: predict on a single customer
    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0,
        "TotalCharges": 1020.0,
    }

    result = predict_single(sample_customer)
    print("\n📊 Single Prediction Result:")
    print(json.dumps(result, indent=2))
