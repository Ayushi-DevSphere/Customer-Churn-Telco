"""
preprocessing.py
================
Handles all data loading, cleaning, and transformation tasks for the
Telco Customer Churn dataset before feature engineering.

Key responsibilities:
  - Load raw CSV data
  - Fix data type issues (TotalCharges coercion)
  - Handle missing values
  - Encode categorical / binary variables
  - Scale numeric features
  - Build a reusable sklearn ColumnTransformer
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.logger import setup_logger

logger = setup_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_raw_data(path: str) -> pd.DataFrame:
    """
    Load raw Telco Churn CSV. Handles encoding issues gracefully.

    Args:
        path: Path to the raw CSV file.

    Returns:
        Raw DataFrame.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty or unreadable.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Raw data file not found: {path}")

    logger.info("Loading raw data from %s", path)
    try:
        df = pd.read_csv(path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding="latin-1")

    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")

    logger.info("Dataset shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Clean raw DataFrame:
      1. Strip column names
      2. Convert TotalCharges to numeric (coerce blanks -> NaN)
      3. Drop the customer ID column
      4. Encode binary target (Churn -> 0/1)

    Args:
        df: Raw DataFrame.
        config: Project configuration dict.

    Returns:
        Cleaned DataFrame.
    """
    logger.info("Cleaning data ...")
    df = df.copy()

    # Standardize column names
    df.columns = df.columns.str.strip()

    # Fix TotalCharges (often has ' ' strings in original dataset)
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Drop CustomerID – not a feature
    id_col = config["data"].get("id_column", "customerID")
    if id_col in df.columns:
        df.drop(columns=[id_col], inplace=True)
        logger.debug("Dropped ID column: %s", id_col)

    # Encode target
    target = config["data"]["target_column"]
    if target in df.columns:
        df[target] = df[target].map({"Yes": 1, "No": 0}).astype(int)
        logger.info("Target distribution:\n%s", df[target].value_counts())

    logger.info("Clean shape: %s", df.shape)
    return df


# ---------------------------------------------------------------------------
# Missing value imputation
# ---------------------------------------------------------------------------

def impute_missing(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Impute missing values using per-column strategy from config.

    Args:
        df: Cleaned DataFrame.
        config: Project configuration dict.

    Returns:
        DataFrame with no missing values.
    """
    logger.info("Imputing missing values ...")
    df = df.copy()

    num_features = config["preprocessing"]["numeric_features"]
    cat_features = config["preprocessing"]["categorical_features"]
    fill_cfg = config["preprocessing"]["fill_strategy"]

    missing_before = df.isnull().sum().sum()
    logger.info("Total missing cells before imputation: %d", missing_before)

    # Numeric: median
    for col in num_features:
        if col in df.columns and df[col].isnull().any():
            fill_val = df[col].median()
            df[col] = df[col].fillna(fill_val)
            logger.debug("Filled %s with median %.4f", col, fill_val)

    # Categorical: mode
    for col in cat_features:
        if col in df.columns and df[col].isnull().any():
            fill_val = df[col].mode()[0]
            df[col] = df[col].fillna(fill_val)
            logger.debug("Filled %s with mode '%s'", col, fill_val)

    missing_after = df.isnull().sum().sum()
    logger.info("Total missing cells after imputation: %d", missing_after)
    return df


# ---------------------------------------------------------------------------
# Sklearn ColumnTransformer (for model pipeline)
# ---------------------------------------------------------------------------

def build_preprocessor(config: dict) -> ColumnTransformer:
    """
    Build a reusable sklearn ColumnTransformer that:
      - Imputes + scales numeric features
      - Imputes + one-hot encodes categorical features
      - Passes binary features through unchanged

    This object is embedded inside the final model Pipeline.

    Args:
        config: Project configuration dict.

    Returns:
        Configured, unfitted ColumnTransformer.
    """
    num_features = config["preprocessing"]["numeric_features"]
    cat_features = config["preprocessing"]["categorical_features"]
    bin_features = config["preprocessing"].get("binary_features", [])

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, num_features),
            ("cat", categorical_pipeline, cat_features),
            ("bin", "passthrough", bin_features),
        ],
        remainder="drop",
    )

    logger.info(
        "Built preprocessor | numeric=%d | categorical=%d | binary=%d",
        len(num_features),
        len(cat_features),
        len(bin_features),
    )
    return preprocessor


# ---------------------------------------------------------------------------
# Feature name extraction (after fitting ColumnTransformer)
# ---------------------------------------------------------------------------

def get_feature_names(preprocessor: ColumnTransformer, config: dict) -> List[str]:
    """
    Extract human-readable feature names from a **fitted** ColumnTransformer.

    Args:
        preprocessor: Fitted ColumnTransformer.
        config: Project configuration dict.

    Returns:
        List of feature names in transformer output order.
    """
    num_features = config["preprocessing"]["numeric_features"]
    cat_features = config["preprocessing"]["categorical_features"]
    bin_features = config["preprocessing"].get("binary_features", [])

    # One-hot encoded names
    ohe: OneHotEncoder = preprocessor.named_transformers_["cat"]["encoder"]
    cat_names = list(ohe.get_feature_names_out(cat_features))

    return num_features + cat_names + bin_features


# ---------------------------------------------------------------------------
# Split helpers
# ---------------------------------------------------------------------------

def split_features_target(
    df: pd.DataFrame, config: dict
) -> Tuple[pd.DataFrame, pd.Series]:
    """Return X (features) and y (target) from cleaned DataFrame."""
    target = config["data"]["target_column"]
    X = df.drop(columns=[target])
    y = df[target]
    return X, y


# ---------------------------------------------------------------------------
# Save processed data
# ---------------------------------------------------------------------------

def save_processed(df: pd.DataFrame, config: dict) -> None:
    """Persist the processed DataFrame to disk."""
    out_path = Path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved processed data -> %s", out_path)


# ---------------------------------------------------------------------------
# End-to-end convenience function
# ---------------------------------------------------------------------------

def run_preprocessing(config_path: str = "config/config.yaml") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Orchestrate the full preprocessing flow:
      load -> clean -> impute -> save -> return

    Returns:
        (X, y) DataFrames.
    """
    config = load_config(config_path)
    raw_df = load_raw_data(config["data"]["raw_path"])
    clean_df = clean_data(raw_df, config)
    imputed_df = impute_missing(clean_df, config)
    save_processed(imputed_df, config)
    X, y = split_features_target(imputed_df, config)
    logger.info("Preprocessing complete. Features: %d | Samples: %d", X.shape[1], len(y))
    return X, y


if __name__ == "__main__":
    X, y = run_preprocessing()
    print(f"\nFeatures shape : {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
