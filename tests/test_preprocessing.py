"""
tests/test_preprocessing.py
============================
Unit tests for the preprocessing module.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing import (
    build_preprocessor,
    clean_data,
    impute_missing,
    split_features_target,
)


# ── Fixtures ────────────────────────────────────────────────

@pytest.fixture
def config():
    with open("config/config.yaml") as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_df():
    """Minimal DataFrame mimicking Telco columns."""
    return pd.DataFrame({
        "customerID": ["A001", "A002", "A003"],
        "gender": ["Female", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0],
        "Partner": ["Yes", "No", "Yes"],
        "Dependents": ["No", "No", "Yes"],
        "tenure": [12, 34, 5],
        "PhoneService": ["Yes", "Yes", "No"],
        "MultipleLines": ["No", "Yes", "No phone service"],
        "InternetService": ["Fiber optic", "DSL", "No"],
        "OnlineSecurity": ["No", "Yes", "No internet service"],
        "OnlineBackup": ["Yes", "No", "No internet service"],
        "DeviceProtection": ["No", "Yes", "No internet service"],
        "TechSupport": ["No", "No", "No internet service"],
        "StreamingTV": ["Yes", "No", "No internet service"],
        "StreamingMovies": ["No", "No", "No internet service"],
        "Contract": ["Month-to-month", "One year", "Two year"],
        "PaperlessBilling": ["Yes", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check", "Bank transfer (automatic)"],
        "MonthlyCharges": [85.0, 65.5, 20.0],
        "TotalCharges": ["1020.0", "2227.0", " "],  # Note: string + blank
        "Churn": ["Yes", "No", "No"],
    })


# ── Tests ────────────────────────────────────────────────────

class TestCleanData:
    def test_drops_customer_id(self, sample_df, config):
        result = clean_data(sample_df, config)
        assert "customerID" not in result.columns

    def test_churn_encoded_as_int(self, sample_df, config):
        result = clean_data(sample_df, config)
        assert result["Churn"].dtype in [np.int64, np.int32, int]
        assert set(result["Churn"].unique()).issubset({0, 1})

    def test_total_charges_numeric(self, sample_df, config):
        result = clean_data(sample_df, config)
        assert pd.api.types.is_numeric_dtype(result["TotalCharges"])

    def test_blank_total_charges_becomes_nan(self, sample_df, config):
        result = clean_data(sample_df, config)
        assert result["TotalCharges"].isna().any()

    def test_shape_preserved(self, sample_df, config):
        result = clean_data(sample_df, config)
        assert len(result) == len(sample_df)


class TestImputeMissing:
    def test_no_missing_after_imputation(self, sample_df, config):
        cleaned = clean_data(sample_df, config)
        result = impute_missing(cleaned, config)
        assert result.isnull().sum().sum() == 0

    def test_numeric_imputed_with_median(self, sample_df, config):
        cleaned = clean_data(sample_df, config)
        # TotalCharges has one NaN
        assert cleaned["TotalCharges"].isna().any()
        result = impute_missing(cleaned, config)
        assert not result["TotalCharges"].isna().any()


class TestSplitFeaturesTarget:
    def test_split_returns_x_y(self, sample_df, config):
        cleaned = clean_data(sample_df, config)
        imputed = impute_missing(cleaned, config)
        X, y = split_features_target(imputed, config)
        assert "Churn" not in X.columns
        assert y.name == "Churn"
        assert len(X) == len(y)


class TestPreprocessor:
    def test_builds_without_error(self, config):
        preprocessor = build_preprocessor(config)
        assert preprocessor is not None

    def test_transforms_data(self, sample_df, config):
        from sklearn.pipeline import Pipeline
        cleaned = clean_data(sample_df, config)
        imputed = impute_missing(cleaned, config)
        X, y = split_features_target(imputed, config)
        preprocessor = build_preprocessor(config)
        X_transformed = preprocessor.fit_transform(X)
        assert X_transformed.shape[0] == len(X)
        assert X_transformed.shape[1] > 0
        # Should be all finite after scaling
        assert np.isfinite(X_transformed).all()
