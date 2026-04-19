"""
train.py
========
Training orchestrator for the Customer Churn Prediction pipeline.

Responsibilities:
  1. Load + preprocess + feature-engineer data
  2. Split into train / validation / test sets
  3. Build sklearn Pipelines for each candidate model
  4. Handle class imbalance (SMOTE + class_weight)
  5. Run GridSearchCV / cross-validation
  6. Select the best model by ROC-AUC
  7. Persist the trained pipeline and metadata

Models trained:
  - Logistic Regression  (baseline)
  - Random Forest
  - XGBoost              (primary candidate)
"""

import json
import os
import time
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
import yaml
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.feature_engineering import ENGINEERED_NUMERIC, run_feature_engineering, update_config_with_engineered
from src.logger import setup_logger
from src.preprocessing import (
    build_preprocessor,
    clean_data,
    impute_missing,
    load_config,
    load_raw_data,
    split_features_target,
)

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_splits(
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Full data loading + feature engineering + train/val/test splitting.

    Returns:
        X_train, X_val, X_test, y_train, y_val, y_test
    """
    cfg_data = config["data"]
    cfg_train = config["training"]
    random_state = config["project"]["random_state"]

    # Load & clean
    raw_df = load_raw_data(cfg_data["raw_path"])
    clean_df = clean_data(raw_df, config)
    imputed_df = impute_missing(clean_df, config)

    # Feature engineering
    eng_df = run_feature_engineering(imputed_df, config=config)
    config = update_config_with_engineered(config)

    X, y = split_features_target(eng_df, config)

    # First split: train+val vs test
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=cfg_data["test_size"],
        stratify=y,
        random_state=random_state,
    )

    # Second split: train vs val
    val_ratio = cfg_data.get("validation_size", 0.10) / (1 - cfg_data["test_size"])
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=val_ratio,
        stratify=y_trainval,
        random_state=random_state,
    )

    logger.info(
        "Split sizes -> train: %d | val: %d | test: %d",
        len(y_train), len(y_val), len(y_test)
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_candidate_models(config: dict, random_state: int) -> Dict[str, Any]:
    """Return unfitted estimator instances keyed by model name."""
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            solver="lbfgs",
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200,
            random_state=random_state,
            n_jobs=-1,
        ),
        "xgboost": XGBClassifier(
            n_estimators=200,
            random_state=random_state,
            eval_metric="logloss",
            n_jobs=-1,
        ),
    }


def get_param_grids(config: dict) -> Dict[str, dict]:
    """Return hyper-parameter grids (from config) keyed by model name."""
    cfg = config["training"]["models"]
    return {
        "logistic_regression": {
            "classifier__C": cfg["logistic_regression"]["C"],
            "classifier__class_weight": cfg["logistic_regression"]["class_weight"],
        },
        "random_forest": {
            "classifier__n_estimators": cfg["random_forest"]["n_estimators"],
            "classifier__max_depth": cfg["random_forest"]["max_depth"],
            "classifier__class_weight": cfg["random_forest"]["class_weight"],
        },
        "xgboost": {
            "classifier__max_depth": cfg["xgboost"]["max_depth"],
            "classifier__learning_rate": cfg["xgboost"]["learning_rate"],
            "classifier__scale_pos_weight": cfg["xgboost"]["scale_pos_weight"],
        },
    }


# ---------------------------------------------------------------------------
# Pipeline builder
# ---------------------------------------------------------------------------

def build_pipeline(
    estimator,
    preprocessor,
    use_smote: bool = True,
    random_state: int = 42,
) -> ImbPipeline:
    """
    Wrap preprocessor + optional SMOTE + classifier into an imbalanced-learn
    Pipeline so that SMOTE is only applied to training data inside each CV fold.

    Args:
        estimator: Unfitted sklearn classifier.
        preprocessor: ColumnTransformer (unfitted).
        use_smote: Whether to include SMOTE oversampling.
        random_state: RNG seed.

    Returns:
        Configured (unfitted) ImbPipeline.
    """
    steps = [("preprocessor", preprocessor)]
    if use_smote:
        steps.append(("smote", SMOTE(random_state=random_state)))
    steps.append(("classifier", estimator))
    return ImbPipeline(steps=steps)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    config: dict,
) -> Tuple[Dict[str, Any], Dict[str, float]]:
    """
    Train all candidate models with GridSearchCV.

    Returns:
        best_pipelines: dict of {model_name: fitted pipeline}
        cv_scores: dict of {model_name: best CV score}
    """
    random_state = config["project"]["random_state"]
    use_smote = config["training"]["use_smote"]
    cv_folds = config["training"]["cv_folds"]
    scoring = config["training"]["scoring_metric"]

    candidates = get_candidate_models(config, random_state)
    param_grids = get_param_grids(config)

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    best_pipelines: Dict[str, Any] = {}
    cv_scores: Dict[str, float] = {}

    for name, estimator in candidates.items():
        logger.info("=" * 60)
        logger.info("Training: %s", name.upper())
        t0 = time.time()

        preprocessor = build_preprocessor(config)
        pipeline = build_pipeline(estimator, preprocessor, use_smote, random_state)

        grid = param_grids.get(name, {})
        if grid:
            search = GridSearchCV(
                pipeline,
                param_grid=grid,
                cv=cv,
                scoring=scoring,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            search.fit(X_train, y_train)
            best_pipe = search.best_estimator_
            best_score = search.best_score_
            logger.info("Best params: %s", search.best_params_)
        else:
            pipeline.fit(X_train, y_train)
            best_pipe = pipeline
            best_score = 0.0  # fallback

        elapsed = time.time() - t0
        logger.info("CV %s = %.4f | Elapsed: %.1fs", scoring, best_score, elapsed)

        best_pipelines[name] = best_pipe
        cv_scores[name] = best_score

    return best_pipelines, cv_scores


# ---------------------------------------------------------------------------
# Model selection & persistence
# ---------------------------------------------------------------------------

def select_best_model(
    best_pipelines: Dict[str, Any],
    cv_scores: Dict[str, float],
) -> Tuple[str, Any]:
    """Select the model with the highest CV score."""
    best_name = max(cv_scores, key=cv_scores.get)
    logger.info(
        "Best model: %s (CV ROC-AUC = %.4f)", best_name, cv_scores[best_name]
    )
    return best_name, best_pipelines[best_name]


def save_models(
    best_pipelines: Dict[str, Any],
    best_name: str,
    cv_scores: Dict[str, float],
    config: dict,
) -> None:
    """Persist all trained pipelines and metadata to disk."""
    save_dir = Path(config["models"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    for name, pipe in best_pipelines.items():
        path = save_dir / f"{name}_pipeline.joblib"
        joblib.dump(pipe, path)
        logger.info("Saved pipeline: %s", path)

    # Save the best model under canonical name
    best_path = save_dir / config["models"]["best_model_name"]
    joblib.dump(best_pipelines[best_name], best_path)
    logger.info("Best model saved: %s", best_path)

    # Save CV score metadata
    meta = {
        "best_model": best_name,
        "cv_scores": cv_scores,
        "random_state": config["project"]["random_state"],
    }
    meta_path = save_dir / "training_metadata.json"
    with open(meta_path, "w") as fh:
        json.dump(meta, fh, indent=2)
    logger.info("Training metadata saved: %s", meta_path)


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def run_training(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """
    Full training pipeline orchestration.

    Returns:
        result dict with keys: X_test, y_test, best_name, best_pipeline, all_pipelines, config
    """
    config = load_config(config_path)
    logger.info("Starting training pipeline...")

    X_train, X_val, X_test, y_train, y_val, y_test = load_splits(config)

    best_pipelines, cv_scores = train_all_models(X_train, y_train, config)
    best_name, best_pipeline = select_best_model(best_pipelines, cv_scores)
    save_models(best_pipelines, best_name, cv_scores, config)

    logger.info("Training complete.")
    return {
        "X_test": X_test,
        "y_test": y_test,
        "X_val": X_val,
        "y_val": y_val,
        "best_name": best_name,
        "best_pipeline": best_pipeline,
        "all_pipelines": best_pipelines,
        "config": config,
    }


if __name__ == "__main__":
    results = run_training()
    print(f"\n✅ Best model: {results['best_name']}")
    print(f"   Test set size: {len(results['y_test'])}")
