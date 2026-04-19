"""
evaluate.py
===========
Comprehensive model evaluation module.

Generates:
  - Classification report (precision / recall / F1)
  - Confusion matrix (raw + normalized)
  - ROC curve + AUC score
  - Precision-Recall curve + Average Precision
  - Feature importance ranking (for tree models)
  - High-risk customer segmentation
  - Business insight summary

All plots are saved to reports/figures/.
All metrics are returned as a structured dict for downstream use.
"""

import json
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.metrics import (
    RocCurveDisplay,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.logger import setup_logger

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)

# Plotting style
plt.rcParams.update({
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 120,
})
PALETTE = {
    "primary": "#4361EE",
    "secondary": "#F72585",
    "accent": "#7209B7",
    "success": "#06D6A0",
    "warning": "#FFB703",
    "background": "#F8F9FA",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(config_path: str = "config/config.yaml") -> dict:
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Hard predictions (0/1).
        y_prob: Predicted probabilities for the positive class.
        threshold: Decision threshold.

    Returns:
        Dict with metric name -> value.
    """
    roc_auc = roc_auc_score(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    report = classification_report(y_true, y_pred, output_dict=True)

    metrics = {
        "roc_auc": round(float(roc_auc), 4),
        "average_precision": round(float(avg_prec), 4),
        "accuracy": round(float(report["accuracy"]), 4),
        "precision": round(float(report["1"]["precision"]), 4),
        "recall": round(float(report["1"]["recall"]), 4),
        "f1": round(float(report["1"]["f1-score"]), 4),
        "threshold": threshold,
        "classification_report": report,
    }

    logger.info("─" * 50)
    logger.info("ROC-AUC          : %.4f", metrics["roc_auc"])
    logger.info("Average Precision : %.4f", metrics["average_precision"])
    logger.info("Accuracy         : %.4f", metrics["accuracy"])
    logger.info("Precision (Churn): %.4f", metrics["precision"])
    logger.info("Recall (Churn)   : %.4f", metrics["recall"])
    logger.info("F1 (Churn)       : %.4f", metrics["f1"])
    logger.info("─" * 50)

    return metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    plots_dir: Path,
) -> None:
    """Plot and save both raw and normalized confusion matrices."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Confusion Matrix — {model_name}", fontsize=14, fontweight="bold")

    for ax, normalize in zip(axes, [None, "true"]):
        cm = confusion_matrix(y_true, y_pred, normalize=normalize)
        fmt = ".2f" if normalize else "d"
        title = "Normalized" if normalize else "Raw counts"
        sns.heatmap(
            cm,
            annot=True,
            fmt=fmt,
            cmap="Blues",
            ax=ax,
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"],
            linewidths=0.5,
        )
        ax.set_title(title)
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")

    plt.tight_layout()
    path = plots_dir / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Confusion matrix saved -> %s", path)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    plots_dir: Path,
    roc_auc: float,
) -> None:
    """Plot and save the ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color=PALETTE["primary"], lw=2.5,
            label=f"AUC = {roc_auc:.4f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random")
    ax.fill_between(fpr, tpr, alpha=0.08, color=PALETTE["primary"])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"ROC Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = plots_dir / f"{model_name}_roc_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("ROC curve saved -> %s", path)


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str,
    plots_dir: Path,
    avg_precision: float,
) -> None:
    """Plot and save the Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(recall, precision, color=PALETTE["secondary"], lw=2.5,
            label=f"AP = {avg_precision:.4f}")
    ax.fill_between(recall, precision, alpha=0.08, color=PALETTE["secondary"])
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title(f"Precision-Recall Curve — {model_name}", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    plt.tight_layout()
    path = plots_dir / f"{model_name}_pr_curve.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("PR curve saved -> %s", path)


def plot_feature_importance(
    feature_names: List[str],
    importances: np.ndarray,
    model_name: str,
    plots_dir: Path,
    top_n: int = 20,
) -> None:
    """Plot and save top-N feature importances as a horizontal bar chart."""
    df_imp = pd.DataFrame({"feature": feature_names, "importance": importances})
    df_imp = df_imp.sort_values("importance", ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, max(5, top_n * 0.4)))
    bars = ax.barh(
        df_imp["feature"][::-1],
        df_imp["importance"][::-1],
        color=PALETTE["accent"],
        edgecolor="white",
        height=0.7,
    )
    ax.set_xlabel("Importance Score", fontsize=12)
    ax.set_title(f"Top {top_n} Feature Importances — {model_name}", fontsize=14, fontweight="bold")
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
    plt.tight_layout()
    path = plots_dir / f"{model_name}_feature_importance.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Feature importance chart saved -> %s", path)


def plot_model_comparison(
    metrics_dict: Dict[str, Dict],
    plots_dir: Path,
) -> None:
    """Bar chart comparing ROC-AUC, F1, Precision, Recall across models."""
    metric_keys = ["roc_auc", "f1", "precision", "recall"]
    models = list(metrics_dict.keys())
    x = np.arange(len(models))
    width = 0.2
    colors = [PALETTE["primary"], PALETTE["secondary"], PALETTE["accent"], PALETTE["success"]]

    fig, ax = plt.subplots(figsize=(11, 6))
    for i, (metric, color) in enumerate(zip(metric_keys, colors)):
        vals = [metrics_dict[m].get(metric, 0) for m in models]
        rects = ax.bar(x + i * width - 0.3, vals, width, label=metric.upper(), color=color, alpha=0.85)
        ax.bar_label(rects, fmt="%.2f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("_", " ").title() for m in models], fontsize=11)
    ax.set_ylim([0, 1.15])
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    path = plots_dir / "model_comparison.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Model comparison chart saved -> %s", path)


# ---------------------------------------------------------------------------
# Feature importance extraction
# ---------------------------------------------------------------------------

def extract_feature_importance(pipeline: Any, config: dict) -> Optional[pd.DataFrame]:
    """
    Extract feature importances from a pipeline's classifier step.

    Supports: XGBoost, RandomForest (feature_importances_),
              LogisticRegression (coef_).

    Returns:
        DataFrame with columns [feature, importance] sorted descending,
        or None if extraction fails.
    """
    try:
        from src.preprocessing import build_preprocessor, get_feature_names
        preprocessor = pipeline.named_steps["preprocessor"]
        feature_names = get_feature_names(preprocessor, config)
    except Exception as exc:
        logger.warning("Could not extract feature names: %s", exc)
        return None

    classifier = pipeline.named_steps["classifier"]
    try:
        if hasattr(classifier, "feature_importances_"):
            importances = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importances = np.abs(classifier.coef_[0])
        else:
            return None

        # Align lengths (OHE expansion can create extra names)
        min_len = min(len(feature_names), len(importances))
        df_imp = pd.DataFrame({
            "feature": feature_names[:min_len],
            "importance": importances[:min_len],
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df_imp

    except Exception as exc:
        logger.warning("Feature importance extraction failed: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Business insights
# ---------------------------------------------------------------------------

def generate_business_insights(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_prob: np.ndarray,
    df_importance: Optional[pd.DataFrame],
    config: dict,
    plots_dir: Path,
) -> Dict[str, Any]:
    """
    Generate business-facing insights:
      - Top churn drivers (from feature importance)
      - High-risk customer segment statistics
      - Revenue at risk estimation

    Args:
        X_test: Test features (original, unscaled).
        y_test: True labels.
        y_prob: Predicted churn probabilities.
        df_importance: Feature importance DataFrame (or None).
        config: Configuration dict.
        plots_dir: Directory to save plots.

    Returns:
        Dict of business insight summaries.
    """
    threshold = config["api"]["churn_threshold"]
    insights: Dict[str, Any] = {}

    # --- High-risk segment ---
    risk_df = X_test.copy()
    risk_df["churn_prob"] = y_prob
    risk_df["predicted_churn"] = (y_prob >= threshold).astype(int)
    risk_df["actual_churn"] = y_test.values

    high_risk = risk_df[risk_df["churn_prob"] >= 0.7]
    insights["high_risk_count"] = int(len(high_risk))
    insights["high_risk_pct"] = round(len(high_risk) / len(risk_df) * 100, 2)

    if "MonthlyCharges" in high_risk.columns:
        monthly_at_risk = high_risk["MonthlyCharges"].sum()
        insights["monthly_revenue_at_risk"] = round(float(monthly_at_risk), 2)

    logger.info("High-risk customers (prob ≥ 0.70): %d (%.1f%%)",
                insights["high_risk_count"], insights["high_risk_pct"])
    if "monthly_revenue_at_risk" in insights:
        logger.info("Monthly revenue at risk: $%.2f", insights["monthly_revenue_at_risk"])

    # --- Top churn drivers table ---
    if df_importance is not None:
        top_drivers = df_importance.head(10)
        insights["top_churn_drivers"] = top_drivers.to_dict(orient="records")
        logger.info("Top 3 churn drivers: %s", top_drivers["feature"].tolist()[:3])

    # --- Risk distribution plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(y_prob[y_test == 0], bins=40, alpha=0.6, label="No Churn", color=PALETTE["success"])
    ax.hist(y_prob[y_test == 1], bins=40, alpha=0.6, label="Churn", color=PALETTE["secondary"])
    ax.axvline(threshold, color="black", linestyle="--", lw=1.5, label=f"Threshold={threshold}")
    ax.set_xlabel("Predicted Churn Probability", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Churn Probability Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    path = plots_dir / "churn_probability_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Probability distribution saved -> %s", path)

    return insights


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def evaluate_model(
    pipeline: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str,
    config: dict,
    plots_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Full evaluation for a single pipeline on the test set.

    Returns:
        metrics dict
    """
    if plots_dir is None:
        plots_dir = Path(config["evaluation"]["plots_dir"])
    _ensure_dir(plots_dir)

    threshold = config["evaluation"]["threshold"]

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    metrics = compute_metrics(y_test.values, y_pred, y_prob, threshold)

    plot_confusion_matrix(y_test.values, y_pred, model_name, plots_dir)
    plot_roc_curve(y_test.values, y_prob, model_name, plots_dir, metrics["roc_auc"])
    plot_precision_recall_curve(y_test.values, y_prob, model_name, plots_dir, metrics["average_precision"])

    df_imp = extract_feature_importance(pipeline, config)
    if df_imp is not None:
        plot_feature_importance(
            df_imp["feature"].tolist(),
            df_imp["importance"].values,
            model_name,
            plots_dir,
        )

    return metrics


def run_evaluation(
    all_pipelines: Dict[str, Any],
    X_test: pd.DataFrame,
    y_test: pd.Series,
    best_name: str,
    config: dict,
    config_path: str = "config/config.yaml",
) -> Dict[str, Any]:
    """
    Evaluate all trained pipelines and generate business insights.

    Returns:
        all_metrics dict keyed by model name, plus business insights.
    """
    plots_dir = Path(config["evaluation"]["plots_dir"])
    _ensure_dir(plots_dir)

    all_metrics: Dict[str, Any] = {}

    for name, pipe in all_pipelines.items():
        logger.info("Evaluating: %s", name.upper())
        metrics = evaluate_model(pipe, X_test, y_test, name, config, plots_dir)
        all_metrics[name] = metrics

    # Model comparison chart
    plot_model_comparison(all_metrics, plots_dir)

    # Business insights for best model
    best_pipe = all_pipelines[best_name]
    y_prob_best = best_pipe.predict_proba(X_test)[:, 1]
    df_imp_best = extract_feature_importance(best_pipe, config)
    insights = generate_business_insights(
        X_test, y_test, y_prob_best, df_imp_best, config, plots_dir
    )
    all_metrics["business_insights"] = insights

    # Persist metrics
    reports_dir = Path("reports")
    reports_dir.mkdir(parents=True, exist_ok=True)
    with open(reports_dir / "evaluation_metrics.json", "w") as fh:
        json.dump(all_metrics, fh, indent=2, default=str)
    logger.info("Metrics saved -> reports/evaluation_metrics.json")

    return all_metrics


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.train import run_training

    results = run_training()
    metrics = run_evaluation(
        results["all_pipelines"],
        results["X_test"],
        results["y_test"],
        results["best_name"],
        results["config"],
    )
    print("\nDONE Evaluation complete.")
    print(f"   Best model ROC-AUC: {metrics[results['best_name']]['roc_auc']}")
