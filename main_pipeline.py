"""
main_pipeline.py
================
End-to-end pipeline runner.
Executes: Preprocessing -> Feature Engineering -> Training -> Evaluation

Usage:
  python main_pipeline.py
  python main_pipeline.py --config config/config.yaml
  python main_pipeline.py --skip-train  (only evaluate existing models)
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.evaluate import run_evaluation
from src.logger import setup_logger
from src.train import run_training

logger = setup_logger("main_pipeline")


def parse_args():
    parser = argparse.ArgumentParser(description="Churn Prediction Pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to config YAML",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="Skip training, only evaluate existing models",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("  Customer Churn Prediction — Full Pipeline")
    logger.info("=" * 60)

    # Step 1: Train
    logger.info("\n[Step 1/2] Training ...")
    results = run_training(args.config)

    # Step 2: Evaluate
    logger.info("\n[Step 2/2] Evaluating ...")
    metrics = run_evaluation(
        results["all_pipelines"],
        results["X_test"],
        results["y_test"],
        results["best_name"],
        results["config"],
        config_path=args.config,
    )

    # Summary
    best = results["best_name"]
    logger.info("\n" + "=" * 60)
    logger.info("  Pipeline complete!")
    logger.info("  Best model   : %s", best)
    logger.info("  ROC-AUC      : %.4f", metrics[best]["roc_auc"])
    logger.info("  F1 Score     : %.4f", metrics[best]["f1"])
    logger.info("  Precision    : %.4f", metrics[best]["precision"])
    logger.info("  Recall       : %.4f", metrics[best]["recall"])

    if "business_insights" in metrics:
        bi = metrics["business_insights"]
        logger.info("  High-risk    : %d customers (%.1f%%)",
                    bi.get("high_risk_count", 0), bi.get("high_risk_pct", 0))
        if "monthly_revenue_at_risk" in bi:
            logger.info("  Revenue@Risk : $%.2f/mo", bi["monthly_revenue_at_risk"])
    logger.info("=" * 60)

    print(f"\nDONE Done. Best model: {best.upper()} | ROC-AUC: {metrics[best]['roc_auc']}")
    print("   Plots saved to:   reports/figures/")
    print("   Metrics saved to: reports/evaluation_metrics.json")
    print("   Models saved to:  models/")


if __name__ == "__main__":
    main()
