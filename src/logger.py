"""
Centralized logging configuration for the Churn Prediction pipeline.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import yaml


def get_config(config_path: str = "config/config.yaml") -> dict:
    """Load and return yaml configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_logger(
    name: str,
    config_path: str = "config/config.yaml",
    log_level: Optional[str] = None,
) -> logging.Logger:
    """
    Create and configure a named logger with both console and file handlers.

    Args:
        name: Logger name (typically __name__ from the calling module).
        config_path: Path to YAML configuration file.
        log_level: Override log level (DEBUG, INFO, WARNING, ERROR).

    Returns:
        Configured Logger instance.
    """
    try:
        cfg = get_config(config_path)
        log_cfg = cfg.get("logging", {})
    except FileNotFoundError:
        log_cfg = {}

    # Resolve log level
    if log_level is None:
        log_level = log_cfg.get("level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)

    # Ensure log directory exists
    log_dir = log_cfg.get("log_dir", "logs")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = os.path.join(log_dir, log_cfg.get("log_file", "churn_pipeline.log"))

    fmt = log_cfg.get(
        "format", "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )
    formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")

    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)

    # Avoid adding duplicate handlers when module is reloaded
    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(numeric_level)
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(numeric_level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    logger.propagate = False
    return logger
