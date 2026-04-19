"""
download_data.py
================
Downloads the Telco Customer Churn dataset from the Kaggle public
URL (no API key required — direct download via requests).

Usage:
  python download_data.py

The file is saved to: data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import sys
import urllib.request
from pathlib import Path


DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
SAVE_PATH = Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"✅ Dataset already exists at {dest}")
        return

    print(f"Downloading dataset from:\n  {url}")
    print(f"Saving to: {dest}")

    try:
        urllib.request.urlretrieve(url, dest)
        size_kb = dest.stat().st_size / 1024
        print(f"✅ Download complete! ({size_kb:.1f} KB, {dest})")
    except Exception as exc:
        print(f"❌ Download failed: {exc}")
        print("\nManual download instructions:")
        print("  1. Go to: https://www.kaggle.com/datasets/blastchar/telco-customer-churn")
        print("  2. Download WA_Fn-UseC_-Telco-Customer-Churn.csv")
        print(f"  3. Place it at: {dest}")
        sys.exit(1)


if __name__ == "__main__":
    download(DATA_URL, SAVE_PATH)
