# /// script
# requires-python = ">=3.13"
# dependencies = ["pandas"]
# ///
"""
Classify cells as tracheid, ray, or resin duct.

Usage:
    uv run scripts/classify.py <features_path> <out_path>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import pandas as pd
from wood_stitch.classify import classify_cells

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("features_path")
    ap.add_argument("out_path")
    args = ap.parse_args()

    print(f"Loading {args.features_path} ...")
    df = pd.read_csv(args.features_path)

    print("Classifying cells ...")
    df = classify_cells(df)

    df.to_csv(args.out_path, index=False)
    print(f"Saved → {args.out_path}")
