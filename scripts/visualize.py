# /// script
# requires-python = ">=3.13"
# dependencies = ["opencv-python", "numpy", "pandas"]
# ///
"""
Generate color-coded overlay of classified cells.

Usage:
    uv run scripts/visualize.py <mosaic_path> <labels_path> <classified_path> <out_path>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import cv2
import numpy as np
import pandas as pd
from wood_stitch.visualize import make_overlay

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mosaic_path")
    ap.add_argument("labels_path")
    ap.add_argument("classified_path")
    ap.add_argument("out_path")
    args = ap.parse_args()

    print(f"Loading {args.mosaic_path} ...")
    img = cv2.imread(args.mosaic_path)

    print(f"Loading {args.labels_path} ...")
    labels = np.load(args.labels_path)

    print(f"Loading {args.classified_path} ...")
    classified_df = pd.read_csv(args.classified_path)

    print("Building overlay ...")
    overlay = make_overlay(img, labels, classified_df)

    cv2.imwrite(args.out_path, overlay)
    print(f"Saved → {args.out_path}")
