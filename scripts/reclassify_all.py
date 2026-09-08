# /// script
# requires-python = ">=3.13"
# dependencies = ["pandas", "numpy", "opencv-python"]
# ///
"""
Rerun classify and visualize for all samples with updated thresholds.

Usage:
    python3 scripts/reclassify_all.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
import numpy as np
import pandas as pd
from wood_stitch.classify import classify_cells
from wood_stitch.visualize import make_overlay

data_dir = Path("data")

for sample_dir in sorted(data_dir.iterdir()):
    features = sample_dir / "features.csv"
    labels_path = sample_dir / "labels.npy"
    mosaic_path = sample_dir / "mosaic.tif"

    if not features.exists():
        continue

    print(f"\nProcessing {sample_dir.name} ...")

    # Classify
    df = pd.read_csv(features)
    df = classify_cells(df)
    df.to_csv(sample_dir / "classified.csv", index=False)

    # Visualize
    if labels_path.exists() and mosaic_path.exists():
        img = cv2.imread(str(mosaic_path))
        labels = np.load(str(labels_path))
        overlay = make_overlay(img, labels, df)
        cv2.imwrite(str(sample_dir / "overlay.png"), overlay)
        print(f"  Overlay saved!")

print("\nAll samples reclassified!")
