# /// script
# requires-python = ">=3.13"
# dependencies = ["opencv-python", "numpy", "scikit-image"]
# ///
"""
Compute tissue mask from a mosaic image.

Usage:
    uv run scripts/tissue_mask.py <mosaic_path> <out_path>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import cv2
import numpy as np
from wood_stitch.tissue_mask import make_tissue_mask

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mosaic_path")
    ap.add_argument("out_path")
    args = ap.parse_args()

    print(f"Loading {args.mosaic_path} ...")
    img = cv2.imread(args.mosaic_path)

    print("Computing tissue mask ...")
    mask = make_tissue_mask(img)

    cv2.imwrite(args.out_path, (mask * 255).astype(np.uint8))
    print(f"Saved → {args.out_path}")
