# /// script
# requires-python = ">=3.13"
# dependencies = ["opencv-python", "numpy", "cellpose"]
# ///
"""
Segment cell lumens using Cellpose.

Usage:
    uv run scripts/segment.py <mosaic_path> <mask_path> <out_path>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import cv2
import numpy as np
from wood_stitch.segment import segment

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mosaic_path")
    ap.add_argument("mask_path")
    ap.add_argument("out_path")
    ap.add_argument("--gpu", action="store_true")
    args = ap.parse_args()

    print(f"Loading {args.mosaic_path} ...")
    img = cv2.imread(args.mosaic_path)

    print(f"Loading {args.mask_path} ...")
    tissue_mask = cv2.imread(args.mask_path, cv2.IMREAD_GRAYSCALE) > 127

    print("Running segmentation ...")
    labels = segment(img, tissue_mask, gpu=args.gpu)

    np.save(args.out_path, labels)
    print(f"Saved → {args.out_path}  ({labels.max()} cells)")
