# /// script
# requires-python = ">=3.13"
# dependencies = ["opencv-python", "numpy"]
# ///
"""
Separate safranin and astra blue stains via Macenko color deconvolution.

Usage:
    uv run scripts/deconvolve.py <mosaic_path>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import cv2
import os
from wood_stitch.deconvolve import deconvolve

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mosaic_path")
    args = ap.parse_args()

    print(f"Loading {args.mosaic_path} ...")
    img = cv2.imread(args.mosaic_path)

    print("Running deconvolution ...")
    results = deconvolve(img)

    out_dir = os.path.dirname(args.mosaic_path)
    for key in ("safranin", "astra_blue", "residual"):
        channel = results[key]
        normalized = cv2.normalize(channel, None, 0, 255, cv2.NORM_MINMAX)
        out_path = os.path.join(out_dir, f"{key}.png")
        cv2.imwrite(out_path, normalized.astype(np.uint8))
        print(f"Saved → {out_path}")
