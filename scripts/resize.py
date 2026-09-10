# /// script
# requires-python = ">=3.13"
# dependencies = ["opencv-python"]
# ///
"""
Resize a mosaic and update its metadata.json with the new effective pixel size.

Usage:
    uv run scripts/resize.py <mosaic_path> <factor>

Example:
    uv run scripts/resize.py data/Marsh_1_Recent/mosaic.tif 0.5
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from wood_stitch.resize import resize_mosaic

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mosaic_path", help="Path to mosaic.tif")
    ap.add_argument("factor", type=float, help="Resize factor (e.g. 0.5 for 50%)")
    args = ap.parse_args()
    resize_mosaic(args.mosaic_path, args.factor)