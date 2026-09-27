"""
Create a separate calibrated analysis OME-TIFF without replacing the source mosaic.

Usage:
    uv run scripts/resize.py <mosaic_path> <factor>

Example:
    uv run scripts/resize.py data/Marsh_1_Recent/mosaic.ome.tif 0.5
"""

import argparse
from wood_stitch.resize import resize_mosaic

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("mosaic_path", help="Path to mosaic.tif")
    ap.add_argument("factor", type=float, help="Resize factor (e.g. 0.5 for half size)")
    ap.add_argument("--out", help="Separate output OME-TIFF; defaults to analysis.ome.tif")
    args = ap.parse_args()
    resize_mosaic(args.mosaic_path, args.factor, args.out)
