# /// script
# requires-python = ">=3.13"
# dependencies = ["opencv-python", "numpy"]
# ///
"""
Stitch microscopy tiles into a single mosaic.

Usage:
    uv run scripts/stitch.py <tile_dir> [--out <output_path>]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
from wood_stitch.stitch import stitch

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tile_dir", help="Directory containing tile images")
    ap.add_argument("--out", default="mosaic.tif", help="Output mosaic path")
    args = ap.parse_args()
    stitch(args.tile_dir, args.out)
