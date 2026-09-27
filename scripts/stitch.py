"""
Stitch microscopy tiles into a single mosaic.

Usage:
    uv run scripts/stitch.py <tile_dir> [--out <output_path>]
"""

import argparse
from wood_stitch.stitch import stitch

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("tile_dir", help="Directory containing tile images")
    ap.add_argument("--out", default="mosaic.ome.tif", help="Output mosaic path")
    args = ap.parse_args()
    stitch(args.tile_dir, args.out)
