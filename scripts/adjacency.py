# /// script
# requires-python = ">=3.13"
# dependencies = ["numpy", "pandas"]
# ///
"""
Compute cell adjacency graph from label map.

Usage:
    uv run scripts/adjacency.py <labels_path> <out_path>
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import argparse
import numpy as np
from wood_stitch.adjacency import compute_adjacency

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("labels_path")
    ap.add_argument("out_path")
    args = ap.parse_args()

    print(f"Loading {args.labels_path} ...")
    labels = np.load(args.labels_path)

    print("Computing adjacency ...")
    edge_df = compute_adjacency(labels)

    edge_df.to_csv(args.out_path, index=False)
    print(f"Saved → {args.out_path}  ({len(edge_df)} adjacent pairs)")
