"""Write an optional undirected proximity graph, not wall-interface lengths."""

import argparse
from wood_stitch.image_io import read_image
from wood_stitch.adjacency import compute_adjacency
from wood_stitch.artifacts import write_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_path")
    parser.add_argument("out_path")
    parser.add_argument("--radius-um", required=True, type=float)
    args = parser.parse_args()
    labels, scale = read_image(args.labels_path)
    write_csv(args.out_path, compute_adjacency(labels, radius_um=args.radius_um, pixel_size_um=scale))
