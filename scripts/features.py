"""Extract physical-unit lumen measurements from calibrated label OME-TIFF."""

import argparse
from wood_stitch.image_io import read_image
from wood_stitch.features import compute_features
from wood_stitch.artifacts import write_csv

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("labels_path")
    parser.add_argument("out_path")
    args = parser.parse_args()
    labels, scale = read_image(args.labels_path)
    write_csv(args.out_path, compute_features(labels, pixel_size_um=scale))
