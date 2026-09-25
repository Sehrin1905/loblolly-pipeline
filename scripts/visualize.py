"""Create a classification overlay without darkening unlabeled image pixels."""

import argparse
import pandas as pd
from wood_stitch.image_io import read_image
from wood_stitch.visualize import make_overlay
from wood_stitch.artifacts import write_preview

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("labels")
    parser.add_argument("cells")
    parser.add_argument("output")
    args = parser.parse_args()
    image, _ = read_image(args.image)
    labels, _ = read_image(args.labels)
    write_preview(args.output, make_overlay(image, labels, pd.read_csv(args.cells)))
