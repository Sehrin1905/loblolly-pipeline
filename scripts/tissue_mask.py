"""Create a heuristic tissue envelope for visual QC before segmentation."""

import argparse
import numpy as np
from wood_stitch.image_io import read_image, write_image
from wood_stitch.tissue_mask import make_tissue_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("output")
    args = parser.parse_args()
    image, scale = read_image(args.image)
    write_image(args.output, make_tissue_mask(image).astype(np.uint8), scale)
