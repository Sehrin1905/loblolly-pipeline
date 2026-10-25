"""Create a heuristic tissue envelope for visual QC before segmentation."""

import argparse
import numpy as np
from wood_stitch.image_io import read_image, write_image
from wood_stitch.tissue_mask import make_tissue_mask

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("output")
    parser.add_argument(
        "--mosaic-resize-factor",
        type=float,
        required=True,
        help="Resize factor the mosaic was made at (1.0 = full resolution, 0.5 = half)",
    )
    args = parser.parse_args()
    image, scale = read_image(args.image)
    mask_scale = args.mosaic_resize_factor / 0.5
    mask = make_tissue_mask(image, scale=mask_scale)
    write_image(args.output, mask.astype(np.uint8), scale)
