"""Segment a calibrated analysis image using an explicit pretrained Cellpose model."""

import argparse
import numpy as np
from wood_stitch.image_io import read_image, write_image
from wood_stitch.segment import create_model, segment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("mask")
    parser.add_argument("output")
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="auto")
    parser.add_argument("--model", default="cpsam_v2")
    parser.add_argument("--diameter", type=float)
    args = parser.parse_args()
    image, scale = read_image(args.image)
    mask, mask_scale = read_image(args.mask)
    if not np.allclose(scale, mask_scale):
        raise ValueError("Mask/image calibration differs")
    model, info = create_model(vars(args))
    labels = segment(image, mask.astype(bool), diameter=args.diameter, model=model)
    write_image(args.output, labels, scale, {"model": info})
