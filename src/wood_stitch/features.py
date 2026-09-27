"""Calibrated lumen morphometrics. Wall thickness requires a validated wall model."""

import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from .image_io import validate_scale


def validate_labels(labels):
    if labels.ndim != 2 or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Labels must be a nonnegative integer YX array")


def compute_morphometrics(labels, pixel_size_um):
    validate_labels(labels)
    scale = (pixel_size_um, pixel_size_um) if np.isscalar(pixel_size_um) else pixel_size_um
    sx, sy = validate_scale(scale)
    props = regionprops_table(
        labels,
        spacing=(sy, sx),
        properties=(
            "label",
            "area",
            "centroid",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "eccentricity",
            "solidity",
            "equivalent_diameter_area",
        ),
    )
    df = pd.DataFrame(props).rename(
        columns={
            "area": "area_um2",
            "centroid-0": "centroid_y_um",
            "centroid-1": "centroid_x_um",
            "major_axis_length": "major_axis_length_um",
            "minor_axis_length": "minor_axis_length_um",
            "orientation": "orientation_rad",
            "equivalent_diameter_area": "equivalent_diameter_um",
        }
    )
    df["area_px2"] = df["area_um2"] / (sx * sy)
    df["centroid_x_px"] = df["centroid_x_um"] / sx
    df["centroid_y_px"] = df["centroid_y_um"] / sy
    df["pixel_size_x_um"] = sx
    df["pixel_size_y_um"] = sy
    return df


def compute_wall_thickness(labels):
    raise NotImplementedError("Lumen spacing is not wall thickness; this unvalidated metric was removed")


def compute_features(labels, pixel_size_um=None):
    if pixel_size_um is None:
        raise ValueError("Physical X/Y calibration is required for quantitative features")
    return compute_morphometrics(labels, pixel_size_um)
