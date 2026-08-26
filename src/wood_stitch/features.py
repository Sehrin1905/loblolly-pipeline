# src/wood_stitch/features.py
import numpy as np
import pandas as pd
from skimage.measure import regionprops_table
from scipy.ndimage import distance_transform_edt, maximum_filter


def compute_morphometrics(labels: np.ndarray) -> pd.DataFrame:
    """
    Vectorized morphometric features for every labeled cell lumen.
    """
    props = regionprops_table(
        labels,
        properties=(
            "label",
            "area",
            "centroid",
            "major_axis_length",
            "minor_axis_length",
            "orientation",
            "eccentricity",
            "solidity",
            "equivalent_diameter",
        )
    )
    df = pd.DataFrame(props)
    df = df.rename(columns={
        "centroid-0": "centroid_y",
        "centroid-1": "centroid_x",
    })
    return df


def compute_wall_thickness(labels: np.ndarray) -> pd.DataFrame:
    """
    Estimate per-cell wall thickness using a distance-transform trick.
    Walls are the space between lumens — Cellpose doesn't segment them directly.

    1. distance_transform_edt(~lumen_mask) gives, at every wall pixel,
       the distance to the nearest lumen edge.
    2. maximum_filter pushes that distance one pixel inward so each lumen's
       boundary pixels pick up the wall distance just outside them.
    3. Group by label and average to get a per-cell wall thickness estimate.
    """
    lumen_mask = labels > 0

    # Distance from every background (wall) pixel to nearest lumen
    dist = distance_transform_edt(~lumen_mask)

    # Push the distance value 1px inward into the lumen boundary
    dist_dilated = maximum_filter(dist, size=3)

    # Only keep values right at lumen boundaries
    boundary_vals = np.where(lumen_mask, dist_dilated, 0)

    df = pd.DataFrame({
        "label": labels[lumen_mask],
        "wall_dist": boundary_vals[lumen_mask]
    })

    wall_thickness = df.groupby("label")["wall_dist"].mean().reset_index()
    wall_thickness = wall_thickness.rename(columns={"wall_dist": "wall_thickness"})
    return wall_thickness


def compute_features(labels: np.ndarray) -> pd.DataFrame:
    """
    Full per-cell feature table: morphometrics + wall thickness.
    """
    print("  Computing morphometrics ...")
    morph_df = compute_morphometrics(labels)

    print("  Computing wall thickness ...")
    wall_df = compute_wall_thickness(labels)

    print("  Merging ...")
    df = morph_df.merge(wall_df, on="label", how="left")
    return df


