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
    Estimate per-cell wall thickness.
    
    For each lumen, measures the mean distance from its outer boundary pixels
    to the nearest pixel of any other lumen. This is the actual wall thickness
    separating adjacent cells.
    """
    from scipy.ndimage import binary_dilation, distance_transform_edt

    all_lumens = labels > 0
    results = []

    for label in np.unique(labels[labels > 0]):
        lumen = labels == label

        # Get 1px outer boundary just outside this lumen
        dilated = binary_dilation(lumen)
        outer_boundary = dilated & ~lumen & ~all_lumens

        if not outer_boundary.any():
            continue

        # Distance from outer boundary to nearest other lumen
        other_lumens = all_lumens & ~lumen
        if not other_lumens.any():
            continue

        dist_to_other = distance_transform_edt(~other_lumens)
        wall_thickness = dist_to_other[outer_boundary].mean()
        results.append({"label": int(label), "wall_thickness": wall_thickness})

    return pd.DataFrame(results)


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


