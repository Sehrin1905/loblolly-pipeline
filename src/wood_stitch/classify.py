# src/wood_stitch/classify.py
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from skimage.filters import threshold_multiotsu

ZONE_K = 30  # nearest tracheids used for each cell's neighborhood
ZONE_MIN_TRACHEIDS = 100  # below this, leave zones empty
ZONE_MIN_SCORE_RANGE = 0.05  # provisional: min spread (log-area) of zone_score


def add_zones(
    df: pd.DataFrame,
    k: int = ZONE_K,
    min_tracheids: int = ZONE_MIN_TRACHEIDS,
    min_score_range: float = ZONE_MIN_SCORE_RANGE,
) -> pd.DataFrame:
    """
    Add 'zone_score' and 'zone' columns (earlywood-like / transition /
    latewood-like). Zones are relative within the sample.

    zone_score: for each tracheid, the median log(area_um2) of its k nearest
    tracheids (the cell itself is excluded). Large objects are excluded from
    the neighborhood and get no score and no zone.

    If the sample has fewer than min_tracheids tracheids, or the scores are
    nearly constant, both columns are left empty (NaN) rather than forced.
    """
    for col in ("centroid_x_um", "centroid_y_um", "area_um2", "cell_type"):
        if col not in df:
            raise ValueError(f"Zone assignment requires column '{col}'")

    df = df.copy()
    df["zone_score"] = np.nan
    df["zone"] = np.nan
    df["zone"] = df["zone"].astype(object)

    is_trach = (df["cell_type"] == "tracheid").to_numpy()
    n = int(is_trach.sum())
    if n < max(min_tracheids, k + 2):
        print(f"  Zones: skipped (only {n} tracheids)")
        return df

    xy = df.loc[is_trach, ["centroid_x_um", "centroid_y_um"]].to_numpy()
    log_area = np.log(df.loc[is_trach, "area_um2"].to_numpy())

    # k+1 because the nearest point to each cell is the cell itself.
    _, idx = cKDTree(xy).query(xy, k=k + 1)
    score = np.median(log_area[idx[:, 1:]], axis=1)

    if not np.isfinite(score).all() or np.ptp(score) < min_score_range:
        print("  Zones: skipped (neighborhood scores nearly constant)")
        return df

    try:
        cuts = threshold_multiotsu(score, classes=3)
    except ValueError:
        print("  Zones: skipped (multi-Otsu could not find 2 cutoffs)")
        return df

    # Small neighborhood area = latewood-like, large = earlywood-like.
    labels = np.array(["latewood-like", "transition", "earlywood-like"], dtype=object)
    zone = labels[np.digitize(score, cuts)]

    df.loc[is_trach, "zone_score"] = score
    df.loc[is_trach, "zone"] = zone

    counts = df["zone"].value_counts()
    print(f"  Zones (k={k}): " + ", ".join(f"{z}={counts.get(z, 0)}" for z in labels[::-1]))
    return df


def classify_cells(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rule-based classification of segmented cell lumens.

    Rays are NOT classified. In these transverse sections rays are dark
    astra blue bands that Cellpose mostly does not segment, and shape rules
    (eccentricity, orientation) mislabel compressed tracheids as rays.
    Ray counts are unmeasured, not zero.

    Objects far larger than the sample median are labelled 'large_object'.
    They may be tears or holes in the tissue, or real resin ducts, and size
    alone cannot tell which. They stay in the table so they can be inspected,
    but they are not tracheids. Every other segmented cell is a tracheid.

    Tracheids also get a within-sample 'zone' (earlywood-like, transition,
    latewood-like) and a continuous 'zone_score'. See add_zones().

    Returns features_df with added 'cell_type', 'zone_score' and 'zone' columns.
    """
    df = features_df.copy()
    if "area_um2" not in df:
        raise ValueError("Classification requires calibrated area_um2; regenerate legacy feature tables")
    median_area = df["area_um2"].median()

    df["cell_type"] = "tracheid"
    df["classification_method"] = "unvalidated_tracheid_large_object_zones_v5"

    # 10x the sample median (tunable). Set from one tear at 16x the median,
    # with the next largest object at 3.7x, and large earlywood lumens at
    # about 5.6x in another sample.
    is_large = df["area_um2"] > 10 * median_area
    df.loc[is_large, "cell_type"] = "large_object"

    print(f"  Median area: {median_area:.1f} µm²")
    print(f"  Large objects (possible tears or resin ducts): {is_large.sum()}")
    print(f"  Tracheids: {(df['cell_type'] == 'tracheid').sum()}")
    print("  Rays: not classified (unmeasured)")

    df = add_zones(df)
    return df
