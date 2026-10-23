# src/wood_stitch/classify.py
import pandas as pd


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

    Returns features_df with an added 'cell_type' column.
    """
    df = features_df.copy()
    if "area_um2" not in df:
        raise ValueError("Classification requires calibrated area_um2; regenerate legacy feature tables")
    median_area = df["area_um2"].median()

    df["cell_type"] = "tracheid"
    df["classification_method"] = "unvalidated_tracheid_large_object_rules_v4"

    # 10x the sample median (tunable). Set from one tear at 16x the median,
    # with the next largest object at 3.7x, and large earlywood lumens at
    # about 5.6x in another sample.
    is_large = df["area_um2"] > 10 * median_area
    df.loc[is_large, "cell_type"] = "large_object"

    print(f"  Median area: {median_area:.1f} µm²")
    print(f"  Large objects (possible tears or resin ducts): {is_large.sum()}")
    print(f"  Tracheids: {(df['cell_type'] == 'tracheid').sum()}")
    print("  Rays: not classified (unmeasured)")

    return df
