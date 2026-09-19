# src/wood_stitch/adjacency.py
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation


def compute_adjacency(labels: np.ndarray,
                      dilation_radius: int = 3) -> pd.DataFrame:
    """
    Find all pairs of neighboring cell lumens separated by walls.

    Uses dilation-based proximity — expands each lumen by dilation_radius
    pixels and finds which other lumens it overlaps with. This correctly
    detects neighbors across cell walls, unlike direct pixel contact.

    Args:
        labels: (H, W) int32 label map from segmentation
        dilation_radius: how many pixels to dilate each lumen (should be
                        slightly larger than typical wall thickness)

    Returns a DataFrame with columns: label_a, label_b, contact_length_px
    """
    from scipy.ndimage import generate_binary_structure

    print(f"  Computing adjacency with dilation radius {dilation_radius}px ...")

    unique_labels = np.unique(labels[labels > 0])
    struct = generate_binary_structure(2, 1)  # 4-connectivity structuring element

    # Build dilation structuring element of given radius
    from skimage.morphology import disk
    selem = disk(dilation_radius)

    pairs = []

    # Dilate each lumen and find which other lumens it overlaps
    for label in unique_labels:
        lumen = labels == label
        dilated = binary_dilation(lumen, structure=selem)

        # Find all other labels that overlap with the dilated lumen
        overlap = labels[dilated & ~lumen]
        neighbors = np.unique(overlap[overlap > 0])

        for neighbor in neighbors:
            if neighbor > label:  # avoid duplicates
                # Contact length = number of overlapping pixels
                neighbor_mask = labels == neighbor
                contact = (dilated & neighbor_mask).sum()
                pairs.append({
                    "label_a": int(label),
                    "label_b": int(neighbor),
                    "contact_length_px": int(contact)
                })

    if not pairs:
        print("  WARNING: No adjacent pairs found")
        return pd.DataFrame(columns=["label_a", "label_b", "contact_length_px"])

    edge_df = pd.DataFrame(pairs)
    print(f"  Found {len(edge_df)} adjacent pairs")
    return edge_df


def count_neighbors(edge_df: pd.DataFrame) -> pd.DataFrame:
    """
    Count how many neighbors each cell has, from the edge list.
    """
    a_counts = edge_df.groupby("label_a").size()
    b_counts = edge_df.groupby("label_b").size()
    neighbor_counts = a_counts.add(b_counts, fill_value=0).astype(int)
    return neighbor_counts.reset_index().rename(
        columns={"index": "label", 0: "n_neighbors"}
    )