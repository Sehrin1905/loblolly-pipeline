# src/wood_stitch/adjacency.py
import numpy as np
import pandas as pd


def compute_adjacency(labels: np.ndarray) -> pd.DataFrame:
    """
    Find all pairs of adjacent cell lumens and the length of their shared boundary.
    Fully vectorized — no per-cell loops.

    Returns a DataFrame with columns: label_a, label_b, boundary_length_px
    """
    print("  Finding horizontal adjacencies ...")
    # Compare each pixel to its right neighbor
    left = labels[:, :-1]
    right = labels[:, 1:]
    h_mask = (left != right) & (left > 0) & (right > 0)
    h_pairs = np.stack([left[h_mask], right[h_mask]], axis=1)

    print("  Finding vertical adjacencies ...")
    # Compare each pixel to its neighbor below
    top = labels[:-1, :]
    bottom = labels[1:, :]
    v_mask = (top != bottom) & (top > 0) & (bottom > 0)
    v_pairs = np.stack([top[v_mask], bottom[v_mask]], axis=1)

    print("  Combining ...")
    all_pairs = np.concatenate([h_pairs, v_pairs], axis=0)

    # Sort each pair so (a,b) and (b,a) are treated the same
    all_pairs = np.sort(all_pairs, axis=1)

    print("  Counting boundary lengths ...")
    df = pd.DataFrame(all_pairs, columns=["label_a", "label_b"])
    edge_df = df.groupby(["label_a", "label_b"]).size().reset_index(name="boundary_length_px")

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


if __name__ == "__main__":
    import sys

    labels_path = sys.argv[1] if len(sys.argv) > 1 else "data/labels.npy"
    out_path    = sys.argv[2] if len(sys.argv) > 2 else "data/adjacency.csv"

    print(f"Loading {labels_path} ...")
    labels = np.load(labels_path)

    print("Computing adjacency ...")
    edge_df = compute_adjacency(labels)

    edge_df.to_csv(out_path, index=False)
    print(f"Saved → {out_path}  ({len(edge_df)} adjacent pairs)")