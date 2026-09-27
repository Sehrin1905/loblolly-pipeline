"""Undirected lumen proximity edges; no inferred wall-interface length."""

import math
import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation, find_objects
from .features import validate_labels
from .image_io import validate_scale


def compute_adjacency(labels, *, radius_um, pixel_size_um):
    validate_labels(labels)
    sx, sy = validate_scale(pixel_size_um)
    if not math.isfinite(radius_um) or radius_um <= 0:
        raise ValueError("Proximity radius must be positive and finite")
    rx, ry = math.ceil(radius_um / sx), math.ceil(radius_um / sy)
    if max(rx, ry) > 64:
        raise ValueError("Proximity radius exceeds 64 pixels; use an explicitly reviewed spatial method")
    yy, xx = np.ogrid[-ry : ry + 1, -rx : rx + 1]
    structure = (xx * sx) ** 2 + (yy * sy) ** 2 <= radius_um**2 + 1e-12
    # Relabel sparse IDs before find_objects so an arbitrary high label cannot allocate huge lists.
    ids, inverse = np.unique(labels, return_inverse=True)
    dense = inverse.reshape(labels.shape).astype(np.int32) + 1
    dense[labels == 0] = 0
    pairs = set()
    for dense_id, box in enumerate(find_objects(dense), 1):
        if box is None:
            continue
        label = int(ids[dense_id - 1])
        if label == 0:
            continue
        y, x = box
        roi = labels[
            max(0, y.start - ry) : min(labels.shape[0], y.stop + ry),
            max(0, x.start - rx) : min(labels.shape[1], x.stop + rx),
        ]
        expanded = binary_dilation(roi == label, structure=structure)
        for other in np.unique(roi[expanded]):
            if other > label:
                pairs.add((label, int(other)))
    return pd.DataFrame(sorted(pairs), columns=["label_a", "label_b"])


def count_neighbors(edge_df, labels=None):
    counts = pd.concat([edge_df["label_a"], edge_df["label_b"]]).value_counts()
    if labels is not None:
        counts = counts.reindex(np.unique(labels[labels > 0]), fill_value=0)
    return counts.rename_axis("label").rename("n_neighbors").reset_index()
