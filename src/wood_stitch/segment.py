"""Explicit Cellpose model/device selection. Inference remains whole-image."""

from pathlib import Path
import logging
import numpy as np
from .artifacts import sha256_file


def create_model(settings):
    from cellpose import models
    import torch

    requested = settings.get("device", "auto")
    if requested == "auto":
        requested = (
            "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but unavailable")
    if requested == "cuda":
        major, minor = torch.cuda.get_device_capability()
        arch = f"sm_{major}{minor}"
        if arch not in torch.cuda.get_arch_list():
            raise RuntimeError(
                f"This torch build does not list {arch}; install the locked CUDA 12.6 build for V100"
            )
    name = settings.get("model", "cpsam_v2")
    if name not in models.MODEL_NAMES and not Path(name).is_file():
        raise ValueError(f"Unknown Cellpose model: {name}")
    model = models.CellposeModel(device=torch.device(requested), pretrained_model=name, use_bfloat16=False)
    info = {
        "model": name,
        "weights_sha256": sha256_file(model.pretrained_model),
        "device": str(model.device),
        "precision": "float32",
        "torch": torch.__version__,
    }
    logging.getLogger(__name__).info("Loaded segmentation model: %s", info)
    return model, info


def segment(img, tissue_mask, bsize=256, gpu=False, diameter=None, *, model=None):
    """BGR uint8 image -> lumen labels. No claim of wood-specific model validation."""
    if img.ndim != 3 or img.shape[-1] != 3 or img.dtype != np.uint8 or tissue_mask.shape != img.shape[:2]:
        raise ValueError("Expected a uint8 BGR image and matching YX tissue mask")
    if not tissue_mask.any():
        return np.zeros(tissue_mask.shape, np.uint32)
    if diameter is not None and (not np.isfinite(diameter) or diameter <= 0):
        raise ValueError("Diameter must be positive and finite, in pixels")
    if model is None:
        model, _ = create_model({"device": "auto" if gpu else "cpu"})
    image = img[..., ::-1].copy()  # explicit RGB for the model
    image[~tissue_mask] = 255
    masks, _, _ = model.eval(image, diameter=diameter, bsize=bsize, channel_axis=-1)
    masks = np.asarray(masks)
    if masks.shape != tissue_mask.shape or masks.dtype.kind not in "iu" or np.any(masks < 0):
        raise ValueError("Cellpose returned an invalid label image")
    # Exclude entire objects crossing the accepted tissue or image boundary; do not measure clipped lumens.
    rejected = np.unique(
        np.concatenate([masks[~tissue_mask], masks[0], masks[-1], masks[:, 0], masks[:, -1]])
    )
    result = masks.astype(np.uint32, copy=True)
    result[np.isin(result, rejected[rejected > 0])] = 0
    return result
