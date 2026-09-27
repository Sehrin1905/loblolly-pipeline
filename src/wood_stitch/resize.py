"""Create a separate calibrated analysis image, preserving the full-resolution source."""

from pathlib import Path
import math
import cv2
from .image_io import read_image, write_image, read_annotations


def resize_mosaic(mosaic_path, factor, out_path=None, *, provenance=None):
    if not math.isfinite(factor) or not 0 < factor <= 1:
        raise ValueError("Analysis resize factor must be in (0, 1]")
    source = Path(mosaic_path)
    target = Path(out_path) if out_path else source.with_name("analysis.ome.tif")
    if target.resolve() == source.resolve():
        raise ValueError("Resize output must differ from the full-resolution source")
    image, (sx, sy) = read_image(source)
    height, width = image.shape[:2]
    shape = (max(1, round(width * factor)), max(1, round(height * factor)))
    resized = cv2.resize(image, shape, interpolation=cv2.INTER_AREA) if shape != (width, height) else image
    scale = (sx * width / shape[0], sy * height / shape[1])
    if provenance is None:
        provenance = read_annotations(source)
        provenance["standalone_resize"] = {"factor": factor, "effective_pixel_size_um": list(scale)}
    write_image(target, resized, scale, provenance)
    return target
