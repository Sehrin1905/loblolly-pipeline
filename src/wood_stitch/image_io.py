"""Single-plane calibrated TIFF I/O. Arrays in the pipeline use BGR color order."""

import json
import math
from xml.etree import ElementTree as ET

import numpy as np
import tifffile

from .artifacts import atomic_path

UNIT_TO_UM = {"µm": 1.0, "μm": 1.0, "um": 1.0, "nm": 0.001, "mm": 1000.0, "m": 1e6}


def validate_scale(scale):
    if len(scale) != 2 or not all(math.isfinite(float(x)) and float(x) > 0 for x in scale):
        raise ValueError("Both physical pixel sizes must be finite positive values in µm")
    return tuple(float(x) for x in scale)


def tiff_info(path):
    with tifffile.TiffFile(path) as tif:
        if len(tif.series) != 1 or len(tif.pages) != 1:
            raise ValueError(f"Expected a single-plane TIFF: {path}")
        series = tif.series[0]
        if series.axes not in ("YX", "YXS") or (len(series.shape) == 3 and series.shape[2] != 3):
            raise ValueError(f"Unsupported axes/shape {series.axes}/{series.shape}: {path}")
        if not tif.ome_metadata:
            raise ValueError(f"Missing embedded OME calibration: {path}")
        root = ET.fromstring(tif.ome_metadata)
        pixels = next((e for e in root.iter() if e.tag.rsplit("}", 1)[-1] == "Pixels"), None)
        if pixels is None:
            raise ValueError(f"Missing OME Pixels element: {path}")
        scale = []
        for axis in "XY":
            unit = pixels.get(f"PhysicalSize{axis}Unit", "µm")  # OME schema default
            if unit not in UNIT_TO_UM:
                raise ValueError(f"Unsupported physical size unit {unit!r}: {path}")
            value = pixels.get(f"PhysicalSize{axis}")
            if value is None:
                raise ValueError(f"Missing PhysicalSize{axis}: {path}")
            scale.append(float(value) * UNIT_TO_UM[unit])
        sx, sy = validate_scale(scale)
        return {
            "shape": list(series.shape),
            "dtype": str(series.dtype),
            "pixel_size_x_um": sx,
            "pixel_size_y_um": sy,
        }


def read_image(path):
    info = tiff_info(path)
    image = tifffile.imread(path)
    if list(image.shape) != info["shape"]:
        raise ValueError(f"TIFF header/array mismatch: {path}")
    if image.ndim == 3:
        image = image[..., ::-1].copy()  # on-disk RGB -> processing BGR
    return image, (info["pixel_size_x_um"], info["pixel_size_y_um"])


def write_image(path, image, scale, provenance=None):
    sx, sy = validate_scale(scale)
    color = image.ndim == 3 and image.shape[-1] == 3
    if image.ndim != 2 and not color:
        raise ValueError("Only YX scalar or YX3 color images are supported")
    metadata = {
        "axes": "YXS" if color else "YX",
        "PhysicalSizeX": sx,
        "PhysicalSizeY": sy,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeYUnit": "µm",
    }
    if provenance:
        metadata["MapAnnotation"] = {
            "Namespace": "https://github.com/Sehrin1905/loblolly-pipeline/provenance/v1",
            "Value": {str(k): json.dumps(v, sort_keys=True) for k, v in provenance.items()},
        }
    pixels = image[..., ::-1] if color else image
    with atomic_path(path) as temp:
        tifffile.imwrite(
            temp,
            pixels,
            ome=True,
            bigtiff=True,
            tile=(256, 256),
            compression="deflate",
            photometric="rgb" if color else "minisblack",
            metadata=metadata,
        )
        written = tiff_info(temp)
        if written["shape"] != list(image.shape) or not np.allclose(
            [written["pixel_size_x_um"], written["pixel_size_y_um"]], [sx, sy]
        ):
            raise OSError(f"Output TIFF validation failed: {path}")


def read_annotations(path):
    """Read this project's provenance annotation without depending on the filename."""
    with tifffile.TiffFile(path) as tif:
        root = ET.fromstring(tif.ome_metadata)
    values = {}
    for annotation in root.iter():
        if annotation.tag.rsplit("}", 1)[-1] != "MapAnnotation":
            continue
        if annotation.get("Namespace") != "https://github.com/Sehrin1905/loblolly-pipeline/provenance/v1":
            continue
        for entry in annotation.iter():
            if entry.tag.rsplit("}", 1)[-1] == "M":
                values[entry.attrib["K"]] = json.loads(entry.text)
    return values
