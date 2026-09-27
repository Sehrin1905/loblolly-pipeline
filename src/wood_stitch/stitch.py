"""Strict calibrated stitching: every declared tile must be used."""

from pathlib import Path
import cv2
import numpy as np
from .artifacts import write_json
from .image_io import tiff_info, read_image, write_image
from .storage import is_sidecar


def read_pixel_size(path):
    info = tiff_info(path)
    if not np.isclose(info["pixel_size_x_um"], info["pixel_size_y_um"]):
        raise ValueError("Scalar calibration requested for anisotropic pixels")
    return info["pixel_size_x_um"]


def tile_paths(directory):
    return sorted(
        p
        for p in Path(directory).rglob("*")
        if p.is_file() and p.suffix.lower() in (".tif", ".tiff") and not is_sidecar(p.as_posix())
    )


def stitch(tile_dir, out_path="mosaic.ome.tif", *, paths=None, max_input_gb=2.0, provenance=None):
    paths = list(paths) if paths is not None else tile_paths(tile_dir)
    if not paths:
        raise ValueError("No TIFF tiles supplied")
    infos = [tiff_info(p) for p in paths]
    scales = [(i["pixel_size_x_um"], i["pixel_size_y_um"]) for i in infos]
    if any(not np.allclose(s, scales[0], rtol=1e-6, atol=0) for s in scales):
        raise ValueError("Tiles have inconsistent physical calibration")
    if not np.isclose(*scales[0], rtol=1e-6, atol=0):
        raise ValueError("Stitching anisotropic pixels needs a calibrated resampling step")
    if any(i["dtype"] != "uint8" or len(i["shape"]) != 3 for i in infos):
        raise ValueError("Stitching currently requires single-plane uint8 RGB TIFFs")
    decoded_bytes = sum(int(np.prod(i["shape"])) for i in infos)
    if decoded_bytes > max_input_gb * 1e9:
        raise MemoryError(
            f"Tiles alone need {decoded_bytes / 1e9:.2f} GB; input limit is {max_input_gb} GB. "
            "Choose a smaller pilot or explicitly raise the limit on a profiled machine."
        )
    images = [read_image(p)[0] for p in paths]
    coverage = {
        "expected": [p.name for p in paths],
        "input_files": [str(p) for p in paths],
        "used": [],
        "excluded": [],
        "complete": False,
        "transforms": [],
    }
    if len(images) == 1:
        mosaic = images[0]
        used = [0]
    else:
        st = cv2.Stitcher.create(cv2.Stitcher_SCANS)
        st.setCompositingResol(-1)  # do not silently downsample at rendering time
        status, mosaic = st.stitch(images)
        used = list(map(int, st.component())) if status == cv2.Stitcher_OK else []
        for cam in st.cameras() if status == cv2.Stitcher_OK else []:
            matrix = np.asarray(cam.R)
            coverage["transforms"].append({"R": matrix.tolist(), "K": cam.K().tolist()})
            singular = np.linalg.svd(matrix[:2, :2], compute_uv=False)
            if not np.allclose(singular, 1.0, atol=0.01, rtol=0):
                write_json(Path(out_path).with_name("coverage.json"), coverage)
                raise ValueError(
                    "Registration changes physical scale by more than 1%; review tile transforms"
                )
        if status != cv2.Stitcher_OK:
            write_json(Path(out_path).with_name("coverage.json"), coverage)
            raise RuntimeError(f"Stitching failed with status code {status}")
    coverage["used"] = [paths[i].name for i in used]
    coverage["excluded"] = [paths[i].name for i in range(len(paths)) if i not in used]
    coverage["complete"] = len(used) == len(paths)
    write_json(Path(out_path).with_name("coverage.json"), coverage)
    if not coverage["complete"]:
        raise ValueError(f"Incomplete mosaic: excluded tiles {coverage['excluded']}")
    write_image(out_path, mosaic, scales[0], provenance)
    return coverage
