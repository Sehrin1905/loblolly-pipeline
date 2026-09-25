"""Experimental RGB unmixing; components are NOT validated safranin/astra-blue concentrations."""

import argparse
from pathlib import Path
from wood_stitch.image_io import read_image, write_image
from wood_stitch.artifacts import write_json
from wood_stitch.deconvolve import deconvolve

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image")
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    image, scale = read_image(args.image)
    results = deconvolve(image[..., ::-1])  # explicitly RGB
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for key in ("component_1", "component_2", "residual"):
        write_image(
            args.output_dir / f"{key}.ome.tif",
            results[key],
            scale,
            {"interpretation": "experimental unmixing component; biochemical identity unvalidated"},
        )
    write_json(
        args.output_dir / "unmixing.json",
        {
            "vectors": results["stain_vectors"].tolist(),
            "input_color_order": "RGB",
            "quantitative_stain_validation": "not established",
        },
    )
