"""Plot calibrated descriptive summaries from one completed pipeline run."""

import argparse
from pathlib import Path
import json
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wood_stitch.analysis import CLASSES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir", type=Path, help="Local completed run directory containing summaries/status"
    )
    args = parser.parse_args(argv)
    if json.loads((args.run_dir / "status.json").read_text())["state"] != "succeeded":
        raise ValueError("Only completed runs can be plotted")
    summary = pd.read_csv(args.run_dir / "sample_summary.csv")
    if (
        not {"site", "period", "tree_id"} <= set(summary)
        or summary[["site", "period", "tree_id"]].isna().any().any()
    ):
        raise ValueError("Provide reviewed sample_registry metadata and rerun before comparative plotting")
    if not set(summary["site"]) <= {"Marsh", "Reference"} or not set(summary["period"]) <= {"Old", "Recent"}:
        raise ValueError("Unknown site/period in summary")
    if summary["total_cells"].eq(0).any():
        raise ValueError("A sample contains no accepted cells; inspect QC before comparing specimens")
    out = args.run_dir / "figures"
    out.mkdir(exist_ok=True)
    values = summary.set_index("sample_id")[[kind + "_percent" for kind in CLASSES]]
    ax = values.plot.bar(stacked=True, figsize=(12, 6))
    ax.set_ylabel("Percent of all classified lumens (includes resin ducts)")
    ax.set_title("Exploratory rule-based classifications; not validated cell identities")
    ax.figure.tight_layout()
    ax.figure.savefig(out / "cell_type_proportions.png", dpi=150)
    plt.close(ax.figure)
    fig, ax = plt.subplots(figsize=(9, 5))
    # One point per specimen; no pooled-cell significance test or guessed pairing.
    for (site, period), frame in summary.groupby(["site", "period"]):
        ax.scatter(
            [f"{site} / {period}"] * len(frame), frame["mean_lumen_area_um2"], label=f"{site} / {period}"
        )
    ax.set_ylabel("Mean lumen area per specimen (µm²)")
    ax.set_title("Descriptive specimen means; pairing/statistical analysis not performed")
    fig.tight_layout()
    fig.savefig(out / "sample_lumen_area.png", dpi=150)
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
