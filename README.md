# Loblolly microscopy pipeline

This project stitches calibrated microscopy TIFFs, segments lumen candidates with a pretrained Cellpose model, and exports physical-unit measurements and exploratory cell-type predictions. It is **not yet a biologically validated classifier**. Annotated wood sections and a reviewed experimental design are needed before interpreting site/period differences.

## Installation

Supported locked environments: Python 3.13 on Apple Silicon macOS and Linux x86-64. Install [uv](https://docs.astral.sh/uv/getting-started/installation/), clone this repository, and run commands from its root.

```bash
uv sync --locked                  # tools, plotting, and tests; no neural-network runtime
uv run --locked pytest -q
uv sync --locked --extra inference  # adds pinned Cellpose and PyTorch
```

All scripts use the project environment and `uv.lock`. There are no separate inline-script dependency environments. Linux inference resolves PyTorch/torchvision from the explicit CUDA 12.6 index; the default CUDA 13 builds do not support V100/Volta. See the [PyTorch architecture matrix](https://github.com/pytorch/pytorch/blob/main/RELEASE.md). The actual cluster driver, device and model still need a pilot inference run. The Mac uses the standard Apple Silicon PyTorch wheel and MPS when available.

## First run

1. Copy `.env.example` to `.env` and fill in the previously supplied bucket-scoped credentials. Keep `.env` out of Git. The pipeline bucket comes from `config.toml`; the upload script reads `R2_BUCKET`.
2. Set `cache.local_cache_dir` to a directory you own. Set `pipeline.samples` to **one** real sample, such as `['REF_636_RECENT']`, for the first run. An empty list selects every manifest.
3. Choose `cellpose.device`: `cpu`, `mps`, `cuda`, or `auto`. The resolved model/device/weights digest is recorded. GPU selection does not validate segmentation quality. `diameter`, if provided, is an analysis-image pixel diameter used for model rescaling; omission uses the model's native scale.
4. For comparative plots, copy `samples.example.json` to `samples.json`, replace the example with reviewed rows for your selected samples, and set `pipeline.sample_registry = 'samples.json'`. Do not infer ambiguous site/tree/period values from filenames. The registry is optional for a processing pilot, but required for comparisons.
5. Run preflight, then process the pilot:

```bash
uv run --locked python scripts/run_pipeline.py --config config.toml --preflight
uv run --locked --extra inference python scripts/run_pipeline.py --config config.toml
```

Preflight validates configuration, requested samples, manifests, remote object sizes and SHA metadata. It performs **no image downloads, model downloads, processing, or writes to R2**. During processing, every downloaded image's bytes are checked against the manifest, and TIFF headers/calibration are validated before stitching. Model construction currently happens before sample download so an invalid inference environment fails early. The first inference use can download pretrained weights.

The runner returns a nonzero exit code if requested work, processing, uploads, or aggregation fail. `status.json` records per-sample state. Successful processing is not reported as a completed run until all required uploads and aggregation have succeeded. A failed run may contain partial objects and must not be used for plots.

## Data and output contract

Existing raw objects do **not** need to be renamed or reuploaded just for this change. The accepted original files already have OME metadata in the inspected examples; all inputs are checked independently when processed.

```text
raw/<existing-sample-id>/<original-tile>.tif
manifests/<existing-sample-id>.json
runs/<unique-run-id>/
  run.json
  status.json
  all_samples.csv
  sample_summary.csv
  samples/<sample-id>/
    mosaic.ome.tif
    analysis.ome.tif
    tissue-mask.ome.tif
    labels.ome.tif
    coverage.json
    cells.csv
    qc.json
    overlay.jpg
    proximity.csv                 # only when configured
```

The legacy upload manifest schema (`sample`, `files[].key/size/sha256/status`) is supported. Failed entries, duplicate/unsafe keys, missing inputs, or unmanifested raw images are errors. AppleDouble `._*` files and `__MACOSX` entries are recorded as excluded metadata, not decoded as TIFFs. Paths sharing a sample-name prefix are not part of that sample. Nested TIFF paths and case-insensitive `.tif`/`.tiff` extensions are supported.

Local input caches use the manifest digest and verify SHA-256 using streaming reads; multipart ETags are never interpreted as MD5. Derived caches use source hashes, settings, sample metadata, code content/revision, software versions, and the actual model weights digest. A completion marker lists required output hashes. A changed input/setting/model or damaged artifact causes recomputation; legacy untracked files are never reused. Incomplete sample stages rerun from the beginning. A local file lock serializes production of identical artifact sets.

Each TIFF is a self-contained, tiled, losslessly compressed OME-BigTIFF. It carries physical X/Y pixel size and documented provenance annotations. Arrays use BGR internally and RGB on disk/model input. A copied or renamed TIFF retains its calibration. Re-saving it through metadata-unaware software may strip calibration and will be rejected by this pipeline. No external JSON file is required to interpret pixel size. Run records still carry detailed provenance and coverage.

A cached image's `artifact_id` identifies its processing result, which can be reused across runs. The run manifest maps run IDs to artifact IDs. Uploaded cell tables additionally include their current run ID. Source mosaics are never resized in place; `analysis.ome.tif` gets the actual effective pixel size after integer dimension rounding.

Stitching currently accepts single-plane uint8 RGB tiles with matching isotropic calibration. It rejects missing calibration, inconsistent scales, unreadable images, disconnected tile components, and estimated registration scale changes greater than 1%. `coverage.json` records expected/used/excluded tiles and camera transforms. This scale tolerance is an operational QC guard, **not a demonstrated scientific accuracy tolerance**. Inspect alignment and seams, especially in repetitive wood texture. Acquisition-grid registration and a complete geometric validation protocol remain future work.

## Measurements and scientific limits

- Lumen area uses `area_um2`; axes use `*_um`; both pixel and physical centroids have explicit suffixes. No silent fallback to pixel units is allowed. Anisotropic calibration is supported for measurements even though stitching currently requires isotropic inputs.
- **Wall thickness has been removed.** Distance to other lumens is not a validated measurement of the wall itself. The old helper now raises an explanatory error instead of returning misleading values. Neither features nor plots require that column.
- Optional `pipeline.adjacency_radius_um` produces an undirected proximity graph. It uses bounded cell neighborhoods and never reports an overlap-pixel count as a wall-interface length. The radius must be selected for the scientific use case; proximity is not a claim of biological adjacency. The default pipeline does not compute it.
- Cell classification remains the existing area/eccentricity rules, clearly marked `unvalidated_area_eccentricity_rules_v1`. Sample-relative thresholds can be affected by segmentation errors. Validate segmentation splits/merges/misses separately from cell-type precision/recall; hold out entire specimens/trees for any learned classifier.
- The tissue mask is a heuristic envelope with enclosed lumens filled, not an annotated tissue model. Blank/uniform inputs produce an empty mask. The runner fails on no detected tissue. Entire labels crossing the mask/image boundary are excluded, rather than measuring clipped lumens. Inspect this exclusion policy on real sections.
- Sample ray percentages always divide by **all classified lumens, including resin ducts**. A zero-cell sample stays in the summary with an undefined percentage, not zero. Comparative plotting rejects these samples pending QC. Sample summaries and plots do not perform statistical tests or infer tree pairing.

After inspecting pilot outputs, plot a completed local run:

```bash
uv run --locked python scripts/make_figures.py data/cache/runs/<run-id>
```

This writes stacked proportions and descriptive specimen-mean area plots. It requires a reviewed sample registry. The old pooled-cell and wall-thickness figures were removed because their inputs/interpretations were not reliable.

## Performance and Pegasus

This release removes the per-cell full-mosaic distance transforms and dilations identified in review. It **does not implement a fully streamed mosaic/inference pipeline**. OpenCV still loads source images and Cellpose still receives the full analysis image. Tiled TIFF storage alone does not bound those allocations.

`max_input_gb` and `max_analysis_pixels` are explicit guardrails, not measured total-RAM requirements. Defaults are conservative pilot limits; larger samples may intentionally stop. Additional arrays, feature matching, mosaic blending, and neural inference need more RAM than the image inputs alone. Record `qc.json` stage timings and actual peak host/GPU memory on a representative pilot before raising limits or scheduling the dataset. No M4 or V100 throughput is promised. MLX and window/seam reconciliation are not implemented here.

Before Pegasus submission, select a user-owned scratch path, install uv/load site-approved modules, and create `logs/` **before** `sbatch` opens output paths. Verify the actual cluster GPU partition/account options with your allocation.

```bash
mkdir -p logs
uv sync --locked --extra inference
uv run --locked --extra inference python scripts/check_device.py --device cuda
sbatch scripts/submit_pegasus.sh
```

The script uses fail-fast shell behavior and runs a device tensor check before the pipeline. It does not guess site module names or install software through a shell download during the job. For a separate config: `LOBLOLLY_CONFIG=/path/to/pilot.toml sbatch scripts/submit_pegasus.sh`.

## Standalone stages

Use `uv run --locked python scripts/<name>.py --help` for argument details. Segmentation additionally requires `--extra inference`. The updated standalone image/label interfaces use calibrated OME-TIFF rather than legacy `.npy` labels plus `metadata.json`.

```bash
uv run --locked python scripts/stitch.py tiles/ --out mosaic.ome.tif
uv run --locked python scripts/resize.py mosaic.ome.tif 0.5 --out analysis.ome.tif
uv run --locked python scripts/tissue_mask.py analysis.ome.tif tissue-mask.ome.tif
uv run --locked --extra inference python scripts/segment.py analysis.ome.tif tissue-mask.ome.tif labels.ome.tif --device mps
uv run --locked python scripts/features.py labels.ome.tif features.csv
uv run --locked python scripts/classify.py features.csv classified.csv
uv run --locked python scripts/adjacency.py labels.ome.tif proximity.csv --radius-um 1.0
uv run --locked python scripts/visualize.py analysis.ome.tif labels.ome.tif classified.csv overlay.jpg
```

Experimental color unmixing is separate from the production runner. It explicitly accepts RGB internally, checks degenerate fits, and saves numeric OME-TIFF component maps plus fitted vectors. Components are not asserted to be validated safranin/astra-blue concentrations. Per-image fits must not be compared quantitatively without additional validation:

```bash
uv run --locked python scripts/deconvolve.py analysis.ome.tif experimental-components/
```

## Uploading new sources

```bash
uv run --locked python scripts/upload_to_r2.py /path/to/dataset --sample 'Sample Folder'
```

The uploader validates TIFF headers/calibration, excludes macOS companion files, rejects conflicting raw object bytes rather than overwriting them, and streams downloaded bytes to verify SHA-256. This verification adds download traffic; it is stronger than reading client-written metadata. A manifest is published only after all selected files pass. Retries reuse matching raw objects. Keep an independent original copy: application-level immutability does not prevent a credential with write/delete permissions from changing the bucket.

## Development checks

```bash
uv run --locked ruff check .
uv run --locked ruff format --check .
uv run --locked pytest -q
uv build
```

Tests use synthetic calibrated TIFFs, actual OpenCV stitching, and fake R2/model boundaries. They exercise the real orchestration, upload call, physical measurements, cache invalidation, error exit statuses, path containment, and sample summaries. They do not replace biological annotations, a full-size resource profile, or real inference on the target machine.
