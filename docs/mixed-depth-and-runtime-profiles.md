# Mixed-bit-depth tiles and local/Pegasus runs: an implementation guide

This is a plan and a learning exercise, not an implemented pipeline change. The
Python examples describe proposed work unless explicitly marked as runnable today.
The TOML profiles and invocation commands use interfaces that already exist.

Start with one REF sample and learn to select a configuration. Then investigate
and implement the Marsh color conversion. Finish by measuring resource use and
checking segmentation quality before processing all samples.

## 1. Understand the two failures

Investigation on September 29, 2026 used source revision
`88280f805126b1025fd6ce3cfa93959304f9b261` and the following R2 records:

| Sample | Run under `runs/` | Recorded failure |
| --- | --- | --- |
| `Marsh_45_Old` | `20260929T190854Z-c2fdf911d318` | `Stitching currently requires single-plane uint8 RGB TIFFs` |
| `REF_851_Recent` | `20260929T194227Z-a6c1b82570b7` | `Requested analysis dimensions exceed max_analysis_pixels; profile before raising it` |
| `REF_636_RECENT` | `20260929T195832Z-d58127363187` | Same analysis-pixel-limit error |

Each run's `run.json` records factor `0.5`, input limit `2.0` GB, and analysis
limit `25000000` pixels. The REF 851 run selected the Apple GPU (`mps`). The REF
errors occurred after stitching and before resizing. They do not demonstrate an
out-of-memory failure.

R2 contains 93 Marsh TIFF objects. `X9_Y1` and `X11_Y4` are each approximately
75.2 MB; the other objects are approximately 37.6 MB. That supports the reported
91 uint8 / 2 uint16 split, but this investigation did not decode those two remote
TIFFs to independently verify their bit depth or intensity encoding. Their
acquisition/export history remains a question to investigate.

Keep these concepts separate:

| Concept | Meaning | Current setting |
| --- | --- | --- |
| Color bit depth | Precision of each channel's intensity; converting 16 to 8 bits does not change width or height | No conversion setting exists yet |
| Spatial resolution | Number of pixels representing the tissue | `pipeline.mosaic_resize_factor` |
| Analysis size guard | Maximum allowed width times height; does not resize or allocate memory | `pipeline.max_analysis_pixels` |
| Stitching input guard | Estimated decoded tile bytes; not total process memory or color bit depth | `pipeline.max_input_gb` |

The current processing order is:

```text
Original TIFFs → load tiles → full-resolution stitch → mosaic.ome.tif
              → check requested analysis dimensions
              → resize → analysis.ome.tif → segmentation → measurements
```

**Changing the analysis resize factor does not reduce stitching memory or alter
the images used for registration.** There is no user-configurable pre-stitch
resize factor in the current pipeline.

## 2. Select a configuration: runnable today

Work from the repository root after following the README installation and
credential setup. Credentials belong in the existing environment setup, never in
these TOML files. The runner already supports `--config`; the Slurm script already
supports `LOBLOLLY_CONFIG`.

Create a `configs/` directory. Save this complete example as
`configs/sehrin_local.toml`:

```toml
[cache]
local_cache_dir = "data/cache"

[pipeline]
samples = ["REF_851_Recent"]
mosaic_resize_factor = 0.35
max_input_gb = 2.0
max_analysis_pixels = 25000000

[cellpose]
device = "mps"
model = "cpsam_v2"

[r2]
bucket_name = "sehrin-image-stitching"
```

Save the following as `configs/pegasus.toml`. Replace the cache path with a real,
user-owned scratch directory before running. This is a full-resolution trial
profile for REF 851, not a guarantee that the job fits its allocation.

```toml
[cache]
local_cache_dir = "/REPLACE_WITH_YOUR_SCRATCH/loblolly-cache"

[pipeline]
samples = ["REF_851_Recent"]
mosaic_resize_factor = 1.0
max_input_gb = 2.0
max_analysis_pixels = 180000000

[cellpose]
device = "cuda"
model = "cpsam_v2"

[r2]
bucket_name = "sehrin-image-stitching"
```

These examples intentionally select one sample. `samples = []` selects all
validated manifests. Sample IDs are case-sensitive. Paths for
`cache.local_cache_dir` are relative to the working directory; an optional
`pipeline.sample_registry` path is instead relative to its TOML file. Copying a
profile into `configs/` may require adjusting that registry path.

Run locally:

```bash
uv run --locked python scripts/run_pipeline.py --config configs/sehrin_local.toml --preflight
uv run --locked --extra inference python scripts/run_pipeline.py --config configs/sehrin_local.toml
```

Preflight checks storage/configuration, not decoded images or memory fitness.
It cannot detect the mixed bit depths or promise a successful processing run.
The processing command can download model weights and uploads run records and
completed results to R2.

On Pegasus, follow the README's site/environment setup, then submit from the
repository root:

```bash
mkdir -p logs
LOBLOLLY_CONFIG=configs/pegasus.toml sbatch scripts/submit_pegasus.sh
```

First verify the cluster environment with a smaller analysis trial: temporarily
set the Pegasus factor to `0.5` and its pixel limit to `45000000`. After that
trial succeeds and memory use is inspected, attempt the `1.0` / `180000000`
profile above. Both trials still perform full-resolution stitching.

### What resolution should I start with?

Using Sehrin's reported REF 851 mosaic dimensions of 8805 × 19452, the current
rounding rule gives:

| Factor | Analysis dimensions, in the same axis order | Pixels | Purpose |
| --- | --- | ---: | --- |
| `0.3` | 2642 × 5836 | 15,418,712 | Smaller local debugging trial |
| `0.35` | 3082 × 6808 | 20,982,256 | Initial local trial below the existing 25M guard |
| `0.5` | 4402 × 9726 | 42,813,852 | Higher-resolution trial; needs a guard of at least this many pixels |
| `1.0` | 8805 × 19452 | 171,274,860 | Full-resolution trial; 180M admits this particular mosaic |

Start locally at `0.35` for this sample. Start on Pegasus at `0.5` as described
above, then try `1.0`. These choices clear the specified guards; neither host has
yet been shown here to complete these exact analysis trials. Compute dimensions
separately for REF 636 and every other mosaic. Do not assume their limits match.

**Checkpoint:** explain which configuration was selected, what dimensions it
requests, and whether any failure is a policy limit, an actual allocation
failure, a stitching failure, or a segmentation-quality problem.

## 3. Inspect the Marsh inputs without changing them

Use the verified local input cache populated by the failed run. Under the default
cache, its layout is `data/cache/inputs/Marsh_45_Old/<manifest-digest>/tiles/`.
Choose the directory corresponding to the run's manifest, not a mixture of cached
versions. Do not edit, rename, or replace files in that directory.

The following read-only diagnostic is runnable today. Replace `TILE_DIR` with
that real directory. It prints header information for each tile, optional OME
significant bits, and sampled intensity statistics for uint16 inputs.

```bash
TILE_DIR='/absolute/path/to/Marsh_45_Old/manifest-digest/tiles'
uv run --locked python - "$TILE_DIR" <<'PY'
import sys
from collections import Counter
from xml.etree import ElementTree as ET

import numpy as np
import tifffile
from wood_stitch.image_io import tiff_info
from wood_stitch.stitch import tile_paths

paths = tile_paths(sys.argv[1])
if not paths:
    raise SystemExit("No TIFF tiles found: check TILE_DIR")
counts = Counter()
source_bytes = 0
rgb8_bytes = 0
for path in paths:
    info = tiff_info(path)
    if len(info["shape"]) != 3 or info["shape"][-1] != 3:
        raise SystemExit(f"Expected RGB tile: {path}")
    elements = int(np.prod(info["shape"]))
    source_bytes += elements * np.dtype(info["dtype"]).itemsize
    rgb8_bytes += elements
    counts[info["dtype"]] += 1
    with tifffile.TiffFile(path) as tif:
        root = ET.fromstring(tif.ome_metadata)
        pixels = next(e for e in root.iter()
                      if e.tag.rsplit("}", 1)[-1] == "Pixels")
        print(path.name, info, "SignificantBits:", pixels.get("SignificantBits"))
        if info["dtype"] == "uint16":
            image = tif.asarray()
            # Subsampling limits the statistics work; these are not exact extrema.
            sampled = image[::16, ::16].reshape(-1, 3)
            print("sampled RGB percentiles 0/1/50/99/100:",
                  np.percentile(sampled, [0, 1, 50, 99, 100], axis=0))
            del sampled, image
print("Tile counts:", dict(counts))
print("Source arrays, decimal GB:", source_bytes / 1e9)
print("Converted uint8 arrays, decimal GB:", rgb8_bytes / 1e9)
PY
```

If an unexpected shape or metadata error appears, investigate it before using
these RGB estimates. Inspect the two unusual images visually beside their
neighbors and compare the microscope/export settings. A 16-bit container can
hold 12-bit values or another encoding. OME's optional
[`SignificantBits`](https://github.com/ome/ome-model/blob/master/specification/src/main/resources/released-schema/2016-06/ome.xsd)
describes meaningful precision, but missing metadata, bit alignment, exposure,
or export tone curves can still require clarification. A dark image alone is
not evidence that its white level should be reduced.

**Checkpoint:** identify a justified mapping to the same brightness scale as
the neighboring uint8 exports. If the metadata and export history are ambiguous,
report that uncertainty rather than silently guessing from each tile's maximum.

## 4. Implement color conversion at the stitching boundary

This section proposes code changes for a later implementation PR. Adding TOML
alone cannot implement conversion, and new configuration keys are currently
rejected by the validator.

Keep the raw files and R2 objects unchanged. Add a small, tested helper near
[`src/wood_stitch/stitch.py`](../src/wood_stitch/stitch.py), and apply it to each
decoded color tile before passing tiles to OpenCV. Preserve the existing BGR
channel order. Do not make the generic reader convert all images: label TIFFs
use integer object IDs that would be corrupted by conversion.

An example for an encoding whose black level is zero and whose white level has
been established from metadata/export settings:

```python
def for_stitching(image, *, white_level):
    if image.dtype == np.uint8:
        return image
    if image.dtype != np.uint16:
        raise ValueError("Expected uint8 or uint16 color pixels")
    if not 0 < white_level <= 65535:
        raise ValueError("Invalid white level")
    scaled = image.astype(np.float32) * (255.0 / white_level)
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)
```

The module needs `import numpy as np`. For confirmed full-range 16-bit data,
`white_level=65535` maps 0 to 0 and 65535 to 255. This example is not a general
solution for unknown export encodings or nonzero black levels. Do not choose
65535 merely because the dtype is uint16, and do not normalize each tile or
channel independently. A direct `astype(np.uint8)` can wrap values instead of
scaling them.

Implement the changes in this order:

1. Define the supported encoding policy and how metadata selects it. Require an
   explicit, reviewed override when metadata cannot establish the mapping.
2. Permit single-plane RGB uint8 and uint16 in the stitcher's validation. Keep
   the existing shape, calibration, coverage, and registration checks.
3. Decode and convert one tile at a time. Retain the converted arrays for the
   current stitcher; avoid retaining a second collection of all uint16 arrays.
4. Update the input estimate to describe what it counts. Report source decoded
   bytes and retained uint8 bytes separately, and explain the additional
   decoder/conversion workspace. Neither number predicts OpenCV peak memory.
5. If introducing a setting, update `validate_config` in
   [`pipeline.py`](../src/wood_stitch/pipeline.py), pass it to stitching, and
   include the selected policy and resolved per-tile mapping in provenance and
   cache identity. Ensure copied output TIFFs retain the relevant provenance.

Changing bit depth must not change array dimensions or physical pixel sizes.
This approach requires no reupload of the original dataset.

**Checkpoint:** unit tests show uint8 passthrough, known uint16 mappings,
rejection of unsupported types/invalid policy, and unchanged source arrays.
An integration test uses calibrated synthetic mixed-depth RGB tiles to exercise
the real stitching path, verifies calibration and complete coverage, and compares
source-file SHA-256 hashes before and after. Also check that a changed conversion
policy invalidates cached results and is recorded in output metadata.

## 5. Measure memory instead of guessing a ceiling

Image sizes provide useful lower bounds, not an exact maximum safe resolution.
For an array, `height × width × channels × bytes_per_channel` gives its pixel
storage. TIFF file size may differ because of compression and metadata.

For the reported full-resolution REF 851 mosaic, one uint8 RGB array is about
0.514 GB, one uint32 label array about 0.685 GB, and one float32 RGB array about
2.06 GB. Multiple arrays, the model, feature matching, blending, and library
workspaces may coexist. These numbers are examples, not a sum of all live memory.

If the 93 Marsh tiles are each 4080 × 3072 RGB, their retained uint8 arrays alone
need about 3.50 GB; the reported mixed source arrays would need about 3.57 GB.
Verify the dimensions with step 3. Such a sample will hit the existing 2 GB input
guard after its dtype problem is fixed. An input guard of 4.0 GB could admit
those converted arrays, but it does not imply that a 4 GB machine can stitch them.

On the Mac, record installed memory and watch Activity Monitor's Memory Pressure
and swap during a one-sample trial. For a process peak measurement, this command
is available on macOS:

```bash
/usr/bin/time -l uv run --locked --extra inference python scripts/run_pipeline.py \
  --config configs/sehrin_local.toml
```

Record the reported maximum resident set size and its units; it is not a complete
measure of all shared/GPU/system memory. The Mac's GPU shares system memory.
Keep ordinary OS/application needs in the budget. Stop increasing workload if
memory pressure or swapping makes the machine unresponsive.

The current Pegasus script requests `--mem=64G` and one V100. Host RAM and GPU
memory are separate limits there. Use the allocated job's GPU information and,
where the site's Slurm accounting supports it, inspect memory after the job:

```bash
sacct -j JOB_ID --format=JobID,State,Elapsed,ReqMem,MaxRSS
```

Replace `JOB_ID` with the submitted job number and inspect the job-step rows too.
Accounting availability and sampling vary; a blank value is not zero memory.
Use `nvidia-smi` on the allocated GPU node to observe device memory during
inference. A larger TOML threshold does not request more RAM or GPU memory from
Slurm. Adjust the job allocation through the cluster's supported options when
measurements justify it.

Keep a small results table for each trial:

| Sample / run ID | Machine / allocation | Factor / pixels | Input guard | Peak host / GPU memory | Stitch / analysis time | Outcome / QC |
| --- | --- | --- | --- | --- | --- | --- |
| Fill from the actual run | | | | | | |

`qc.json` contains stage timings after a sample completes. Failed runs may not
have that file; retain the terminal/Slurm logs and `status.json` too. Change one
setting at a time so the result is interpretable.

### What is the minimum acceptable resolution?

For the current analysis factor, the lower bound comes from segmentation and
measurement quality, not stitch matching. Compare the same representative tissue
region at factors such as 0.35, 0.5, and 1.0. Inspect missed, split, and merged
lumens and compare measurements against reviewed annotations. Preserving
micrometers-per-pixel metadata cannot restore detail lost by resizing. If using
an explicit Cellpose `diameter`, remember that it is expressed in analysis-image
pixels; keep the intended physical scale consistent across comparisons.

Only if a separate pre-stitch resizing feature is later added does a stitching
resolution floor enter this decision. Validate matching, alignment, tile coverage,
and physical calibration at each candidate scale. Visual quality and memory
feasibility define constraints; an arbitrary midpoint between two scale factors
is not automatically a good choice. Pixel counts scale approximately as the
square of the factor, and peak memory need not follow one simple proportionality.

**Checkpoint:** select a resolution that passes both measured resource limits
and a documented quality check. Neither a completed run nor full resolution
alone validates the pretrained model or cell-type rules. Wall thickness is not
currently measured by this pipeline.

## 6. Improve errors and make the implementation reviewable

Replace the generic analysis-limit error with actual numbers and available
choices. An example for REF 851 is:

```text
Requested analysis: 4402 × 9726 = 42,813,852 pixels (factor 0.5).
Configured maximum: 25,000,000 pixels.
No resize was attempted.
To retain this resolution, raise max_analysis_pixels in the selected config
after checking the machine's resources. For a smaller debugging run, reduce
mosaic_resize_factor. This limit is a policy guard, not measured free memory.
```

Prefer an explicit configuration/resource-limit exception over `MemoryError`
for this check, so a deliberate refusal is distinguishable from a real memory
allocation failure. Record the failing stage and useful dimensions in run status.
For mixed-depth errors, identify the affected filenames and dtypes. Keep messages
free of credentials.

The implementation PR should include the conversion policy and its evidence,
focused tests, example profiles, clearer errors, and a short trial report. A
reasonable completion checklist is:

- [ ] Metadata/export evidence justifies the uint16 conversion policy.
- [ ] Tests cover mixed-depth input, immutable originals, calibration, provenance,
      cache invalidation, and unsupported input handling.
- [ ] Tests cover the pixel guard at/over its boundary and an informative error.
- [ ] Both example TOMLs validate; selecting a profile requires no source edit.
- [ ] One REF run completes with recorded settings and resource observations.
- [ ] Marsh uses all declared tiles after conversion, with inspected seams and
      unchanged originals; remaining failures are reported accurately.
- [ ] Reduced/full-resolution segmentation comparisons document an acceptable
      analysis choice. No claim of wall-thickness measurement is introduced.

Run the existing development checks listed in the README when implementing code.
Synthetic tests establish behavior; the real sample checks establish whether the
chosen encoding, alignment, resource settings, and model work on this dataset.

## Follow-up work, separate from these fixes

Currently, changing analysis settings invalidates the whole derived result and
can repeat stitching; failed sample processing also restarts its stages. A future
cache improvement could reuse a verified mosaic while varying analysis settings.
Do not manually mark an incomplete result complete to bypass this behavior.

A truly bounded-memory workflow needs more than TOML: consider windowed reading,
registration/rendering strategies, and tiled segmentation with reconciled object
IDs at window boundaries. The current analysis resize setting and tiled TIFF
storage do not implement those features. Keep that larger redesign out of the
first mixed-depth/configuration implementation PR.
