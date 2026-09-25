#!/bin/bash
#SBATCH --job-name=loblolly-pipeline
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --error=logs/pipeline_%j.err
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:?Submit from the repository root}"
# Before submission: load site-approved modules, install uv, and create logs/.
# The locked torch wheel contains its CUDA runtime. Do not assume a CUDA module repairs a wheel.
command -v uv >/dev/null
uv sync --locked --extra inference
uv run --locked --extra inference python scripts/check_device.py --device cuda
uv run --locked --extra inference python scripts/run_pipeline.py --config "${LOBLOLLY_CONFIG:-config.toml}"
