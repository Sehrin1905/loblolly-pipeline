#!/bin/bash
#SBATCH --job-name=loblolly-pipeline
#SBATCH --partition=gpu
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
# Create logs/ BEFORE sbatch; Slurm opens output files before this script runs.
# Prepare uv and the locked environment before submission. No CUDA toolkit module
# is needed for the prebuilt torch wheel; the driver and GPU must support it.
command -v uv >/dev/null
config="${LOBLOLLY_CONFIG:-configs/pegasus_ref636.toml}"
test -f "$config"
echo "Job started: $(date); host: $(hostname); config: $config"
uv sync --locked --extra inference
uv run --locked --extra inference python scripts/check_device.py --device cuda
# The pipeline honors an explicitly supplied LOBLOLLY_CACHE_DIR; otherwise use
# the TOML path. Never replace it with another user's scratch directory.
exec uv run --locked --extra inference python scripts/run_pipeline.py --config "$config"
