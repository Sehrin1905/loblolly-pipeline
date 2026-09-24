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
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your_email@gwu.edu

# ── Environment setup ─────────────────────────────────────────────────────────
echo "Job started: $(date)"
echo "Running on: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'No GPU')"

# Load modules (adjust for Pegasus module system)
module load python/3.13
module load cuda/11.8

# Go to repo
cd $SLURM_SUBMIT_DIR

# Install uv if not available
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source ~/.bashrc
fi

# Install dependencies
uv sync

# Create logs directory
mkdir -p logs

# Run pipeline
echo "Starting pipeline..."
uv run scripts/run_pipeline.py --config config.toml

echo "Job finished: $(date)"
EOF