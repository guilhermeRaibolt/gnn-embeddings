#!/bin/bash
# =============================================================================
# Slurm batch script: text-embedding scale vs. GNN node classification.
# Submit with:  sbatch run.sh
# =============================================================================
#SBATCH --job-name=gnn-embed-scale
#SBATCH --partition=A100            # GPU partition (adjust to your cluster)
#SBATCH --nodes=1                   # single node
#SBATCH --ntasks=1                  # single task / process
#SBATCH --gres=gpu:1                # this cluster exposes GPU type as features
#SBATCH --constraint=A100           # explicitly request an A100 node
#SBATCH --cpus-per-task=16          # CPU cores for data loading / vectorisers
#SBATCH --mem=96G                   # host RAM
#SBATCH --time=24:00:00             # wall-clock limit
#SBATCH --output=logs/slurm-%j.out  # Slurm stdout -> logs/
#SBATCH --error=logs/slurm-%j.err   # Slurm stderr -> logs/

set -euo pipefail

# --- Move to the project directory -------------------------------------------
# Under Slurm the script is copied to a (read-only) spool dir, so BASH_SOURCE is
# NOT the project path. Slurm sets SLURM_SUBMIT_DIR to the directory you ran
# `sbatch` from; use that, and fall back to the script dir for direct execution.
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

# Slurm directives above reference logs/ before the job's CWD is guaranteed to
# exist, so create the output directories up front.
mkdir -p logs data results figures

# --- Environment -------------------------------------------------------------
# Load cluster modules if your site uses Lmod/environment-modules; harmless when
# 'module' is unavailable. Adjust the names to match your cluster.
if command -v module >/dev/null 2>&1; then
    module load anaconda3 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi

# Activate the conda env that already has torch, PyG, transformers, sklearn,
# scipy and sentence-transformers installed.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn-embedding

# --- Diagnostics -------------------------------------------------------------
echo "Host:            $(hostname)"
echo "Started:         $(date)"
echo "Working dir:     $(pwd)"
echo "CUDA_VISIBLE:    ${CUDA_VISIBLE_DEVICES:-not_set}"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi is unavailable; this job does not appear to have a usable NVIDIA GPU." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
GPU_NAME="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)"
if [[ "${GPU_NAME}" != *"A100"* ]]; then
    echo "Expected an A100 GPU, but Slurm assigned: ${GPU_NAME}" >&2
    echo "Check the cluster's GPU GRES/constraint name, then resubmit." >&2
    exit 1
fi
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
python -c "import torch; print('torch', torch.__version__, '| cuda available:', torch.cuda.is_available(), '| device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"

# --- Run ---------------------------------------------------------------------
python main.py \
    --gnn sage \
    --device cuda \
    --encoders bow,tfidf,sbert_minilm,qwen3_0.6b,qwen3_4b,qwen3_8b \
    --sbert-batch-size 256 \
    --qwen-batch-size 16 \
    --max-length 1024 \
    --max-epochs 200 \
    --patience 30 \
    "$@"

echo "Finished:        $(date)"
