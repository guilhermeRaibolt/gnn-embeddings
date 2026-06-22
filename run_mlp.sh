#!/bin/bash
# =============================================================================
# Slurm batch script: text-embedding scale vs. MLP node classification baseline.
# Submit with:  sbatch run_mlp.sh
# =============================================================================
#SBATCH --job-name=mlp-embed-scale
#SBATCH --partition=A100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --constraint=A100
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"

mkdir -p logs data results figures

if command -v module >/dev/null 2>&1; then
    module load anaconda3 2>/dev/null || true
    module load cuda 2>/dev/null || true
fi

source ~/miniconda3/etc/profile.d/conda.sh
conda activate "${CONDA_ENV:-gnn-embedding}"

echo "Host:            $(hostname)"
echo "Started:         $(date)"
echo "Working dir:     $(pwd)"
echo "Conda env:       ${CONDA_DEFAULT_ENV}"
echo "CUDA_VISIBLE:    ${CUDA_VISIBLE_DEVICES:-not_set}"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
fi

export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

python main.py \
    --gnn mlp \
    --device cuda \
    --encoders "${ENCODERS:-bow,tfidf,sbert_minilm,qwen3_0.6b,qwen3_4b,qwen3_8b}" \
    --sbert-batch-size "${SBERT_BATCH_SIZE:-256}" \
    --qwen-batch-size "${QWEN_BATCH_SIZE:-16}" \
    --max-length "${MAX_LENGTH:-1024}" \
    --max-epochs "${MAX_EPOCHS:-200}" \
    --patience "${PATIENCE:-30}" \
    "$@"

echo "Finished:        $(date)"
