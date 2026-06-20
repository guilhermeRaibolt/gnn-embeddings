#!/bin/bash
# =============================================================================
# QUICK sanity check: build the real Toys & Games graph and run the lexical
# baselines (BoW + TF-IDF) end-to-end on CPU. No GPU needed -> starts fast.
# Submit from inside the project dir:  cd ~/trygnn && sbatch run_quick.sh
# =============================================================================
#SBATCH --job-name=gnn-quick
#SBATCH --partition=CPU
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# Slurm sets the working dir to the submission directory; use it (the script
# itself runs from a read-only spool dir, so BASH_SOURCE is not the project).
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs data results figures

if command -v module >/dev/null 2>&1; then
    module load anaconda3 2>/dev/null || true
fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn-embedding

echo "Host:        $(hostname)"
echo "Started:     $(date)"
echo "Working dir: $(pwd)"

# CPU-only, baselines only, short training — just proves the full chain works.
python main.py \
    --gnn sage \
    --device cpu \
    --encoders bow,tfidf \
    --max-epochs 60 \
    --patience 15

echo "Finished:    $(date)"
