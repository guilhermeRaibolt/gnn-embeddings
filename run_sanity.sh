#!/bin/bash
# =============================================================================
# FAST end-to-end confirmation on GPU: real graph + BoW/TF-IDF baselines with a
# minimal 1-point grid and 2 seeds -> finishes in minutes and produces the final
# Markdown table + all figures. Reuses cached features. Submit from project dir:
#   cd ~/trygnn && sbatch run_sanity.sh
# =============================================================================
#SBATCH --job-name=gnn-sanity
#SBATCH --partition=A40
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
mkdir -p logs data results figures

if command -v module >/dev/null 2>&1; then module load anaconda3 2>/dev/null || true; fi
source ~/miniconda3/etc/profile.d/conda.sh
conda activate gnn-embedding

echo "Host: $(hostname) | Started: $(date) | Dir: $(pwd)"
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

# Minimal search (1 config) + 2 seeds so the whole chain finishes fast.
python main.py \
    --gnn sage --device cuda \
    --encoders bow,tfidf \
    --learning-rates 0.01 --hidden-channels 128 --dropouts 0.5 \
    --max-epochs 40 --patience 12 \
    --eval-seeds 0,1

echo "Finished: $(date)"
