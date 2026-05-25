#!/bin/bash
#SBATCH --job-name=gnn_h3
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00

set -euo pipefail

module purge
module load python/3.11 cuda/12.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_gnn

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

nvidia-smi

python3 scripts/run_h2.py --config experiments/h2_multimodal.yaml

echo "Job finished at $(date)"