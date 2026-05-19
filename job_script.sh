#!/bin/bash
#SBATCH --job-name=my_first_job
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --partition=P100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=4:00:00

module purge
module load python/3.11 cuda/12.4

source ~/miniconda3/etc/profile.d/conda.sh
conda activate env_gnn

echo "Job $SLURM_JOB_ID started on $(hostname) at $(date)"

nvidia-smi

python scripts/run_h1.py --config experiments/h1_graph_vs_nograph.yaml

echo "Job finished at $(date)"