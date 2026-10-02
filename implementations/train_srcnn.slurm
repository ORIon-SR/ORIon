#!/bin/bash
#SBATCH --job-name=x16_srcnn
#SBATCH --partition=prime
#SBATCH --nodelist=griffin
#SBATCH --output=/home-mscluster/%u/ORIon/scripts/results/patch-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/ORIon/scripts/results/patch-error-%N-%j.txt

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p results

source /home-mscluster/$USER/ORIon/.venv/bin/activate

torchrun --nproc-per-node=2 srcnn.py --batch-size 64
