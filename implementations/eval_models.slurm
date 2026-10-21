#!/bin/bash
#SBATCH --job-name=models_eval
#SBATCH --partition=prime
#SBATCH --nodelist=phoenix
#SBATCH --output=/home-mscluster/%u/ORIon/scripts/results/eval-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/ORIon/scripts/results/eval-error-%N-%j.txt

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p results

source /home-mscluster/$USER/ORIon/.venv/bin/activate

python implementations/models_eval.py --split test --wandb
