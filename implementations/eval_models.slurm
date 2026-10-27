#!/bin/bash
#SBATCH --job-name=models_eval
#SBATCH --partition=biggpu
#SBATCH --nodelist=mscluster110
#SBATCH --output=/home-mscluster/%u/ORIon/implementations/results/eval-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/ORIon/implementations/results/eval-error-%N-%j.txt

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p results

source /home-mscluster/$USER/ORIon/.venv/bin/activate

python models_eval.py --split test --wandb --force --diffusion-batch-size 64
