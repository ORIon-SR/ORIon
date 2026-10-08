#!/bin/bash
#SBATCH --job-name=x4_consistency
#SBATCH --partition=prime
#SBATCH --nodelist=phoenix
#SBATCH --output=/home-mscluster/%u/ORIon/implementations/results/consistency-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/ORIon/implementations/results/consistency-error-%N-%j.txt

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p results
pwd
ls -la
echo "Activating venv"
source /home-mscluster/$USER/ORIon/.venv/bin/activate

echo "Running torchrun"
torchrun --nproc-per-node=2 consistency_model.py --batch-size 8 --epochs 200 --patience 50
