#!/bin/bash
#SBATCH --job-name=x16_srcnn
#SBATCH --partition=prime
#SBATCH --nodelist=griffin
#SBATCH --output=/home-mscluster/%u/ORIon/implementations/results/srcnn-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/ORIon/implementations/results/srcnn-error-%N-%j.txt

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p results
pwd
ls -la
echo "Activating venv"
source /home-mscluster/$USER/ORIon/.venv/bin/activate

echo "Running torchrun"
torchrun --nproc-per-node=2 srcnn.py --batch-size 64
