#!/bin/bash
#SBATCH --job-name=x16_diffusion
#SBATCH --partition=prime
#SBATCH --nodelist=griffin
#SBATCH --output=/home-mscluster/%u/ORIon/implementations/results/diffusion-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/ORIon/implementations/results/diffusion-error-%N-%j.txt

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p results
pwd
ls -la
echo "Activating venv"
source /home-mscluster/$USER/ORIon/.venv/bin/activate

echo "Running torchrun"
torchrun --nproc-per-node=2 bicubic_diffusion.py --batch-size 32
