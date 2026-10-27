#!/bin/bash
#SBATCH --job-name=swin2sr_zeroshot
#SBATCH --partition=prime
#SBATCH --nodelist=griffin
#SBATCH --output=/home-mscluster/%u/ORIon/implementations/results/swin2sr-output-%N-%j.txt
#SBATCH --error=/home-mscluster/%u/ORIon/implementations/results/swin2sr-error-%N-%j.txt

set -euo pipefail

cd $SLURM_SUBMIT_DIR
mkdir -p results
pwd
echo "Activating venv"
source /home-mscluster/$USER/ORIon/.venv/bin/activate

# Compute nodes may have no internet. Download the weights once on a login node:
#   huggingface-cli download caidas/swin2SR-classical-sr-x4-64 --local-dir models/swin2sr-classical-x4
# then run offline against the local folder.
export HF_HUB_OFFLINE=1

echo "Running zero-shot Swin2SR + bicubic on the test split"
python swin2sr_eval.py --split test --model-id models/swin2sr-classical-x4 --batch-size 16 --wandb
