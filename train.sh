#!/bin/bash
#SBATCH -A csds312
#SBATCH -p markov_gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint=gpu2h100
#SBATCH --mem=24gb
#SBATCH -c 11
#SBATCH --time=13:00:00
#SBATCH -o logs/train_%j.out
#SBATCH -e logs/train_%j.err
#SBATCH --job-name=artgan_train
# Create logs directory
mkdir -p /mnt/vstor/courses/csds312/cvx166/ArtGan/logs

# Load any modules if needed
# module load python/3.x

# Activate your environment (if using conda/venv)
# source activate gpt2

# Run training
cd /mnt/vstor/courses/csds312/cvx166/ArtGan/code
PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run train.py