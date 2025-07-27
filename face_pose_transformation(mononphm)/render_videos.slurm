#!/bin/bash
#SBATCH -J EN2-L-512
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu-a100_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32GB
#SBATCH -G 1
#SBATCH --job-name="mononphm"
#SBATCH --output=output.out
#SBATCH --ntasks=1

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

module load anaconda

source ~/scratch1/miniconda3/etc/profile.d/conda.sh
conda activate step5

srun python step5.py



