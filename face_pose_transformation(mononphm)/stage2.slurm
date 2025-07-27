#!/bin/bash
#SBATCH -J EN2-L-512
#SBATCH --partition=gpu_a100
#SBATCH --qos=gpu-a100_default
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=32GB
#SBATCH -G 1
#SBATCH --job-name="s4_lrs3"
#SBATCH --output=step4_lrs3.out
#SBATCH --ntasks=1

echo "CUDA_DEVICE=/dev/nvidia/$CUDA_VISIBLE_DEVICES"
nvidia-smi

module load anaconda

source ~/scratch1/miniconda3/etc/profile.d/conda.sh
conda activate mononphm

# Define the directory containing the folders
DATASET_DIR="tracking_input"
OUTPUT_DIR="tracking_output/pretrained_mononphm_original/stage2"

# Loop through each folder inside DATASET_DIR
for folder in "$DATASET_DIR"/lrs3*/; do
    seq_name=$(basename "$folder")

    # Check if the sequence has already been processed
    if [ -d "$OUTPUT_DIR/$seq_name" ]; then
        echo "Skipping $seq_name (already processed)"
        continue
    fi

    echo "Running inference for: $seq_name"

    # Run the Python script with the folder name as seq_name
    srun python scripts/inference/rec.py --model_type nphm --exp_name pretrained_mononphm_original --ckpt 2500 --seq_name "$seq_name" --is_video --no-intrinsics-provided --is_stage2
done
