#!/bin/bash

#SBATCH --job-name=train
#SBATCH --time=12:00:00  
#SBATCH -p medium
#SBATCH -c 32  
#SBATCH --mem=32GB 
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
#SBATCH --gres=gpu:a100:1 

# Set environment variables for CUDA and GPU
#export CUDA_VISIBLE_DEVICES=0  # Ensure the A100 GPU is correctly targeted
#export NCCL_P2P_DISABLE=1  # Disable P2P (not needed for single-GPU tasks, reduces overhead)
#export NCCL_IB_DISABLE=1   # Disable InfiniBand (if not using, reduces overhead)

# Set environment variables for parallel processing
#export OMP_NUM_THREADS=32  # Match the number of CPU cores
#export MKL_NUM_THREADS=32  # Ensure MKL (used by some PyCaret models) uses all CPU cores
#export OPENBLAS_NUM_THREADS=32  # For OpenBLAS (used by some numpy operations)

# Activate conda environment
conda activate schnetpack-env

# Run Python script with srun, explicitly requesting 32 CPUs
srun run-md-opt config_opt.yaml
#srun --cpus-per-task=32 --gpus-per-task=1 python old_md.py config.yaml


