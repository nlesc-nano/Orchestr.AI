#!/bin/bash

#SBATCH --qos=regular
#SBATCH --job-name=training
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --constraint=a100-pcie
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load Anaconda3/2023.03-1
source activate /scratch/mfernandez/ml

CDIR=`pwd`

# Generate calculations folder
export SCRATCH_DIR=/scratch/$USER/wrktmp/$SLURM_JOBID
mkdir -p $SCRATCH_DIR

# Get config file from input parameter
CONFIG_FILE=$1

# If config file is not provided, set the default one
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="input.yaml"
fi

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' does not exist."
    exit 1
fi

echo "Using configuration file: $CONFIG_FILE"

# Copy necessary files to SCRATCH_DIR
cp "$CONFIG_FILE" $SCRATCH_DIR
cp *.npz $SCRATCH_DIR 

cd $SCRATCH_DIR

# Run training
python -m mlff_qd.training.main --config "$CONFIG_FILE"

# Check if training was successful and do the inference if it is possible
if [ $? -eq 0 ]; then
    echo "Training completed successfully. Starting inference."
    # Run inference
    python -m mlff_qd.training.inference --config "$CONFIG_FILE"
else
    echo "Training failed. Skipping inference."
fi


mkdir -p $CDIR/$SLURM_JOB_ID
cp -r * $CDIR/$SLURM_JOB_ID
rm -fr $SCRATCH_DIR
