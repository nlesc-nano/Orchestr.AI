#!/bin/bash

#SBATCH --job-name=new_testx_1
#SBATCH --time=00:03:00  
#SBATCH -c 32
#SBATCH -p medium  
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100 
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# Activate environment  
source /mnt/netapp1/Store_RES/home/res/bcma1/resh000407/miniconda3/etc/profile.d/conda.sh
conda activate D_MLFF_latest5

CDIR=`pwd`

# Generate temporary calculations folder
export SCRATCH_DIR=/$LUSTRE/wrktmp/$SLURM_JOBID
mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"

###############################################
# 1. Argument Parsing
###############################################

CONFIG_FILE="$1"

# If config file is not provided, set the default one
if [ -z "$CONFIG_FILE" ]; then
    CONFIG_FILE="input_new.yaml"
fi

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' does not exist."
    exit 1
fi

echo "Using configuration file: $CONFIG_FILE"

###############################################
# 2. Prepare Files
###############################################

# Copy to scratch with error suppression
cp "$CONFIG_FILE" "$SCRATCH_DIR"
cp *.npz "$SCRATCH_DIR" 2>/dev/null || true
cp *.xyz "$SCRATCH_DIR" 2>/dev/null || true

cd "$SCRATCH_DIR"

###############################################
# 3. Run Training
###############################################

python -m mlff_qd.training --config "$CONFIG_FILE" "${@:2}" || echo "Training failed, proceeding with copy"

###############################################
# 4. Copy Back to Original Dir
###############################################

mkdir -p "$CDIR/$SLURM_JOB_ID"
if [ -d "$SCRATCH_DIR/standardized" ]; then
    cp -r "$SCRATCH_DIR/standardized/"* "$CDIR/$SLURM_JOB_ID/" 2>/dev/null || true
else
    cp -r "$SCRATCH_DIR/"* "$CDIR/$SLURM_JOB_ID/" 2>/dev/null || true
fi
rm -rf "$SCRATCH_DIR"