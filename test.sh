#!/bin/bash

#SBATCH --qos=long
#SBATCH --job-name=scheduler
#SBATCH --time=2-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --constraint=a100-pcie
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

module load Anaconda3/2023.03-1
source activate /scratch/mfernandez/ml

CDIR=`pwd`

export SCRATCH_DIR=/scratch/$USER/wrktmp/$SLURM_JOBID
mkdir -p $SCRATCH_DIR
cp training.py $SCRATCH_DIR
cp input.yaml $SCRATCH_DIR
cp cspbbr3.npz $SCRATCH_DIR

cd $SCRATCH_DIR

python3.11 training.py

mkdir -p $CDIR/$SLURM_JOB_ID
cp -r * $CDIR/$SLURM_JOB_ID
rm -fr $SCRATCH_DIR
