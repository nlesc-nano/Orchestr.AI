#!/bin/bash
#SBATCH --account=AIFAC_S01-026
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --job-name=mlff_qd_train
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --output=mlff_qd_train_%j.out
#SBATCH --error=mlff_qd_train_%j.err
#SBATCH --hint=nomultithread

cd $SLURM_SUBMIT_DIR

source /leonardo_work/AIFAC_S01-026/musman00/miniconda3/etc/profile.d/conda.sh
conda activate /leonardo_work/AIFAC_S01-026/musman00/miniconda3/envs/MLFF

export SCRATCH_DIR=/leonardo_scratch/large/userexternal/musman00/AIFAC_S01-026/wrktmp/$SLURM_JOBID
mkdir -p "$SCRATCH_DIR"
export TMPDIR="$SCRATCH_DIR"

export PYTHONWARNINGS="ignore"
export OMP_NUM_THREADS=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Set environment variables for multi-GPU training
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export CUDA_VISIBLE_DEVICES=0,1

# Distributed training settings
export MASTER_ADDR=$(hostname)
export MASTER_PORT=$((29500 + RANDOM % 1000))

unset PYTORCH_CUDA_ALLOC_CONF

echo "=== MLFF-QD Multi-GPU Training ==="
echo "Job ID: $SLURM_JOBID"
echo "Tasks: $SLURM_NTASKS, GPUs: $SLURM_GPUS"
echo "MASTER_ADDR: $MASTER_ADDR, MASTER_PORT: $MASTER_PORT"

nvidia-smi
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

CONFIG_FILE="${1:-config.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found!"
    exit 1
fi
echo "Using config: $CONFIG_FILE"

cp "$CONFIG_FILE" "$SCRATCH_DIR/"
cp *.yaml *.xyz *.csv "$SCRATCH_DIR/" 2>/dev/null || true
cp -r data "$SCRATCH_DIR/" 2>/dev/null || true
cd "$SCRATCH_DIR"

# Use srun to launch 2 tasks (one per GPU) and let PyTorch Lightning handle the distribution
srun --cpu-bind=none \
    python -m mlff_qd.training \
    --config "$CONFIG_FILE" "${@:2}"

# Copy results back
mkdir -p "$SLURM_SUBMIT_DIR/$SLURM_JOBID"
cp -r ./* "$SLURM_SUBMIT_DIR/$SLURM_JOBID/" 2>/dev/null || true

rm -rf "$SCRATCH_DIR"
echo "Job completed"