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

export PYTHONUNBUFFERED=1

CDIR="$(pwd)"

###############################################
# 0. Safe scratch setup (DB-safe)
###############################################

SCRATCH_BASE="/leonardo_scratch/large/userexternal/$USER"
SCRATCH_DIR="$SCRATCH_BASE/job_${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRATCH_DIR"

# Redirect all temp/DB/HDF5 activity into scratch
export SCRATCH_DIR
export TMPDIR="$SCRATCH_DIR"
export SQLITE_TMPDIR="$SCRATCH_DIR"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="$SCRATCH_DIR/.mpl"


###############################################
# 1. Argument Parsing
###############################################

CONFIG_FILE="${1:-input_new.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' does not exist."
    exit 1
fi
echo "Using configuration file: $CONFIG_FILE"

###############################################
# 2. Prepare Files
###############################################

cp "$CONFIG_FILE" "$SCRATCH_DIR/"
cp *.npz "$SCRATCH_DIR" 2>/dev/null || true
cp *.xyz "$SCRATCH_DIR" 2>/dev/null || true

cd "$SCRATCH_DIR"

###############################################
# 3. Run Training
###############################################

python -m mlff_qd.training --config "$(basename "$CONFIG_FILE")" "${@:2}" || echo " Training failed, proceeding with copy"

###############################################
# 4. Copy Back to Original Dir
###############################################

JOB_OUT="$CDIR/$SLURM_JOB_ID"
mkdir -p "$JOB_OUT"

if [ -d "$SCRATCH_DIR/benchmark_results" ]; then
    cp "$CONFIG_FILE" "$JOB_OUT/" 2>/dev/null || echo "Warning: Config YAML not found"
    [ -f "$SCRATCH_DIR/benchmark_summary.csv" ] && cp "$SCRATCH_DIR/benchmark_summary.csv" "$JOB_OUT/"
    cp -r "$SCRATCH_DIR/benchmark_results" "$JOB_OUT/"
else
    if [ -d "$SCRATCH_DIR/standardized" ]; then
        cp -r "$SCRATCH_DIR/standardized" "$JOB_OUT/"
        cp -f "$SCRATCH_DIR/$(basename "$CONFIG_FILE")" "$JOB_OUT/" 2>/dev/null || true
    else
        cp -r "$SCRATCH_DIR/"* "$JOB_OUT/" 2>/dev/null || true
    fi
fi

echo "✓ Done. Results at: $JOB_OUT"

# Safety cleanup
cleanup() {
  if [[ -n "${SCRATCH_DIR:-}" && -d "$SCRATCH_DIR" && "$SCRATCH_DIR" != "/" && "$SCRATCH_DIR" != "/tmp" ]]; then
    rm -rf "$SCRATCH_DIR"
  else
    echo " Skip cleanup; unsafe SCRATCH_DIR='$SCRATCH_DIR'"
  fi
}
trap cleanup EXIT