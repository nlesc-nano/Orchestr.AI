#!/bin/bash

#SBATCH --job-name=schnet_test
#SBATCH --time=00:10:00
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# --- Multi-GPU launch notes ----------------------------------------------------
# For SchNet/PAINN/NequIP/Allegro:
#   - Multi-GPU works with a single SLURM task; frameworks spawn DDP workers internally.
#   - Use ONLY:
#       #SBATCH --gres=gpu:a100:<N>
#   - (Do NOT set --ntasks-per-node=<N> to avoid duplicate logs.)
#
# For MACE:
#   - Requires one SLURM task per GPU; launcher must set WORLD_SIZE/RANK/LOCAL_RANK.
#   - Use BOTH:
#       #SBATCH --gres=gpu:a100:<N>
#       #SBATCH --ntasks-per-node=<N>
#   - Expect duplicate INFO lines (one per rank) unless you mute non-rank0 in code.
# -------------------------------------------------------------------------------


###############################################
# 1. ENVIRONMENT SETUP
###############################################

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mlff_newx3

# Fix GLIBCXX mismatch: prefer conda's libstdc++
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1

echo "SLURM_TMPDIR=${SLURM_TMPDIR:-<unset>}"
echo "TMPDIR=${TMPDIR:-<unset>}"
echo "PWD(before)=$(pwd)"

CDIR="$(pwd)"

###############################################
# 2. SCRATCH DIRECTORY SETUP
# Purpose:
#   - Create job-local scratch for fast I/O
#   - Ensure safe temp usage for SQLite, HDF5, matplotlib, etc.
###############################################

# 2.1 Safe scratch setup (robust)
SCRATCH_DIR="${SLURM_TMPDIR:-${TMPDIR:-/tmp}/job_${SLURM_JOB_ID}}"
mkdir -p "$SCRATCH_DIR"

# 2.2 Export temp-related environment variables
export TMPDIR="$SCRATCH_DIR"
export SQLITE_TMPDIR="$SCRATCH_DIR"
export HDF5_USE_FILE_LOCKING=FALSE
export MPLCONFIGDIR="$SCRATCH_DIR/.mpl"
export SCRATCH_DIR  # IMPORTANT: expose for cli.py


###############################################
# 3. ARGUMENT PARSING
# Purpose:
#   - Accept YAML config as first argument
#   - Abort if file missing
###############################################
CONFIG_FILE="${1:-input_new.yaml}"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Configuration file '$CONFIG_FILE' does not exist."
    exit 1
fi
echo "Using configuration file: $CONFIG_FILE"


###############################################
# 4. PREPARE INPUT FILES
# Purpose:
#   - Copy config and data files into scratch directory
#   - Ensure working directory is local to node
###############################################
CFG_BASENAME="$(basename "$CONFIG_FILE")"
cp "$CONFIG_FILE" "$SCRATCH_DIR"
cp *.npz "$SCRATCH_DIR" 2>/dev/null || true
cp *.xyz "$SCRATCH_DIR" 2>/dev/null || true

cd "$SCRATCH_DIR" || { echo "✘ Failed to cd to $SCRATCH_DIR"; exit 1; }
mkdir -p results

echo "→ DB (if created) will be at: $SCRATCH_DIR/results/$DB_NAME"

###############################################
# 5. GPU VISIBILITY CHECK (Sanity Info)
# Purpose:
#   - Verify GPU allocation before training
#   - Ensure CUDA_VISIBLE_DEVICES matches SLURM allocation
###############################################
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi || echo "nvidia-smi not available"


###############################################
# FIX: PyTorch 2.6 + e3nn constants.pt load
###############################################
cat > "$SCRATCH_DIR/sitecustomize.py" << 'PY'
try:
    import torch
    from torch.serialization import add_safe_globals
    add_safe_globals([slice])
except Exception:
    pass
PY
export PYTHONPATH="$SCRATCH_DIR:${PYTHONPATH}"

echo "SCRATCH_DIR=$SCRATCH_DIR"; ls -ld "$SCRATCH_DIR" "$SCRATCH_DIR/results" || true


###############################################
# 7. RUN TRAINING
# Purpose:
#   - Execute mlff_qd.training with given config
#   - Fail-fast if DDP configuration is invalid (handled in Python)
###############################################

srun --chdir="$SCRATCH_DIR" python -m mlff_qd.training --config "$CFG_BASENAME" "${@:2}" || echo "Training failed, proceeding with copy"

###############################################
# 8. COPY RESULTS BACK
# Purpose:
#   - Move standardized outputs, results, and benchmarks
#     back to original directory ($CDIR)
###############################################

JOB_OUT="$CDIR/$SLURM_JOB_ID"
mkdir -p "$JOB_OUT"

# Always copy the config used
cp -f "$SCRATCH_DIR/$CFG_BASENAME" "$JOB_OUT/" 2>/dev/null || true

# ------------------------------------------------------------------
# 1) HIGHEST PRIORITY: standardized bundle
# ------------------------------------------------------------------
if [ -d "$SCRATCH_DIR/standardized" ]; then
    echo "✔ Found standardized outputs – copying and exiting"
    cp -r "$SCRATCH_DIR/standardized" "$JOB_OUT/"
    exit 0
fi

# ------------------------------------------------------------------
# 2) NEXT PRIORITY: benchmark results
# ------------------------------------------------------------------
if [ -d "$SCRATCH_DIR/benchmark_results" ]; then
    echo "✔ Found benchmark results – copying and exiting"
    cp -r "$SCRATCH_DIR/benchmark_results" "$JOB_OUT/"
    [ -f "$SCRATCH_DIR/benchmark_summary.csv" ] && \
        cp "$SCRATCH_DIR/benchmark_summary.csv" "$JOB_OUT/"
    exit 0
fi

# ------------------------------------------------------------------
# 3) FALLBACK: raw outputs (NO standardization)
# ------------------------------------------------------------------
echo "⚠ No standardized or benchmark outputs – copying raw results"
echo "Copying ALL scratch contents back to $JOB_OUT"

# Copy everything in scratch to job output
cp -a "$SCRATCH_DIR/." "$JOB_OUT/"

###############################################
# 9. CLEANUP
# Purpose:
#   - Safely remove scratch directory after copying results
#   - Prevent accidental deletion of system paths
###############################################
if [[ -n "$SCRATCH_DIR" && -d "$SCRATCH_DIR" && "$SCRATCH_DIR" != "/" && "$SCRATCH_DIR" != "/tmp" ]]; then
  rm -rf "$SCRATCH_DIR"
else
  echo "⚠️  Skip cleanup; unsafe SCRATCH_DIR='$SCRATCH_DIR'"
fi