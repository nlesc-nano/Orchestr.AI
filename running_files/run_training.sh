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
export PYTHONUNBUFFERED=1

CDIR="$(pwd)"


###############################################
# 2. SCRATCH DIRECTORY SETUP
# Purpose:
#   - Create job-local scratch for fast I/O
#   - Ensure safe temp usage for SQLite, HDF5, matplotlib, etc.
###############################################

# 2.1 Optional default SLURM tmpdir (commented)
# SCRATCH_DIR=${TMPDIR:-/scratch/${SLURM_JOB_ID:-$$}}
# mkdir -p "$SCRATCH_DIR"

# 2.2 Safe scratch setup (custom)
BASE_TMP=${TMPDIR:-/scratch}
SCRATCH_DIR="$BASE_TMP/job_${SLURM_JOB_ID:-$$}"
mkdir -p "$SCRATCH_DIR"

# 2.3 Export temp-related environment variables
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


###############################################
# 5. DATABASE LOGIC
# Purpose:
#   - Extract 'database_name' from YAML (if defined)
#   - Default to 'Database.db' if not found
###############################################
DB_NAME="Database.db"
DB_FROM_YAML=$(grep -E '^[[:space:]]*database_name:' "$CFG_BASENAME" | head -n1 \
  | sed -E 's/\r$//' \
  | sed -E 's/^[[:space:]]*database_name:[[:space:]]*//; s/[[:space:]]+#.*$//; s/^[[:space:]]*["'\''"]?//; s/["'\''"]?[[:space:]]*$//')
if [ -n "$DB_FROM_YAML" ] && [ "$DB_FROM_YAML" != "null" ] && [ "$DB_FROM_YAML" != "~" ]; then
  DB_NAME="$DB_FROM_YAML"
  echo "→ Parsed database_name from YAML: $DB_NAME"
else
  echo "→ No valid database_name in YAML; using default: $DB_NAME"
fi

# Force logging folder to local scratch results
sed -i "s|^\(\s*folder:\s*\).*|\1'./results'|g" "$CFG_BASENAME" 2>/dev/null || true
echo "→ logging.folder set to './results'"
echo "→ DB (if created) will be at: $SCRATCH_DIR/results/$DB_NAME"



###############################################
# 6. GPU VISIBILITY CHECK (Sanity Info)
# Purpose:
#   - Verify GPU allocation before training
#   - Ensure CUDA_VISIBLE_DEVICES matches SLURM allocation
###############################################
echo "SLURM_JOB_ID=$SLURM_JOB_ID"
echo "SLURM_JOB_GPUS=$SLURM_JOB_GPUS"
echo "SLURM_GPUS_ON_NODE=$SLURM_GPUS_ON_NODE"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi 


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

# Prefer standardized bundle (copy the directory, not its contents)
if [ -d "$SCRATCH_DIR/standardized" ]; then
    cp -r "$SCRATCH_DIR/standardized" "$JOB_OUT/"
    cp -f "$SCRATCH_DIR/$CFG_BASENAME" "$JOB_OUT/" 2>/dev/null || true
fi

# Always copy results dirs (DB, logs) for convenience
[ -d "$SCRATCH_DIR/results" ] && cp -r "$SCRATCH_DIR/results" "$JOB_OUT/"

# Benchmarks if present
if [ -d "$SCRATCH_DIR/benchmark_results" ]; then
    cp -r "$SCRATCH_DIR/benchmark_results" "$JOB_OUT/"
    [ -f "$SCRATCH_DIR/benchmark_summary.csv" ] && cp "$SCRATCH_DIR/benchmark_summary.csv" "$JOB_OUT/"
fi

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
