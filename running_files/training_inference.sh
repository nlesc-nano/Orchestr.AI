#!/bin/bash
#SBATCH --job-name=train
#SBATCH --time=1-00:00:00  
#SBATCH -c 32 
#SBATCH -p medium  
#SBATCH --mem=64GB 
#SBATCH --gres=gpu:a100
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err

# 1) Load your env
conda deactivate 
conda activate env-name # make sure that the environment name is the one that was created

# 2) Remember original dir
CDIR=$(pwd)

# 3) Use SLURM's node-local tmp (guaranteed local disk)
SCRATCH_DIR=${TMPDIR:-/scratch/$SLURM_JOB_ID}
mkdir -p "$SCRATCH_DIR"

# 4) Copy code & static data (no .db yet)
CONFIG_FILE=${1:-input.yaml}
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Error: config '$CONFIG_FILE' not found"
  exit 1
fi
echo "Using config: $CONFIG_FILE"

cp "$CONFIG_FILE" *.npz *.hdf5 "$SCRATCH_DIR"

# 5) cd into the local node scratch
cd "$SCRATCH_DIR"

# 6) Prepare results folder on local disk
mkdir -p results

# 7) Extract your DB name from the YAML
DB_NAME=$(grep -Po "(?<=database_name:\s').*(?=')" "$CONFIG_FILE")
if [ -z "$DB_NAME" ]; then
  echo "Error: couldn't parse database_name"
  exit 1
fi

# 8) Patch the YAML so logging.folder → ./results
sed -i "s|^\(\s*folder:\s*\).*|\1'./results'|g" "$CONFIG_FILE"

echo "→ logging.folder set to './results'"
echo "→ DB will be created at: $SCRATCH_DIR/results/$DB_NAME"

# 9) Run training & inference in the node-local scratch
srun --chdir="$SCRATCH_DIR" python -m mlff_qd.training --config "$CONFIG_FILE"
if [ $? -eq 0 ]; then
  echo "✔ Training succeeded — running inference"
  srun --chdir="$SCRATCH_DIR" python -m mlff_qd.training.inference --config "$CONFIG_FILE"
else
  echo "✘ Training failed — skipping inference"
fi

# 10) Copy back all outputs (including results/*.db)
cp -r ./results *.npz *.pkl *.csv "$CDIR"

# 11) Clean up
rm -rf "$SCRATCH_DIR"



