#!/usr/bin/env bash
set -euo pipefail

# =========================
# ENV NAMES (user override supported)
# =========================
CORE_ENV="${MLFFQD_CORE_CONDA_ENV:-mlffqd-core}"
NEQUIP_ENV="${MLFFQD_NEQUIP_CONDA_ENV:-mlffqd-nequip}"
MACE_ENV="${MLFFQD_MACE_CONDA_ENV:-mlffqd-mace}"

# =========================
# FILE PATHS
# =========================
CORE_YAML="${MLFFQD_CORE_YAML:-envs/mlff_qd-core.yaml}"
NEQUIP_YAML="${MLFFQD_NEQUIP_YAML:-envs/mlff_qd-nequip.yaml}"
MACE_YAML="${MLFFQD_MACE_YAML:-envs/mlff_qd-mace.yaml}"

PRINT_VERSIONS="${PRINT_VERSIONS:-1}"

need_cmd () {
  command -v "$1" >/dev/null 2>&1 || { echo "[ERROR] Missing: $1"; exit 1; }
}

# Create/update using micromamba
create_or_update_env_from_yaml () {
  local ENVNAME="$1"
  local YAMLFILE="$2"

  if micromamba env list | awk '{print $1}' | grep -qx "$ENVNAME"; then
    echo "[INFO] Env '$ENVNAME' exists -> updating from $YAMLFILE"
    micromamba env update -n "$ENVNAME" -f "$YAMLFILE" --prune
  else
    echo "[INFO] Creating env '$ENVNAME' from $YAMLFILE"
    micromamba env create -n "$ENVNAME" -f "$YAMLFILE"
  fi
}

install_mlff_qd_editable () {
  local ENVNAME="$1"
  echo "[INFO] Installing MLFF_QD into '$ENVNAME' (editable)"
  micromamba run -n "$ENVNAME" python -m pip install -U pip
  micromamba run -n "$ENVNAME" python -m pip install -e .
}

print_versions () {
  local ENVNAME="$1"
  local TYPE="$2"

  echo ""
  echo "=== [$ENVNAME] Versions ==="
  micromamba run -n "$ENVNAME" python -c "import sys; print('python', sys.version.split()[0])"
  micromamba run -n "$ENVNAME" python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda); print('cuda_available', torch.cuda.is_available())" 2>/dev/null || true

  if [[ "$TYPE" == "core" ]]; then
    micromamba run -n "$ENVNAME" python -c "import schnetpack; print('schnetpack', schnetpack.__version__)" 2>/dev/null || true
  elif [[ "$TYPE" == "nequip" ]]; then
    micromamba run -n "$ENVNAME" python -c "import nequip; print('nequip', nequip.__version__)" 2>/dev/null || true
  elif [[ "$TYPE" == "mace" ]]; then
    micromamba run -n "$ENVNAME" python -c "import mace; print('mace', mace.__version__)" 2>/dev/null || true
  fi

  echo "==========================="
  echo ""
}


need_cmd micromamba

for f in "$CORE_YAML" "$NEQUIP_YAML" "$MACE_YAML"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing YAML file: $f"; exit 1; }
done

# ======== START SETUP ========
echo "[SETUP] Envs (micromamba):"
echo "  CORE   = $CORE_ENV   ($CORE_YAML)"
echo "  NEQUIP = $NEQUIP_ENV ($NEQUIP_YAML)"
echo "  MACE   = $MACE_ENV   ($MACE_YAML)"
echo ""

# ---------- CORE ------------
echo "[1/3] Core env"
create_or_update_env_from_yaml "$CORE_ENV" "$CORE_YAML"
install_mlff_qd_editable "$CORE_ENV"

# ---------- NEQUIP ------------
echo "[2/3] NequIP env"
create_or_update_env_from_yaml "$NEQUIP_ENV" "$NEQUIP_YAML"
install_mlff_qd_editable "$NEQUIP_ENV"

# ---------- MACE ------------
echo "[3/3] MACE env"
create_or_update_env_from_yaml "$MACE_ENV" "$MACE_YAML"
install_mlff_qd_editable "$MACE_ENV"

# ========== VERIFY (PRINT VERSIONS) =============
if [[ "$PRINT_VERSIONS" == "1" ]]; then
  echo ""
  echo "[VERIFY] Printing versions for all environments..."
  print_versions "$CORE_ENV" core
  print_versions "$NEQUIP_ENV" nequip
  print_versions "$MACE_ENV" mace
fi

# ============ EXPORT ENV VARS FOR DISPATCH ============
ENV_EXPORT_FILE="scripts/mlff_qd_env.sh"
mkdir -p scripts

cat > "$ENV_EXPORT_FILE" <<EOF
export MLFFQD_CORE_CONDA_ENV="$CORE_ENV"
export MLFFQD_NEQUIP_CONDA_ENV="$NEQUIP_ENV"
export MLFFQD_MACE_CONDA_ENV="$MACE_ENV"
EOF

echo ""
echo "[INFO] Wrote env variables to: $ENV_EXPORT_FILE"
echo "To enable dispatch in your shell, run:"
echo "  source $ENV_EXPORT_FILE"
echo ""
echo "[DONE] micromamba environments created/updated."
