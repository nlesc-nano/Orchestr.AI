#!/usr/bin/env bash
set -euo pipefail

# =========================
# ENV NAMES (user override supported)
# =========================
CORE_ENV="${MLFFQD_CORE_CONDA_ENV:-mlffqd-corex_micromamba_uvZ}"
NEQUIP_ENV="${MLFFQD_NEQUIP_CONDA_ENV:-mlffqd-nequipx_micromamba_uvZ}"
MACE_ENV="${MLFFQD_MACE_CONDA_ENV:-mlffqd-macex_micromamba_uvZ}"

# =========================
# FILE PATHS
# =========================
CORE_YAML="${MLFFQD_CORE_YAML:-envs/mlff_qd-core-uv.yaml}"
NEQUIP_YAML="${MLFFQD_NEQUIP_YAML:-envs/mlff_qd-nequip-uv.yaml}"
MACE_YAML="${MLFFQD_MACE_YAML:-envs/mlff_qd-mace-uv.yaml}"

CORE_REQ="${MLFFQD_CORE_REQ:-requirements/requirements-core.txt}"
NEQUIP_REQ="${MLFFQD_NEQUIP_REQ:-requirements/requirements-nequip.txt}"
MACE_REQ="${MLFFQD_MACE_REQ:-requirements/requirements-mace.txt}"
UV_BIN="${UV_BIN:-uv}"

PRINT_VERSIONS="${PRINT_VERSIONS:-1}"

need_cmd () {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] Missing command: $1"
    exit 1
  }
}

create_or_update_env () {
  local ENVNAME="$1"
  local YAMLFILE="$2"

  if micromamba env list | awk '{print $1}' | grep -qx "$ENVNAME"; then
    echo "[INFO] Updating env '$ENVNAME' from $YAMLFILE"
    micromamba env update -y -n "$ENVNAME" -f "$YAMLFILE" --prune
  else
    echo "[INFO] Creating env '$ENVNAME' from $YAMLFILE"
    micromamba create -y -n "$ENVNAME" -f "$YAMLFILE"
  fi
}

ensure_uv () {
  local ENVNAME="$1"
  echo "[INFO] Ensuring uv in '$ENVNAME'"

  if ! micromamba run -n "$ENVNAME" "$UV_BIN" --version >/dev/null 2>&1; then
    micromamba run -n "$ENVNAME" python -m pip install -U uv
  fi
}

install_requirements_uv () {
  local ENVNAME="$1"
  local REQFILE="$2"

  if [[ -f "$REQFILE" ]]; then
    echo "[INFO] Installing requirements for '$ENVNAME' via uv → $REQFILE"
    micromamba run -n "$ENVNAME" "$UV_BIN" pip install -r "$REQFILE"
  else
    echo "[WARN] Missing requirements file: $REQFILE"
  fi
}

install_mlff_qd () {
  local ENVNAME="$1"
  echo "[INFO] Installing MLFF_QD (editable) in '$ENVNAME'"
  micromamba run -n "$ENVNAME" "$UV_BIN" pip install -e .
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

  micromamba run -n "$ENVNAME" "$UV_BIN" --version 2>/dev/null || true

  echo "==========================="
  echo ""
}


need_cmd micromamba
need_cmd "$UV_BIN"

for f in "$CORE_YAML" "$NEQUIP_YAML" "$MACE_YAML" "$CORE_REQ" "$NEQUIP_REQ" "$MACE_REQ"; do
  [[ -f "$f" ]] || { echo "[ERROR] Missing file: $f"; exit 1; }
done

# ======== START SETUP ========
echo "[SETUP] micromamba + uv mode"
echo "CORE   = $CORE_ENV"
echo "NEQUIP = $NEQUIP_ENV"
echo "MACE   = $MACE_ENV"
echo ""

# ---------- CORE ------------
echo "[1/3] CORE ENV"
create_or_update_env "$CORE_ENV" "$CORE_YAML"
ensure_uv "$CORE_ENV"
install_requirements_uv "$CORE_ENV" "$CORE_REQ"
install_mlff_qd "$CORE_ENV"

# ---------- NEQUIP ------------
echo "[2/3] NEQUIP ENV"
create_or_update_env "$NEQUIP_ENV" "$NEQUIP_YAML"
ensure_uv "$NEQUIP_ENV"
install_requirements_uv "$NEQUIP_ENV" "$NEQUIP_REQ"
install_mlff_qd "$NEQUIP_ENV"

# ---------- MACE ------------
echo "[3/3] MACE ENV"
create_or_update_env "$MACE_ENV" "$MACE_YAML"
ensure_uv "$MACE_ENV"
install_requirements_uv "$MACE_ENV" "$MACE_REQ"
install_mlff_qd "$MACE_ENV"


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
echo "[INFO] Env variables written to: $ENV_EXPORT_FILE"
echo "Run:"
echo "  source $ENV_EXPORT_FILE"
echo ""
echo "[DONE] Setup complete (micromamba + uv)"