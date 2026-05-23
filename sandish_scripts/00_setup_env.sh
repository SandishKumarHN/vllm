#!/usr/bin/env bash
# 00_setup_env.sh - one-time environment setup
#   - install uv if missing
#   - create venv at $VENV_DIR (Python 3.12)
#   - install build deps (cmake, ninja, packaging, setuptools, torch, etc.)
#   - install ccache via system pkg if available
#
# Usage:  ./scripts/00_setup_env.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

ensure_dirs
print_env

cd "${VLLM_ROOT}"

# --- 1. ccache (best-effort) -------------------------------------------------
if ! command -v ccache >/dev/null 2>&1; then
  warn "ccache not found - rebuilds will be slower."
  if command -v apt-get >/dev/null 2>&1 && [[ $EUID -eq 0 ]]; then
    apt-get update && apt-get install -y ccache
  elif command -v dnf >/dev/null 2>&1 && [[ $EUID -eq 0 ]]; then
    dnf install -y ccache
  else
    warn "Install ccache manually (apt/dnf/conda) for faster rebuilds."
  fi
fi

# --- 2. uv (preferred) -------------------------------------------------------
if ! command -v uv >/dev/null 2>&1; then
  log "Installing uv (Python package manager) ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="${HOME}/.local/bin:${PATH}"
fi
command -v uv >/dev/null 2>&1 || die "uv install failed"
log "uv version: $(uv --version)"

# --- 3. virtual env ----------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
  log "Creating venv at ${VENV_DIR} (Python 3.12) ..."
  uv venv --python 3.12 --seed "${VENV_DIR}"
fi
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
log "Python: $(python --version)  @  $(which python)"

# --- 4. build dependencies ---------------------------------------------------
log "Installing build dependencies (requirements/build/cuda.txt) ..."
uv pip install -r "${VLLM_ROOT}/requirements/build/cuda.txt" --torch-backend="${TORCH_BACKEND}"

# numpy isn't in the build requirements but torch warns without it, and several
# vllm runtime paths need it. Install it now.
log "Installing numpy ..."
uv pip install numpy

# --- 5. CUDA toolkit check (advisory) ---------------------------------------
# vLLM's C++/CUDA build needs a full CUDA toolkit: nvcc + headers (cuda.h,
# cuda_runtime.h) + libs (libcudart.so). Meta devservers ship a compiler-only
# stub at /usr/local/cuda - the build WILL fail at cmake configure on these
# hosts. Detect and warn loudly.
if [[ ! -f "${CUDA_HOME}/include/cuda_runtime.h" || ! -f "${CUDA_HOME}/lib64/libcudart.so" ]]; then
  warn "================================================================"
  warn "Incomplete CUDA toolkit at ${CUDA_HOME}"
  warn "  - cuda_runtime.h : $([[ -f ${CUDA_HOME}/include/cuda_runtime.h ]] && echo OK || echo MISSING)"
  warn "  - libcudart.so   : $([[ -f ${CUDA_HOME}/lib64/libcudart.so   ]] && echo OK || echo MISSING)"
  warn ""
  warn "01_build_vllm.sh will FAIL on this host. Build vLLM on a properly"
  warn "provisioned GPU box (e.g. your 8xH100 target, an OnDemand devGPU,"
  warn "or a sandbox with full CUDA toolkit installed)."
  warn "================================================================"
fi

# Confirm torch sees CUDA
python - <<'PY'
import torch
print(f"torch       = {torch.__version__}")
print(f"cuda avail  = {torch.cuda.is_available()}")
print(f"cuda built  = {torch.version.cuda}")
print(f"nccl        = {torch.cuda.nccl.version() if torch.cuda.is_available() else 'n/a'}")
print(f"device cnt  = {torch.cuda.device_count()}")
PY

log "OK - environment ready. Next: ./scripts/01_build_vllm.sh"
