#!/usr/bin/env bash
# 01_build_vllm.sh - full from-source vLLM build with CUDA kernel compilation.
#
# Builds:  uv pip install -e .  (with --no-build-isolation so the venv torch
# is reused, avoiding a multi-GB download for build isolation).
# Also runs the helper that emits CMakeUserPresets.json so subsequent
# incremental rebuilds (03_rebuild_after_patch.sh) work via cmake/ninja.
#
# Usage:  ./scripts/01_build_vllm.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

ensure_dirs
activate_venv
print_env

cd "${VLLM_ROOT}"

LOG="${LOG_DIR}/01_build_$(date +%Y%m%d_%H%M%S).log"
log "Build log -> ${LOG}"
log "Starting full vLLM build (this can take 30-60 min on a fresh ccache)."

# Use the existing torch already installed by 00_setup_env.sh.
# `python use_existing_torch.py` rewrites requirements so build won't pull a
# different torch version.
python use_existing_torch.py

# Build deps (re-affirm, in case venv was reset).
uv pip install -r requirements/build/cuda.txt --torch-backend="${TORCH_BACKEND}"

# Time the build and tee output to log.
START=$(date +%s)
set -o pipefail
uv pip install --no-build-isolation -e . --torch-backend="${TORCH_BACKEND}" 2>&1 | tee "${LOG}"
END=$(date +%s)
log "Build finished in $(( (END - START) / 60 )) min $(( (END - START) % 60 )) s"

# Generate CMake presets so we can do fast incremental rebuilds after a patch.
log "Generating CMakeUserPresets.json for incremental rebuilds ..."
python tools/generate_cmake_presets.py --force-overwrite || warn "preset gen failed (incremental rebuilds may be slower)"

# Sanity check the installed vllm
python - <<'PY'
import vllm, torch, importlib
print(f"vllm        = {vllm.__version__}  ({vllm.__file__})")
print(f"torch       = {torch.__version__} cuda={torch.version.cuda}")
import vllm._C as _c          # noqa: F401  - core kernels
print("vllm._C    : OK")
try:
    import vllm._moe_C as _m  # noqa: F401  - MoE kernels (needed for DeepSeek)
    print("vllm._moe_C: OK")
except Exception as e:
    print(f"vllm._moe_C: MISSING ({e})")
PY

log "OK - vLLM installed from source. Next: ./scripts/02_baseline.sh"
