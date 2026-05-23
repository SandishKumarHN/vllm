#!/usr/bin/env bash
# 03_rebuild_after_patch.sh - incremental rebuild after a CUDA/C++ kernel patch.
#
# Uses CMake/Ninja (much faster than re-running `pip install -e .`) because
# the editable install + CMakeUserPresets.json from step 01 keeps everything
# wired up.  Falls back to a full `pip install -e .` if the preset is missing.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

ensure_dirs
activate_venv
cd "${VLLM_ROOT}"

LOG="${LOG_DIR}/03_rebuild_$(date +%Y%m%d_%H%M%S).log"
log "Rebuild log -> ${LOG}"

START=$(date +%s)

if [[ -f "${VLLM_ROOT}/CMakeUserPresets.json" ]]; then
  log "CMakeUserPresets.json detected - doing incremental cmake build."
  # configure (cheap if already configured) + build & install
  cmake --preset release 2>&1 | tee -a "${LOG}"
  cmake --build --preset release --target install 2>&1 | tee -a "${LOG}"
else
  warn "No CMakeUserPresets.json - falling back to full pip rebuild."
  uv pip install --no-build-isolation -e . --torch-backend="${TORCH_BACKEND}" 2>&1 | tee -a "${LOG}"
fi

END=$(date +%s)
log "Rebuild finished in $(( (END - START) / 60 )) min $(( (END - START) % 60 )) s"

# Quick import sanity check (catches undefined-symbol issues from kernel patches)
python - <<'PY'
import importlib, sys
for m in ("vllm._C", "vllm._moe_C"):
    try:
        importlib.import_module(m)
        print(f"{m}: OK")
    except Exception as e:
        print(f"{m}: FAILED -> {e}", file=sys.stderr)
        sys.exit(1)
PY

log "OK - rebuild ready. Next: ./scripts/04_optimized.sh"
