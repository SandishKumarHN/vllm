#!/usr/bin/env bash
# run_all.sh - end-to-end orchestrator.
#
# Pipeline:
#   00 setup_env  -> 01 build  -> 02 baseline  -> [APPLY PATCH]  ->
#   03 rebuild    -> 04 optimized -> 05 compare
#
# Flags:
#   --skip-setup      skip 00_setup_env.sh
#   --skip-build      skip 01_build_vllm.sh (use already-installed vllm)
#   --skip-baseline   skip 02_baseline.sh
#   --skip-patch      skip the interactive "apply patch" pause
#   --patch-cmd CMD   run CMD to apply the patch instead of pausing
#   --no-rebuild      skip 03_rebuild_after_patch.sh (Python-only patch)
#   --skip-optimized  skip 04_optimized.sh
#   --help            show this help
#
# Any BENCH_* / TORCH_CUDA_ARCH_LIST / MAX_JOBS env override still applies.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SKIP_SETUP=0; SKIP_BUILD=0; SKIP_BASELINE=0
SKIP_PATCH=0; PATCH_CMD=""; NO_REBUILD=0; SKIP_OPT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-setup)     SKIP_SETUP=1 ;;
    --skip-build)     SKIP_BUILD=1 ;;
    --skip-baseline)  SKIP_BASELINE=1 ;;
    --skip-patch)     SKIP_PATCH=1 ;;
    --patch-cmd)      PATCH_CMD="$2"; shift ;;
    --no-rebuild)     NO_REBUILD=1 ;;
    --skip-optimized) SKIP_OPT=1 ;;
    --help|-h)        sed -n '1,30p' "$0"; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
  shift
done

ensure_dirs
print_env

step() { log "================  $*  ================"; }

(( SKIP_SETUP ))    || { step "00 setup_env";       "${SCRIPT_DIR}/00_setup_env.sh"; }
(( SKIP_BUILD ))    || { step "01 build_vllm";      "${SCRIPT_DIR}/01_build_vllm.sh"; }
(( SKIP_BASELINE )) || { step "02 baseline bench";  "${SCRIPT_DIR}/02_baseline.sh"; }

if (( ! SKIP_PATCH )); then
  if [[ -n "${PATCH_CMD}" ]]; then
    step "Applying patch: ${PATCH_CMD}"
    bash -c "${PATCH_CMD}"
  else
    step "APPLY YOUR PATCH NOW"
    log "Modify files under ${VLLM_ROOT}/ as needed, then press ENTER to continue."
    read -r -p "Press ENTER once the patch is applied ... " _
  fi
fi

(( NO_REBUILD )) || { step "03 rebuild after patch"; "${SCRIPT_DIR}/03_rebuild_after_patch.sh"; }
(( SKIP_OPT ))   || { step "04 optimized bench";     "${SCRIPT_DIR}/04_optimized.sh"; }

step "05 compare"
"${SCRIPT_DIR}/05_compare.sh"

log "DONE. Results in ${RESULTS_DIR}/  ;  logs in ${LOG_DIR}/"
