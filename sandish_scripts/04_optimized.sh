#!/usr/bin/env bash
# 04_optimized.sh - run the post-patch benchmark, write optimized.json.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
ensure_dirs
activate_venv

OUT="${RESULTS_DIR}/optimized.json"
LOG="${LOG_DIR}/04_optimized_$(date +%Y%m%d_%H%M%S).log"

log "Optimized JSON -> ${OUT}"
log "Optimized log  -> ${LOG}"

N_GPU=$(nvidia-smi -L | wc -l)
(( N_GPU >= BENCH_TP )) || die "Need ${BENCH_TP} GPUs, found ${N_GPU}."

# shellcheck disable=SC2086
vllm bench latency \
  --model "${BENCH_MODEL}" \
  --tensor-parallel-size "${BENCH_TP}" \
  --batch-size "${BENCH_BATCH}" \
  --input-len "${BENCH_INPUT_LEN}" \
  --output-len "${BENCH_OUTPUT_LEN}" \
  --num-iters "${BENCH_ITERS}" \
  ${BENCH_EXTRA_FLAGS} \
  --output-json "${OUT}" 2>&1 | tee "${LOG}"

log "OK - optimized complete: ${OUT}"
log "Next: ./scripts/05_compare.sh"
