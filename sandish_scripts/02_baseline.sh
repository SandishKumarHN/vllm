#!/usr/bin/env bash
# 02_baseline.sh - run baseline benchmark, write baseline.json to RESULTS_DIR.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
ensure_dirs
activate_venv

OUT="${RESULTS_DIR}/baseline.json"
LOG="${LOG_DIR}/02_baseline_$(date +%Y%m%d_%H%M%S).log"

log "Baseline JSON  -> ${OUT}"
log "Baseline log   -> ${LOG}"
log "Model=${BENCH_MODEL}  TP=${BENCH_TP}  batch=${BENCH_BATCH}  in=${BENCH_INPUT_LEN}  out=${BENCH_OUTPUT_LEN}  iters=${BENCH_ITERS}"

# Pre-flight: # GPUs must >= TP size
N_GPU=$(nvidia-smi -L | wc -l)
(( N_GPU >= BENCH_TP )) || die "Need ${BENCH_TP} GPUs, found ${N_GPU}. Lower BENCH_TP or move to a bigger box."

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

log "OK - baseline complete: ${OUT}"
