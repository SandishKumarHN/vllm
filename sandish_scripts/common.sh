#!/usr/bin/env bash
# Common environment / helpers sourced by every script in scripts/.
# Edit these defaults for your machine. Each value can be overridden by
# exporting the variable BEFORE invoking any script.

set -euo pipefail

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
VLLM_ROOT="${VLLM_ROOT:-/home/sandish/vllm}"
VENV_DIR="${VENV_DIR:-${VLLM_ROOT}/.venv}"
RESULTS_DIR="${RESULTS_DIR:-${VLLM_ROOT}/bench_results}"
LOG_DIR="${LOG_DIR:-${VLLM_ROOT}/build_logs}"

# ---------------------------------------------------------------------------
# CUDA / build
# ---------------------------------------------------------------------------
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
# Ensure user-local bin (uv install target) and CUDA bin are on PATH.
export PATH="${HOME}/.local/bin:${CUDA_HOME}/bin:${PATH}"

# Some shell rc files (notably Meta's dotsync) export CUDA_NVCC_EXECUTABLE to
# a stale ccache shim path. PyTorch's bundled FindCUDA.cmake trusts that env
# var blindly and bails with "Failed to execute '<path> --version'". Strip the
# stale value if it points to something that doesn't exist; force CUDA_HOME
# nvcc when the env var is missing so cmake never autodetects a stale path.
if [[ -n "${CUDA_NVCC_EXECUTABLE:-}" && ! -x "${CUDA_NVCC_EXECUTABLE}" ]]; then
  unset CUDA_NVCC_EXECUTABLE
fi
export CUDA_NVCC_EXECUTABLE="${CUDA_NVCC_EXECUTABLE:-${CUDA_HOME}/bin/nvcc}"
export CUDA_BIN_PATH="${CUDA_BIN_PATH:-${CUDA_HOME}/bin}"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_TOOLKIT_ROOT_DIR:-${CUDA_HOME}}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"

# Build for A100 (8.0) + H100 (9.0). Set TORCH_CUDA_ARCH_LIST in env to override.
# Use H100-only (9.0) on the benchmark box to cut build time roughly in half.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0+PTX}"

# Pin torch CUDA backend. By default we prefer cu130 because that's the
# variant whose nvidia-* pip wheels ship with full headers + libs (the cu128
# wheels at the time of writing only ship libcudart, not headers, which
# breaks the from-source build on Meta devservers where /usr/local/cuda is a
# compiler-only stub). Override TORCH_BACKEND in env if you have a real
# CUDA 12.x toolkit installed system-wide.
export TORCH_BACKEND="${TORCH_BACKEND:-cu130}"

# Parallelism. nvcc is memory-hungry; keep MAX_JOBS modest if RAM is tight.
NPROC="$(nproc)"
export MAX_JOBS="${MAX_JOBS:-$(( NPROC > 16 ? 16 : NPROC ))}"
export NVCC_THREADS="${NVCC_THREADS:-4}"

# Enable ccache automatically if present.
if command -v ccache >/dev/null 2>&1; then
  export CMAKE_C_COMPILER_LAUNCHER=ccache
  export CMAKE_CXX_COMPILER_LAUNCHER=ccache
  export CMAKE_CUDA_COMPILER_LAUNCHER=ccache
  export CCACHE_NOHASHDIR="true"
fi

# ---------------------------------------------------------------------------
# Benchmark defaults (override via env)
# ---------------------------------------------------------------------------
BENCH_MODEL="${BENCH_MODEL:-deepseek-ai/DeepSeek-V3}"
BENCH_TP="${BENCH_TP:-8}"
BENCH_BATCH="${BENCH_BATCH:-1}"
BENCH_INPUT_LEN="${BENCH_INPUT_LEN:-128}"
BENCH_OUTPUT_LEN="${BENCH_OUTPUT_LEN:-128}"
BENCH_ITERS="${BENCH_ITERS:-100}"
BENCH_EXTRA_FLAGS="${BENCH_EXTRA_FLAGS:---enforce-eager}"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
log()  { printf '\033[1;36m[%(%H:%M:%S)T]\033[0m %s\n' -1 "$*"; }
warn() { printf '\033[1;33m[%(%H:%M:%S)T] WARN\033[0m %s\n' -1 "$*" >&2; }
die()  { printf '\033[1;31m[%(%H:%M:%S)T] ERR\033[0m  %s\n' -1 "$*" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Synthetic CUDA assembly was removed. /usr/local/cuda* on Meta devservers is
# a compiler-only stub (no cuda.h, no libcudart.so). Attempts to stitch a
# working toolkit from pip wheels + system nvcc hit nvcc's _HERE_/TOP
# resolution and lead to a workaround spiral. Run the build on a host with a
# properly installed CUDA toolkit (an OnDemand devGPU, your H100 box, etc.).
# ---------------------------------------------------------------------------

activate_venv() {
  [[ -f "${VENV_DIR}/bin/activate" ]] || die "venv missing at ${VENV_DIR}. Run 00_setup_env.sh first."
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
}

ensure_dirs() { mkdir -p "${RESULTS_DIR}" "${LOG_DIR}"; }

print_env() {
  log "VLLM_ROOT             = ${VLLM_ROOT}"
  log "VENV_DIR              = ${VENV_DIR}"
  log "CUDA_HOME             = ${CUDA_HOME}"
  log "TORCH_CUDA_ARCH_LIST  = ${TORCH_CUDA_ARCH_LIST}"
  log "MAX_JOBS / NVCC       = ${MAX_JOBS} / ${NVCC_THREADS}"
  log "TORCH_BACKEND         = ${TORCH_BACKEND}"
  log "ccache                = $(command -v ccache || echo 'NOT FOUND')"
  log "nvcc                  = $(command -v nvcc  || echo 'NOT FOUND')"
  log "GPUs                  = $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
}
