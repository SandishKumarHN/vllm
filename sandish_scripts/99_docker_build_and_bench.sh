#!/usr/bin/env bash
# 99_docker_build_and_bench.sh
#
# Runs the entire build + benchmark pipeline inside a CUDA-devel container
# (nvidia/cuda:12.8.1-devel-ubuntu22.04 by default). This sidesteps the
# "no CUDA toolkit on host" problem on Meta devservers.
#
# Defaults assume Podman (Meta devserver). Override CONTAINER_CMD=docker if
# you have docker installed instead.
#
# Usage:
#   ./scripts/99_docker_build_and_bench.sh            # baseline-only build + bench
#   ./scripts/99_docker_build_and_bench.sh --patch /path/to/patch.diff
#   ./scripts/99_docker_build_and_bench.sh --shell   # drop into container shell
#
# IMPORTANT: The host still needs nvidia-smi + a real GPU + CDI / nvidia-container-toolkit
# configured for the container runtime.  On Meta devservers see:
#   feature install msl_docker        # docker
#   devfeature install k8s_tools      # podman
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CONTAINER_CMD="${CONTAINER_CMD:-podman}"
IMAGE="${IMAGE:-nvcr.io/nvidia/cuda:12.8.1-devel-ubuntu22.04}"
PATCH_FILE=""
SHELL_ONLY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --patch) PATCH_FILE="$2"; shift ;;
    --shell) SHELL_ONLY=1 ;;
    --image) IMAGE="$2"; shift ;;
    --cmd)   CONTAINER_CMD="$2"; shift ;;
    --help|-h) sed -n '1,30p' "$0"; exit 0 ;;
    *) die "unknown flag: $1" ;;
  esac
  shift
done

command -v "${CONTAINER_CMD}" >/dev/null || \
  die "${CONTAINER_CMD} not on PATH. Try: feature install msl_docker (docker) or devfeature install k8s_tools (podman)."

# GPU flag differs between podman and docker.
if [[ "${CONTAINER_CMD}" == "podman" ]]; then
  GPU_FLAG="--device nvidia.com/gpu=all --security-opt label=disable"
else
  GPU_FLAG="--gpus all"
fi

# Host paths to bind-mount.
HOST_VLLM="${VLLM_ROOT}"
HOST_CACHE="${HOME}/.cache"
mkdir -p "${HOST_CACHE}/huggingface" "${HOST_CACHE}/uv"

# Inside-container script body (heredoc kept short; calls our scripts).
read -r -d '' IN_CONTAINER <<'EOSH' || true
set -euo pipefail
echo "=== container info ==="
uname -a
nvidia-smi -L || { echo "GPU not visible in container"; exit 1; }
nvcc --version | tail -2

cd /work/vllm

# OS deps the nvidia/cuda image lacks
apt-get update -qq
apt-get install -y --no-install-recommends \
  python3 python3-pip python3-venv python3-dev \
  build-essential git curl ccache pkg-config >/dev/null

# Reuse host venv if it works in container; otherwise rebuild.
if [[ ! -x .venv/bin/python ]] || ! .venv/bin/python -c 'import sys' 2>/dev/null; then
  rm -rf .venv
  python3 -m venv .venv
  .venv/bin/pip install -U pip
fi

# Install uv inside venv (no curl-to-shell needed)
.venv/bin/pip install -q uv

export PATH=/work/vllm/.venv/bin:$PATH
export VLLM_ROOT=/work/vllm
export VENV_DIR=/work/vllm/.venv
export CUDA_HOME=/usr/local/cuda

# Run the standard pipeline. Patch & rebuild are optional (driven by env).
./scripts/00_setup_env.sh
./scripts/01_build_vllm.sh

if [[ -n "${PATCH_FILE_IN:-}" ]]; then
  echo "=== applying patch: ${PATCH_FILE_IN} ==="
  ./scripts/02_baseline.sh
  git apply "${PATCH_FILE_IN}"
  ./scripts/03_rebuild_after_patch.sh
  ./scripts/04_optimized.sh
  ./scripts/05_compare.sh
else
  echo "No patch supplied -> skipping benchmarks (would still need >= ${BENCH_TP:-8} GPUs)."
  echo "Rerun later with: bash inside container -> ./scripts/02_baseline.sh, etc."
fi
EOSH

# Translate patch path into container path if provided.
patch_mount=""
patch_env=""
if [[ -n "${PATCH_FILE}" ]]; then
  [[ -f "${PATCH_FILE}" ]] || die "patch file not found: ${PATCH_FILE}"
  patch_mount="-v $(realpath "${PATCH_FILE}"):/tmp/patch.diff:ro"
  patch_env="-e PATCH_FILE_IN=/tmp/patch.diff"
fi

log "Image           = ${IMAGE}"
log "Container cmd   = ${CONTAINER_CMD}"
log "Bind mounts     = ${HOST_VLLM} -> /work/vllm,  ${HOST_CACHE} -> /root/.cache"

if (( SHELL_ONLY )); then
  exec "${CONTAINER_CMD}" run --rm -it \
    ${GPU_FLAG} \
    -v "${HOST_VLLM}:/work/vllm" \
    -v "${HOST_CACHE}:/root/.cache" \
    ${patch_mount} ${patch_env} \
    -w /work/vllm \
    --shm-size=16g \
    "${IMAGE}" bash
fi

exec "${CONTAINER_CMD}" run --rm -i \
  ${GPU_FLAG} \
  -v "${HOST_VLLM}:/work/vllm" \
  -v "${HOST_CACHE}:/root/.cache" \
  ${patch_mount} ${patch_env} \
  -w /work/vllm \
  --shm-size=16g \
  "${IMAGE}" bash -c "${IN_CONTAINER}"
