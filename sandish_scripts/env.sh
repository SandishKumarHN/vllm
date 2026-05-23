#!/usr/bin/env bash
# Source this BEFORE running `vllm ...` or `python -c "import vllm"` directly.
# Sets the CUDA toolkit + LD_PRELOAD for cublas/cudart/etc. so:
#   1. flashinfer's runtime JIT compile finds cuda_runtime.h via our stitched
#      CUDA_HOME (the system /usr/local/cuda is a stub on Meta devvms)
#   2. unversioned cublas symbols in vllm/_C.abi3.so resolve at import time
#      (we built against pip cu13 toolkit which lacks GNU symbol versioning,
#      so torch's RTLD_LOCAL libcublas load leaves them unresolved unless
#      LD_PRELOAD promotes the same libs to global scope)
#
# Usage:
#   source sandish_scripts/env.sh
#   vllm bench latency ...

# --- 1. activate venv ---
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  # shellcheck disable=SC1091
  source /home/sandish/vllm/.venv/bin/activate
fi

# --- 2. point CUDA at the stitched toolkit ---
export CUDA_HOME=/home/sandish/vllm/.deps/cuda_home
export PATH="${CUDA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export CUDA_NVCC_EXECUTABLE="${CUDA_HOME}/bin/nvcc"
export CUDA_BIN_PATH="${CUDA_HOME}/bin"
export CUDA_TOOLKIT_ROOT_DIR="${CUDA_HOME}"

# --- 3. LD_PRELOAD cublas/cudart/nvrtc/cudnn for unversioned-symbol resolution ---
# Equivalent to ctypes.CDLL(..., RTLD_GLOBAL) but happens before Python starts.
# Replaces what we previously did via a patch to vllm/env_override.py.
_PRELOAD=""
_CU13_LIB=/home/sandish/vllm/.venv/lib/python3.12/site-packages/nvidia/cu13/lib
for lib in libcudart.so.13 libcublas.so.13 libcublasLt.so.13 \
           libnvrtc.so.13 libnvjitlink.so.13; do
  if [[ -f "${_CU13_LIB}/${lib}" ]]; then
    [[ -z "${_PRELOAD}" ]] && _PRELOAD="${_CU13_LIB}/${lib}" \
                          || _PRELOAD="${_PRELOAD}:${_CU13_LIB}/${lib}"
  fi
done
# cudnn lives in a separate pip package
_CUDNN=/home/sandish/vllm/.venv/lib/python3.12/site-packages/nvidia/cudnn/lib/libcudnn.so.9
if [[ -f "${_CUDNN}" ]]; then
  [[ -z "${_PRELOAD}" ]] && _PRELOAD="${_CUDNN}" \
                        || _PRELOAD="${_PRELOAD}:${_CUDNN}"
fi
export LD_PRELOAD="${_PRELOAD}${LD_PRELOAD:+:${LD_PRELOAD}}"
unset _PRELOAD _CU13_LIB _CUDNN

echo "env set:"
echo "  VIRTUAL_ENV  = ${VIRTUAL_ENV}"
echo "  CUDA_HOME    = ${CUDA_HOME}"
echo "  nvcc         = $(command -v nvcc)"
echo "  GPUs         = $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd, -)"
echo "  LD_PRELOAD   = $(awk -F: '{print NF}' <<<"${LD_PRELOAD}") libs preloaded"
