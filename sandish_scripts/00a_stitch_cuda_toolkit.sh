#!/usr/bin/env bash
# 00a_stitch_cuda_toolkit.sh — build a CMake-/nvcc-compatible CUDA_HOME tree
# from pieces available on this host. Use ONLY when /usr/local/cuda is a
# compiler-only stub (e.g. Meta devvm*.pnb0). Skip on hosts with a real
# CUDA toolkit installed (apt/dnf cuda-toolkit, or properly-provisioned
# DGX/cloud images).
#
# Output:
#   $VLLM_ROOT/.deps/cuda_home/  (merged tree with bin/, nvvm/, include/,
#                                 lib64/, targets/x86_64-linux/{include,lib})
#
# Detection of "incomplete" toolkit = missing $CUDA_HOME/include/cuda_runtime.h
# or $CUDA_HOME/lib64/libcudart.so.
#
# Sources combined:
#   - nvcc + nvvm from /usr/local/cuda-13.0 (the compiler-only system install)
#   - headers + libs from pip-installed `cuda-toolkit==13.0.2`
#     (lives at .venv/lib/python3.12/site-packages/nvidia/cu13/)
#   - CCCL headers (cub/thrust) from pip `cuda-cccl`
#     (lives at .venv/lib/python3.12/site-packages/cuda/cccl/headers/include)
#   - libcuda.so driver stub from /usr/lib64/libcuda.so
#
# IMPORTANT: /home/sandish/vllm/.deps/cuda_home/bin/ and nvvm/ must be REAL
# directories (not symlinks), so the kernel's `bin/..` resolution physically
# points back to our merged root rather than dereferencing the symlink target.
# This was the prior agent's "_HERE_/TOP" stumbling block.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

CUDA_HOME_OUT="${VLLM_ROOT}/.deps/cuda_home"
SYS_CUDA="${SYS_CUDA:-/usr/local/cuda-13.0}"
PIP_CU13="${VENV_DIR}/lib/python3.12/site-packages/nvidia/cu13"
PIP_CCCL="${VENV_DIR}/lib/python3.12/site-packages/cuda/cccl/headers/include"
DRIVER_LIB="${DRIVER_LIB:-/usr/lib64/libcuda.so}"

log "Stitching CUDA_HOME at ${CUDA_HOME_OUT}"
log "  system nvcc:  ${SYS_CUDA}"
log "  pip cu13:     ${PIP_CU13}"
log "  pip cccl:     ${PIP_CCCL}"
log "  driver stub:  ${DRIVER_LIB}"

# --- preflight ---
[[ -x "${SYS_CUDA}/bin/nvcc" ]] || die "no nvcc at ${SYS_CUDA}/bin/nvcc — install cuda-13.0 system package or set SYS_CUDA"
[[ -d "${PIP_CU13}/include" && -d "${PIP_CU13}/lib" ]] || die "missing pip cu13 toolkit — run: uv pip install cuda-toolkit"
[[ -d "${PIP_CCCL}/cub" ]] || die "missing pip cccl headers — run: uv pip install cuda-cccl"
[[ -e "${DRIVER_LIB}" ]] || warn "no driver libcuda.so at ${DRIVER_LIB} — link step may fail"

# --- build tree ---
rm -rf "${CUDA_HOME_OUT}"
mkdir -p "${CUDA_HOME_OUT}/targets/x86_64-linux/include" "${CUDA_HOME_OUT}/targets/x86_64-linux/lib/stubs"

# bin/ as a REAL dir of symlinks (so bin/.. resolves to CUDA_HOME_OUT, not SYS_CUDA)
mkdir -p "${CUDA_HOME_OUT}/bin"
for f in "${SYS_CUDA}/bin"/*; do
  ln -s "$f" "${CUDA_HOME_OUT}/bin/$(basename "$f")"
done

# nvvm/ same trick
mkdir -p "${CUDA_HOME_OUT}/nvvm"
for f in "${SYS_CUDA}/nvvm"/*; do
  ln -s "$f" "${CUDA_HOME_OUT}/nvvm/$(basename "$f")"
done

# Other top-level dirs can be plain symlinks (nvcc doesn't `..`-traverse them)
for d in extras share compute-sanitizer; do
  [[ -e "${SYS_CUDA}/${d}" ]] && ln -s "${SYS_CUDA}/${d}" "${CUDA_HOME_OUT}/${d}"
done

# targets/x86_64-linux/include/ : merge pip cu13 headers + system targets headers
TARG_INC="${CUDA_HOME_OUT}/targets/x86_64-linux/include"
for f in "${PIP_CU13}/include"/*; do
  ln -s "$f" "${TARG_INC}/$(basename "$f")"
done
for f in "${SYS_CUDA}/targets/x86_64-linux/include"/*; do
  name="$(basename "$f")"
  [[ ! -e "${TARG_INC}/${name}" ]] && ln -s "$f" "${TARG_INC}/${name}"
done

# cccl/ subdir (CMake's CUDA::cudart interface requires it)
mkdir -p "${TARG_INC}/cccl"
for f in "${PIP_CCCL}"/{cub,thrust,cuda,nv}; do
  [[ -e "$f" ]] && ln -s "$f" "${TARG_INC}/cccl/$(basename "$f")"
done

# targets/x86_64-linux/lib/ : merge pip cu13 libs + system targets libs
TARG_LIB="${CUDA_HOME_OUT}/targets/x86_64-linux/lib"
for f in "${PIP_CU13}/lib"/*; do
  ln -s "$f" "${TARG_LIB}/$(basename "$f")"
done
for f in "${SYS_CUDA}/targets/x86_64-linux/lib"/*; do
  name="$(basename "$f")"
  [[ ! -e "${TARG_LIB}/${name}" ]] && ln -s "$f" "${TARG_LIB}/${name}"
done

# Unversioned .so aliases CMake FindCUDAToolkit needs
for ver in libnvrtc.so.13 libcudart.so.13 libnvjitlink.so.13; do
  base="${ver%.so.*}.so"
  [[ -e "${TARG_LIB}/${ver}" && ! -e "${TARG_LIB}/${base}" ]] && ln -s "${ver}" "${TARG_LIB}/${base}"
done

# Driver stub for link-time libcuda resolution
[[ -e "${DRIVER_LIB}" ]] && ln -s "${DRIVER_LIB}" "${TARG_LIB}/stubs/libcuda.so"
[[ ! -e "${TARG_LIB}/libcuda.so" ]] && ln -s "stubs/libcuda.so" "${TARG_LIB}/libcuda.so"

# Top-level convenience symlinks mirroring stock CUDA layout
ln -s targets/x86_64-linux/include "${CUDA_HOME_OUT}/include"
ln -s targets/x86_64-linux/lib     "${CUDA_HOME_OUT}/lib"
ln -s targets/x86_64-linux/lib     "${CUDA_HOME_OUT}/lib64"

# --- sanity: tiny nvcc + cmake test ---
log "Smoke-testing the stitched toolkit ..."
T="$(mktemp -d)"
trap 'rm -rf "$T"' EXIT
cat > "$T/t.cu" <<'EOF'
#include <cuda_runtime.h>
__global__ void k(int* o){ *o = 42; }
int main(){ int* d; cudaMalloc(&d, 4); k<<<1,1>>>(d); int h=0; cudaMemcpy(&h,d,4,cudaMemcpyDeviceToHost); cudaFree(d); return h==42?0:1; }
EOF
"${CUDA_HOME_OUT}/bin/nvcc" -arch=sm_80 -I "${CUDA_HOME_OUT}/include" -L "${CUDA_HOME_OUT}/lib64" -o "$T/t" "$T/t.cu" \
  || die "stitched nvcc compile failed"
LD_LIBRARY_PATH="${CUDA_HOME_OUT}/lib64:${LD_LIBRARY_PATH:-}" "$T/t" \
  || warn "stitched toolkit compiled OK but kernel exec failed (driver/runtime mismatch?)"

log "OK — CUDA_HOME stitched at ${CUDA_HOME_OUT}"
log "Use it via: export CUDA_HOME=${CUDA_HOME_OUT}"
