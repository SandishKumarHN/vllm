# vLLM build handoff (2026-05-22)

## Status: BUILD SUCCEEDED, BENCHMARK DEFERRED

vllm `0.20.2rc1.dev147+gd2f22dfc9.d20260522` built from source in 37m34s.
All extensions load. Verified with:

```bash
source /home/sandish/vllm/.venv/bin/activate
python -c "import vllm._C, vllm._moe_C, vllm.cumem_allocator, vllm.vllm_flash_attn._vllm_fa2_C, vllm; print(vllm.__version__)"
```

## What I did, in summary

### CUDA toolkit stitching (devvm has compiler-only stub at /usr/local/cuda)
- Built `/home/sandish/vllm/.deps/cuda_home/` — merged tree of:
  - `bin/`, `nvvm/` (real dirs containing symlinks): nvcc 13.0.88 from /usr/local/cuda-13.0/
  - `targets/x86_64-linux/include/`: 151 headers from pip `cuda-toolkit==13.0.2`  + 3 from system + symlink to `cuda/cccl/headers/include` (cub, thrust, etc.)
  - `targets/x86_64-linux/lib/`: cudart/cublas/cudnn/... from pip cu13/lib + unversioned `.so` aliases + `stubs/libcuda.so` -> /usr/lib64/libcuda.so

### Source mirrors (github blocked for agent — user cloned)
- `VLLM_CUTLASS_SRC_DIR=/data/repos/fbsource/third-party/cutlass/4.4.2`
- `TRITON_KERNELS_SRC_DIR=/data/repos/fbsource/third-party/triton-lang-kernels`
- 4 repos manually cloned by user to `/home/sandish/vllm/.deps/sources/`: deepgemm, flashmla, qutlass, vllm-flash-attn
- 3 missing cutlass submodules (flashmla, qutlass, vllm-flash-attn) symlinked to fbsource cutlass 4.4.2

### Source-tree patches (3 files; revert when moving to real hardware)
1. `/home/sandish/vllm/.deps/sources/vllm-flash-attn/CMakeLists.txt`:
   - L8: `set(FA3_ENABLED OFF)` (was `ON`)
   - After endif of FA3 block: added `if (NOT TARGET _vllm_fa3_C) add_custom_target(_vllm_fa3_C) endif()`
   - Reason: FA3 hopper kernels fail to compile against fbsource cutlass 4.4.2 (API drift vs vllm-flash-attn's pinned cutlass commit). FA3 unused on A100 anyway.
2. `/home/sandish/vllm/setup.py` line ~1005:
   - `CMakeExtension(name="vllm.vllm_flash_attn._vllm_fa3_C", optional=True)` (added `optional=True`)
   - Reason: FA3 produces no .so when disabled; setup.py's install step would fail without `optional`.
3. `/home/sandish/vllm/vllm/env_override.py` end:
   - Appended `_preload_cublas_global()` that RTLD_GLOBAL-preloads libcudart/cublas/cublasLt/nvrtc/nvjitlink/cudnn.
   - Reason: pip cu13 toolkit ships libs without GNU symbol versioning, so unversioned `cublasGemmEx` (etc) in vllm/_C.abi3.so can't resolve from torch's RTLD_LOCAL-loaded libcublas. Stock vllm wheels (against system CUDA with versioned libs) don't need this.

## When you move to real hardware (8xH100)

1. The host MUST have a real CUDA toolkit (headers + libs at $CUDA_HOME). NOT a Meta devvm.
2. Revert FA3_ENABLED -> ON in vllm-flash-attn (H100 supports it).
3. Drop the `optional=True` from setup.py FA3 line (or keep — harmless).
4. The env_override.py preload is safe to leave (harmless on real CUDA installs since libcublas will already be in DT_NEEDED via standard build).
5. Set TORCH_CUDA_ARCH_LIST="9.0+PTX" (H100) instead of "8.0" (A100).
6. Re-run `scripts/01_build_vllm.sh` with the same env vars I used.

## Original optimization patch — STILL MISSING

Your Day 1 message said "Apply the proposed patch (see below)" but no patch was attached.
Before the optimized benchmark run, please paste the patch or point me at a file.

## Hardware reality reminder

- This devvm: 2x A100 80GB (160 GB total)
- DeepSeek-V3: 671B params, ~700 GB FP8 — does NOT fit
- Original command: `--tensor-parallel-size 8` — requires 8 GPUs
- So both build AND benchmark of the original spec require moving to 8xH100/H200.

## Quick build-resume / re-run snippet (if you ever need to rebuild)

```bash
cd /home/sandish/vllm
nohup env \
  CUDA_HOME=/home/sandish/vllm/.deps/cuda_home \
  VLLM_CUTLASS_SRC_DIR=/data/repos/fbsource/third-party/cutlass/4.4.2 \
  TRITON_KERNELS_SRC_DIR=/data/repos/fbsource/third-party/triton-lang-kernels \
  DEEPGEMM_SRC_DIR=/home/sandish/vllm/.deps/sources/deepgemm \
  FLASH_MLA_SRC_DIR=/home/sandish/vllm/.deps/sources/flashmla \
  QUTLASS_SRC_DIR=/home/sandish/vllm/.deps/sources/qutlass \
  VLLM_FLASH_ATTN_SRC_DIR=/home/sandish/vllm/.deps/sources/vllm-flash-attn \
  TORCH_CUDA_ARCH_LIST="8.0" TORCH_BACKEND=cu130 \
  MAX_JOBS=24 NVCC_THREADS=2 \
  bash scripts/01_build_vllm.sh > .build_unattended.log 2>&1 &
echo $! > .build_unattended.pid
disown
# Watch with: ./scripts/99_build_status.sh
```
