# sandish_scripts — local vLLM build + benchmark helpers

All non-upstream tooling lives here so you can `git pull upstream main` cleanly.
The upstream tree is left untouched except for 2 small patches under
`patches/`, which are applied/reverted on demand via `00c_apply_devvm_patches.sh`.

## File map

| File                            | Purpose                                                            |
|---------------------------------|--------------------------------------------------------------------|
| `common.sh`                     | Shared env, paths, helpers. Source-only.                           |
| `env.sh`                        | Source before `vllm` / `python -c "import vllm"`. Sets CUDA_HOME + LD_PRELOAD cublas/cudart/nvrtc/cudnn. |
| `00_setup_env.sh`               | Install `uv`, create venv, install build deps.                     |
| `00a_stitch_cuda_toolkit.sh`    | **Devvm-only.** Stitches a CMake-/nvcc-compatible CUDA_HOME at `.deps/cuda_home` from system nvcc + pip cu13 toolkit. |
| `00b_fetch_external_sources.sh` | **Github-blocked-only.** Clones the 4 deps vLLM's CMake fetches + populates cutlass submodules. |
| `00c_apply_devvm_patches.sh`    | Apply / revert the 2 source-tree patches (`patches/*.patch`). Idempotent. **Run `revert` before `git pull`.** |
| `01_build_vllm.sh`              | Full from-source build via `uv pip install -e .`                   |
| `02_baseline.sh`                | Run baseline `vllm bench latency`, write `baseline.json`.          |
| `03_rebuild_after_patch.sh`     | Incremental CMake/Ninja rebuild after a CUDA/C++ patch.            |
| `04_optimized.sh`               | Run post-patch benchmark, write `optimized.json`.                  |
| `05_compare.sh`                 | Tabulate baseline vs optimized.                                    |
| `99_build_status.sh`            | Inspect unattended-build state.                                    |
| `99_docker_build_and_bench.sh`  | Alternative docker path.                                           |
| `run_all.sh`                    | Orchestrator chaining the standard scripts.                        |
| `BUILD_HANDOFF.md`              | Full history of what we did and why.                               |
| `patches/`                      | The 2 source-tree patches applied by `00c`.                        |

## Patches in `patches/`

| File                                  | Touches                                              | Why                                                                                                       |
|---------------------------------------|------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|
| `setup_py_fa3_optional.patch`         | `setup.py` (upstream-tracked, ~2 lines)              | Marks the `_vllm_fa3_C` ext as `optional=True` so packaging doesn't fail when FA3 is disabled.            |
| `vllm_flash_attn_fa3_off.patch`       | `.deps/sources/vllm-flash-attn/CMakeLists.txt` (gitignored, no merge concern) | Disables FA3 Hopper kernels (they fail against fbsource cutlass 4.4.2 and don't run on A100 anyway).      |

There is no patch on `vllm/env_override.py` anymore — the cublas RTLD_GLOBAL
preload is now done via `LD_PRELOAD` in `env.sh`, leaving env_override.py
upstream-clean.

## Workflow when pulling upstream

```bash
cd /home/sandish/vllm

# 1. revert patches so the working tree matches upstream
./sandish_scripts/00c_apply_devvm_patches.sh revert

# 2. merge upstream
git pull upstream main

# 3. re-apply patches
./sandish_scripts/00c_apply_devvm_patches.sh apply

# 4. (if upstream changed C++/CUDA files) rebuild
source sandish_scripts/env.sh
./sandish_scripts/03_rebuild_after_patch.sh

# 5. verify
python -c "import vllm, vllm._C; print(vllm.__version__)"
```

If step 3 reports `FAILED (does not apply cleanly — upstream changed?)`,
the patch needs to be regenerated against the new upstream state:

```bash
# manually re-apply the change, then:
cd /home/sandish/vllm && git diff setup.py > sandish_scripts/patches/setup_py_fa3_optional.patch
cd /home/sandish/vllm/.deps/sources/vllm-flash-attn && git diff CMakeLists.txt > /home/sandish/vllm/sandish_scripts/patches/vllm_flash_attn_fa3_off.patch
```

## Recipes

### A. Daily use (build is already done)

```bash
cd /home/sandish/vllm
source sandish_scripts/env.sh    # one shot per shell session
vllm bench latency --model deepseek-ai/DeepSeek-V2-Lite --tensor-parallel-size 2 ...
```

### B. Devvm fresh build (the path we walked)

```bash
cd /home/sandish/vllm
./sandish_scripts/00_setup_env.sh
source .venv/bin/activate
uv pip install cuda-toolkit cuda-cccl   # if not already
./sandish_scripts/00a_stitch_cuda_toolkit.sh
./sandish_scripts/00b_fetch_external_sources.sh   # only if github is blocked
# export the *_SRC_DIR env vars it prints
./sandish_scripts/00c_apply_devvm_patches.sh apply
source sandish_scripts/env.sh
TORCH_CUDA_ARCH_LIST="8.0" TORCH_BACKEND=cu130 \
MAX_JOBS=24 NVCC_THREADS=2 \
  ./sandish_scripts/01_build_vllm.sh
```

### C. Patch-then-rebench cycle (current 2× A100 host)

```bash
source sandish_scripts/env.sh

# 1. baseline (model swapped to fit 2×A100; original DeepSeek-V3 + TP=8 can't run here)
BENCH_MODEL=deepseek-ai/DeepSeek-V2-Lite BENCH_TP=2 ./sandish_scripts/02_baseline.sh

# 2. edit vllm source files (your optimization)
# ...

# 3. incremental rebuild
./sandish_scripts/03_rebuild_after_patch.sh

# 4. optimized run
BENCH_MODEL=deepseek-ai/DeepSeek-V2-Lite BENCH_TP=2 ./sandish_scripts/04_optimized.sh

# 5. compare
./sandish_scripts/05_compare.sh
```

## Heads-up: other in-tree changes NOT made by me

The prior agent session in this directory also modified these upstream-tracked
files (looks like cu130 backend support, requirements pruning):

```
modified:   pyproject.toml
modified:   requirements/build/cpu.txt
modified:   requirements/build/cuda.txt
modified:   requirements/build/rocm.txt
modified:   requirements/cpu.txt
modified:   requirements/cuda.txt
modified:   requirements/test/cuda.in
modified:   requirements/test/cuda.txt
modified:   requirements/test/rocm.in
modified:   requirements/test/rocm.txt
modified:   requirements/test/xpu.txt
modified:   requirements/xpu.txt
modified:   .gitignore
```

These WILL also conflict on `git pull upstream`. They aren't my changes so I
haven't moved them, but you'll want to either:
- stash them as patches the same way I did with my 2 (see workflow above), or
- accept upstream's version of those files and re-tune as needed.

To inspect what's changed: `git diff pyproject.toml requirements/ .gitignore`.

## Tuning env vars

See `common.sh` for defaults. Override via env, e.g.
`BENCH_MODEL=... BENCH_TP=2 ./sandish_scripts/02_baseline.sh`.
