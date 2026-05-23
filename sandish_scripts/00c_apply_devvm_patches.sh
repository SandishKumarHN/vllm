#!/usr/bin/env bash
# 00c_apply_devvm_patches.sh — apply (or revert) the 2 source-tree patches
# needed to build vLLM on a sm_80-only host with the pip cu13 toolkit.
#
# Patches (all stored under sandish_scripts/patches/):
#   1. setup_py_fa3_optional.patch       — mark FA3 ext as optional=True so
#                                          setup.py wheel-packaging doesn't
#                                          fail when FA3 isn't built
#   2. vllm_flash_attn_fa3_off.patch     — disable FA3 in vllm-flash-attn (its
#                                          Hopper kernels won't compile against
#                                          fbsource cutlass 4.4.2 anyway, and
#                                          they're useless on A100)
#
# NOTE: there is NO env_override.py patch anymore. The cublas RTLD_GLOBAL
# preload is now done via LD_PRELOAD in sandish_scripts/env.sh.
#
# DO NOT RUN on a real H100/H200 host with proper system CUDA — FA3 works
# there and you want the kernels enabled.
#
# Usage:
#   ./sandish_scripts/00c_apply_devvm_patches.sh apply
#   ./sandish_scripts/00c_apply_devvm_patches.sh revert
#
# Workflow when pulling upstream:
#   1. ./sandish_scripts/00c_apply_devvm_patches.sh revert
#   2. git pull upstream main && git merge
#   3. ./sandish_scripts/00c_apply_devvm_patches.sh apply
#   4. (rebuild if upstream changed any C++/CUDA files)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PATCH_DIR="${SCRIPT_DIR}/patches"

ACTION="${1:-}"
[[ "${ACTION}" == "apply" || "${ACTION}" == "revert" ]] \
  || { echo "usage: $0 {apply|revert}" >&2; exit 1; }

# Each entry: "<patch file>:<target dir>"
PATCHES=(
  "setup_py_fa3_optional.patch:${VLLM_ROOT}"
  "vllm_flash_attn_fa3_off.patch:${VLLM_ROOT}/.deps/sources/vllm-flash-attn"
)

apply_one() {
  local patch="$1" target_dir="$2" patch_path="${PATCH_DIR}/$1"
  [[ -f "${patch_path}" ]] || { echo "missing patch: ${patch_path}" >&2; return 1; }
  [[ -d "${target_dir}" ]] || { echo "skip ${patch}: ${target_dir} does not exist"; return 0; }

  # Check if already applied: `patch --dry-run -R` succeeds iff already applied
  if patch --dry-run -p1 -R -d "${target_dir}" -i "${patch_path}" >/dev/null 2>&1; then
    echo "[apply ] ${patch}: already applied (skip)"
    return 0
  fi
  if patch --dry-run -p1 -d "${target_dir}" -i "${patch_path}" >/dev/null 2>&1; then
    patch -p1 -d "${target_dir}" -i "${patch_path}"
    echo "[apply ] ${patch}: OK"
  else
    echo "[apply ] ${patch}: FAILED (does not apply cleanly — upstream changed?)" >&2
    return 1
  fi
}

revert_one() {
  local patch="$1" target_dir="$2" patch_path="${PATCH_DIR}/$1"
  [[ -f "${patch_path}" ]] || { echo "missing patch: ${patch_path}" >&2; return 1; }
  [[ -d "${target_dir}" ]] || { echo "skip ${patch}: ${target_dir} does not exist"; return 0; }

  if patch --dry-run -p1 -d "${target_dir}" -i "${patch_path}" >/dev/null 2>&1; then
    echo "[revert] ${patch}: not applied (skip)"
    return 0
  fi
  if patch --dry-run -p1 -R -d "${target_dir}" -i "${patch_path}" >/dev/null 2>&1; then
    patch -p1 -R -d "${target_dir}" -i "${patch_path}"
    echo "[revert] ${patch}: OK"
  else
    echo "[revert] ${patch}: FAILED (won't reverse cleanly)" >&2
    return 1
  fi
}

for entry in "${PATCHES[@]}"; do
  IFS=":" read -r p t <<< "${entry}"
  case "${ACTION}" in
    apply)  apply_one  "${p}" "${t}" ;;
    revert) revert_one "${p}" "${t}" ;;
  esac
done

echo ""
echo "OK — patches ${ACTION}ed."
[[ "${ACTION}" == "apply"  ]] && echo "Don't forget: \`source sandish_scripts/env.sh\` before \`vllm ...\` or \`import vllm\` (handles cublas LD_PRELOAD)."
[[ "${ACTION}" == "revert" ]] && echo "Source tree is now clean — safe to \`git pull\` and merge."
