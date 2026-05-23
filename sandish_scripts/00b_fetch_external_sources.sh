#!/usr/bin/env bash
# 00b_fetch_external_sources.sh — populate $VLLM_ROOT/.deps/sources/ with the
# 4 github-only deps vLLM's CMake fetches at configure time.
#
# Why this script exists: on hosts where github.com is blocked (e.g. some AI
# agent identities at Meta), vLLM's FetchContent calls fail. This script
# performs the equivalent clones from a user shell that DOES have git access.
#
# On hosts with working github access, you can SKIP this script entirely —
# vLLM's CMake will fetch the deps automatically.
#
# Also handles:
#   - --recurse-submodules for deepgemm (needs third-party/cutlass, fmt)
#   - submodule init for vllm-flash-attn (needs csrc/cutlass)
#   - On Meta devvms, symlinks the 3 cutlass submodules to fbsource cutlass
#     instead of cloning (since github is blocked).
#
# After this script: set the *_SRC_DIR env vars (printed at the end) and run
# scripts/01_build_vllm.sh.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"

SRC_DIR="${VLLM_ROOT}/.deps/sources"
FBSOURCE_CUTLASS="${FBSOURCE_CUTLASS:-/data/repos/fbsource/third-party/cutlass/4.4.2}"
FBSOURCE_TRITON_KERNELS="${FBSOURCE_TRITON_KERNELS:-/data/repos/fbsource/third-party/triton-lang-kernels}"

# pinned revisions (must match the GIT_TAGs in vLLM's cmake/external_projects/)
DEEPGEMM_REV="891d57b4db1071624b5c8fa0d1e51cb317fa709f"
FLASHMLA_REV="a6ec2ba7bd0a7dff98b3f4d3e6b52b159c48d78b"
QUTLASS_REV="830d2c4537c7396e14a02a46fbddd18b5d107c65"
VLLM_FA_REV="f5bc33cfc02c744d24a2e9d50e6db656de40611c"

mkdir -p "${SRC_DIR}"
cd "${SRC_DIR}"

clone_at_rev() {
  local dir="$1" url="$2" rev="$3" extra_args="${4:-}"
  if [[ -d "${SRC_DIR}/${dir}/.git" ]]; then
    log "${dir}: already cloned, fetching ${rev}"
    cd "${SRC_DIR}/${dir}"
    git fetch --depth 1 origin "${rev}" 2>/dev/null || git fetch origin
    git checkout "${rev}"
    cd "${SRC_DIR}"
  else
    log "${dir}: cloning from ${url}"
    # shellcheck disable=SC2086
    git clone ${extra_args} "${url}" "${dir}"
    cd "${SRC_DIR}/${dir}"
    git checkout "${rev}"
    cd "${SRC_DIR}"
  fi
}

# --- 1. deepgemm (needs cutlass + fmt submodules) ---
clone_at_rev deepgemm https://github.com/deepseek-ai/DeepGEMM.git "${DEEPGEMM_REV}"
( cd "${SRC_DIR}/deepgemm" && git submodule update --init --recursive --depth 1 -- third-party/cutlass third-party/fmt )

# --- 2. flashmla (cutlass submodule) ---
clone_at_rev flashmla https://github.com/vllm-project/FlashMLA "${FLASHMLA_REV}"

# --- 3. qutlass (cutlass submodule) ---
clone_at_rev qutlass https://github.com/IST-DASLab/qutlass.git "${QUTLASS_REV}"

# --- 4. vllm-flash-attn (cutlass submodule) ---
clone_at_rev vllm-flash-attn https://github.com/vllm-project/flash-attention.git "${VLLM_FA_REV}"

# --- Populate cutlass submodules ---
# Try real submodule init first; on hosts where github is blocked, fall back
# to symlinking from fbsource cutlass (Meta-specific path).
populate_cutlass_submodule() {
  local repo="$1" sub_path="$2"
  local full="${SRC_DIR}/${repo}/${sub_path}"
  if [[ -e "${full}/include/cutlass/numeric_types.h" ]]; then
    log "${repo}/${sub_path}: already populated"
    return 0
  fi
  log "${repo}/${sub_path}: populating"
  if ( cd "${SRC_DIR}/${repo}" && git submodule update --init --depth 1 -- "${sub_path}" 2>/dev/null ); then
    log "  -> populated via git submodule"
  elif [[ -d "${FBSOURCE_CUTLASS}" ]]; then
    log "  -> github blocked, symlinking from fbsource cutlass ${FBSOURCE_CUTLASS}"
    rmdir "${full}" 2>/dev/null || rm -f "${full}"
    ln -s "${FBSOURCE_CUTLASS}" "${full}"
  else
    die "${repo}/${sub_path}: can't populate (no github, no fbsource cutlass)"
  fi
}
populate_cutlass_submodule flashmla        csrc/cutlass
populate_cutlass_submodule qutlass         third_party/cutlass
populate_cutlass_submodule vllm-flash-attn csrc/cutlass

# --- Verify ---
log "Verifying SOURCE_DIRs ..."
for d in deepgemm flashmla qutlass vllm-flash-attn; do
  [[ -d "${SRC_DIR}/${d}" ]] || die "${d}: missing"
  log "  ${d}: $(git -C "${SRC_DIR}/${d}" rev-parse HEAD)"
done

# --- Print env vars to set ---
cat <<EOF

OK — external sources ready. Set these env vars before scripts/01_build_vllm.sh:

  export DEEPGEMM_SRC_DIR=${SRC_DIR}/deepgemm
  export FLASH_MLA_SRC_DIR=${SRC_DIR}/flashmla
  export QUTLASS_SRC_DIR=${SRC_DIR}/qutlass
  export VLLM_FLASH_ATTN_SRC_DIR=${SRC_DIR}/vllm-flash-attn
EOF

# fbsource shortcuts (Meta-only) — emit only if those dirs exist
if [[ -d "${FBSOURCE_CUTLASS}" ]]; then
  echo "  export VLLM_CUTLASS_SRC_DIR=${FBSOURCE_CUTLASS}"
fi
if [[ -d "${FBSOURCE_TRITON_KERNELS}" ]]; then
  echo "  export TRITON_KERNELS_SRC_DIR=${FBSOURCE_TRITON_KERNELS}"
fi
