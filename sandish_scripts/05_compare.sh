#!/usr/bin/env bash
# 05_compare.sh - tabulate baseline.json vs optimized.json side-by-side.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/common.sh"
activate_venv

BASE="${1:-${RESULTS_DIR}/baseline.json}"
OPT="${2:-${RESULTS_DIR}/optimized.json}"

[[ -f "${BASE}" ]] || die "baseline json not found: ${BASE}"
[[ -f "${OPT}"  ]] || die "optimized json not found: ${OPT}"

python - "${BASE}" "${OPT}" <<'PY'
import json, sys
b = json.load(open(sys.argv[1]))
o = json.load(open(sys.argv[2]))

# vllm bench latency emits "avg_latency", "latencies", "percentiles": {"10":..,"25":..,"50":..,"75":..,"90":..,"99":..}
def pick(d):
    return {
        "avg_latency_s": d.get("avg_latency"),
        **{f"p{int(float(k))}_latency_s": v for k, v in (d.get("percentiles") or {}).items()},
    }

bb, oo = pick(b), pick(o)
keys = sorted(set(bb) | set(oo))
print(f"\n{'metric':<24}{'baseline':>14}{'optimized':>14}{'delta':>14}{'speedup x':>12}")
print('-' * 78)
for k in keys:
    bv = bb.get(k); ov = oo.get(k)
    if bv is None or ov is None:
        print(f"{k:<24}{str(bv):>14}{str(ov):>14}")
        continue
    d  = ov - bv
    sx = bv / ov if ov else float('inf')
    print(f"{k:<24}{bv:>14.4f}{ov:>14.4f}{d:>+14.4f}{sx:>12.3f}")
print()
PY
