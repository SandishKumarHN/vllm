#!/usr/bin/env bash
# Stress-run test_eplb_race_condition.py to expose the race that
# vllm/distributed/eplb/rebalance_execute.py:586 (torch.accelerator.synchronize)
# is meant to prevent.
#
# Usage:
#   ./run_eplb_race_stress.sh                    # 20 runs, default amplification
#   RUNS=50 ITER=1000 HIDDEN='[512,1024]' ./run_eplb_race_stress.sh

set -u

cd "$(dirname "$0")/../.."   # cd to vllm repo root

VLLM_DIR=$(pwd)
TEST_FILE=tests/distributed/test_eplb_race_condition.py
REBAL_FILE=vllm/distributed/eplb/rebalance_execute.py

RUNS=${RUNS:-20}
ITER=${ITER:-500}
HIDDEN=${HIDDEN:-'[512, 1024]'}

# 1. Verify the synchronize() removal is loaded by Python.
echo "=== [1/4] Verifying synchronize() edit is live ==="
python - <<'PY'
import inspect, re, vllm.distributed.eplb.rebalance_execute as m
call_pat = re.compile(r'\.synchronize\s*\(')
hits = []
for i, l in enumerate(inspect.getsource(m).splitlines(), 1):
    if call_pat.search(l):
        hits.append((i, "commented" if l.lstrip().startswith('#') else "ACTIVE", l.strip()))
print(f"Found {len(hits)} .synchronize(...) call sites in loaded module:")
for ln, status, txt in hits:
    print(f"  {ln}: [{status}] {txt}")
active = [h for h in hits if h[1] == "ACTIVE"]
if active:
    print("WARN: an active .synchronize() call is still present.")
else:
    print("OK: no active .synchronize() calls — edit is live.")
PY
echo

# 2. Patch the test in place; restore on exit.
BACKUP=$(mktemp -t test_eplb_race_condition.XXXXXX.py)
cp "$TEST_FILE" "$BACKUP"
trap 'cp "$BACKUP" "$TEST_FILE"; rm -f "$BACKUP"; echo; echo "[cleanup] restored $TEST_FILE"' EXIT INT TERM

echo "=== [2/4] Patching test: iterations=$ITER, hidden_sizes=$HIDDEN ==="
python - "$TEST_FILE" "$ITER" "$HIDDEN" <<'PY'
import re, sys, pathlib
path, iters, hidden = sys.argv[1], sys.argv[2], sys.argv[3]
src = pathlib.Path(path).read_text()
src = re.sub(r'(_stress_test_worker\([^)]*iterations=)\d+', r'\g<1>' + iters, src)
src = re.sub(r'hidden_sizes\s*=\s*\[[^\]]*\]', f'hidden_sizes = {hidden}', src)
pathlib.Path(path).write_text(src)
print("Patched lines:")
import subprocess
subprocess.run(["grep", "-nE", "iterations=|hidden_sizes", path])
PY
echo

# 3. Make sure no env var forces serial CUDA execution.
echo "=== [3/4] CUDA env hygiene ==="
unset CUDA_LAUNCH_BLOCKING
unset TORCH_USE_CUDA_DSA
export CUDA_LAUNCH_BLOCKING=0
echo "CUDA_LAUNCH_BLOCKING=$CUDA_LAUNCH_BLOCKING (must be 0/unset for races to surface)"
env | grep -E '^(CUDA_|NCCL_|TORCH_)' | sort || true
echo

# 4. Loop pytest RUNS times. Capture pass/fail and any non-PASS output.
echo "=== [4/4] Running pytest x$RUNS ==="
LOG_DIR=$(mktemp -d -t eplb_race_runs.XXXXXX)
echo "logs: $LOG_DIR"
PASS=0; FAIL=0; FAIL_RUNS=()
for i in $(seq 1 "$RUNS"); do
    LOG="$LOG_DIR/run_$i.log"
    if pytest -s -v "$TEST_FILE" >"$LOG" 2>&1; then
        PASS=$((PASS+1))
        printf "  run %3d: PASS\n" "$i"
    else
        FAIL=$((FAIL+1))
        FAIL_RUNS+=("$i")
        printf "  run %3d: FAIL  (log: %s)\n" "$i" "$LOG"
    fi
done

echo
echo "=== Summary ==="
echo "Total: $RUNS   Pass: $PASS   Fail: $FAIL"
if [ "$FAIL" -gt 0 ]; then
    echo "Failing runs: ${FAIL_RUNS[*]}"
    echo "Tail of first failing log:"
    tail -30 "$LOG_DIR/run_${FAIL_RUNS[0]}.log"
fi
