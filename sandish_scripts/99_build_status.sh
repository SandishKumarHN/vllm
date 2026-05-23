#!/usr/bin/env bash
# Check unattended-build status. Safe to run anytime.
PIDFILE=/home/sandish/vllm/.build_unattended.pid
LOGFILE=/home/sandish/vllm/.build_unattended.log
echo "=== state ==="
if [[ -f "$PIDFILE" ]]; then
  PID=$(cat "$PIDFILE")
  if kill -0 "$PID" 2>/dev/null; then
    echo "running, PID=$PID"
    echo "elapsed: $(ps -o etime= -p $PID | tr -d ' ')"
  else
    echo "process not running (PID was $PID)"
  fi
else
  echo "no pid file (build never started or was cleaned up)"
fi
echo "---"
echo "=== latest build log ==="
LATEST=$(ls -t /home/sandish/vllm/build_logs/01_build_*.log 2>/dev/null | head -1)
[[ -n "$LATEST" ]] && echo "log: $LATEST  lines=$(wc -l < $LATEST)" && tail -5 "$LATEST"
echo "---"
echo "=== vllm importable? ==="
source /home/sandish/vllm/.venv/bin/activate 2>/dev/null
python -c "import vllm; print('vllm', vllm.__version__, vllm.__file__)" 2>&1 | head -3
