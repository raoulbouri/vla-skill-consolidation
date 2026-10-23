#!/usr/bin/env bash
# Checkpoint-baseline evaluation of the official pi05_libero on the frozen 10-task manifest (one command).
# Starts a FRESH official OpenPI policy server (so its sampling RNG sequence is reproducible), runs scripts/run_eval.py
# in the simulator env, then stops the server. Extra args go to run_eval.py, e.g.
#   bash scripts/baseline_eval.sh --episodes-per-task 10
#   bash scripts/baseline_eval.sh --from-run outputs/baseline/<run>      # rerun from saved config + manifest
# Long runs: setsid nohup bash scripts/baseline_eval.sh ... > logs/baseline_<stamp>.log 2>&1 < /dev/null &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
"$ROOT/scripts/gpu_guard.sh"
PORT=$("$CLVLA_SIM_PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
mkdir -p "$ROOT/logs"
SLOG="$ROOT/logs/policy_server_$(date -u +%Y%m%dT%H%M%SZ)_$PORT.log"
setsid "$ROOT/scripts/serve_policy.sh" "$PORT" > "$SLOG" 2>&1 < /dev/null &
SPID=$!
cleanup() { kill -TERM -- "-$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }
trap cleanup EXIT
echo "[baseline_eval] policy server pid $SPID port $PORT log $SLOG"
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && break
  kill -0 "$SPID" 2>/dev/null || { echo "policy server died:"; tail -30 "$SLOG"; exit 1; }
  sleep 3
done
curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null || { echo "policy server not healthy"; exit 1; }
export CLVLA_SERVER_INFO
CLVLA_SERVER_INFO=$(printf '{"command": "scripts/serve_policy.sh %s", "log": "%s", "openpi_commit": "%s", "config": "pi05_libero", "checkpoint_dir": "%s"}' \
  "$PORT" "$SLOG" "$(git -C "$OPENPI_DIR" rev-parse HEAD)" "$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero")
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$ROOT/scripts/py-sim" "$ROOT/scripts/run_eval.py" --host 127.0.0.1 --port "$PORT" "$@"
