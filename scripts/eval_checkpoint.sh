#!/usr/bin/env bash
# Evaluate a TRAINED run (smoke / specialist) in the simulator with the same loop as the baseline.
# Starts a fresh server for <run_dir>, runs scripts/run_eval.py, stops the server. Extra args go to run_eval.py.
#   bash scripts/eval_checkpoint.sh outputs/finetune/<run> --kind smoke --split selection --episodes-per-task 3 --positions 0
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
RUN="$(cd "$1" && pwd)"; shift
"$ROOT/scripts/gpu_guard.sh"
PORT=$("$CLVLA_SIM_PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
mkdir -p "$ROOT/logs"
SLOG="$ROOT/logs/trained_server_$(date -u +%Y%m%dT%H%M%SZ)_$PORT.log"
setsid "$ROOT/scripts/serve_trained_policy.sh" "$RUN" "$PORT" > "$SLOG" 2>&1 < /dev/null &
SPID=$!
cleanup() { kill -TERM -- "-$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true; }
trap cleanup EXIT
for _ in $(seq 1 300); do
  curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && break
  kill -0 "$SPID" 2>/dev/null || { echo "trained policy server died:"; tail -30 "$SLOG"; exit 1; }
  sleep 3
done
curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null || { echo "server not healthy"; exit 1; }
FP=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1])).get('trained_checkpoint_fingerprint',''))" "$RUN/summary.json" 2>/dev/null || true)
export CLVLA_SERVER_INFO
CLVLA_SERVER_INFO=$(printf '{"command": "scripts/serve_trained_policy.sh %s %s", "log": "%s", "policy": "trained", "trained_run": "%s", "trained_checkpoint_fingerprint": "%s", "xla_flags": "%s"}' \
  "$RUN" "$PORT" "$SLOG" "$RUN" "$FP" "${CLVLA_XLA_FLAGS:-}")
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$ROOT/scripts/py-sim" "$ROOT/scripts/run_eval.py" --host 127.0.0.1 --port "$PORT" "$@"
