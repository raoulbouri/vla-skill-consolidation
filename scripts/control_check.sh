#!/usr/bin/env bash
# In-distribution control on libero_spatial with a fresh official server:
#   (a) clvla's evaluation loop (scripts/control_eval.py) and (b) OpenPI's unmodified examples/libero/main.py.
# Usage: bash scripts/control_check.sh [trials_per_task=2]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
TRIALS="${1:-2}"
"$ROOT/scripts/gpu_guard.sh"
PORT=$("$CLVLA_SIM_PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
STAMP=$(date -u +%Y%m%dT%H%M%SZ); mkdir -p "$ROOT/logs"
setsid "$ROOT/scripts/serve_policy.sh" "$PORT" > "$ROOT/logs/policy_server_${STAMP}_$PORT.log" 2>&1 < /dev/null &
SPID=$!
trap 'kill -TERM -- "-$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true' EXIT
until curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null; do kill -0 "$SPID" || exit 1; sleep 3; done
echo "=== (a) clvla loop"
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$ROOT/scripts/py-sim" "$ROOT/scripts/control_eval.py" --port "$PORT" --trials "$TRIALS"
echo "=== (b) OpenPI examples/libero/main.py (unmodified)"
VID="$CLVLA_OUTPUT_ROOT/smoke/${STAMP}_openpi_main_libero_spatial"
cd "$OPENPI_DIR" && "$ROOT/scripts/py-sim" examples/libero/main.py --args.host 127.0.0.1 --args.port "$PORT" \
  --args.task-suite-name libero_spatial --args.num-trials-per-task "$TRIALS" --args.video-out-path "$VID" 2>&1 \
  | grep -E "Total success rate|Total episodes"
