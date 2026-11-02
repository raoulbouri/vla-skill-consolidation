#!/usr/bin/env bash
# R8 screening with a fresh OFFICIAL pi05_libero server (GPU guard first). Extra args go to scripts/screen_tasks.py.
#   bash scripts/screen_tasks.sh                       # all tasks in configs/proposed/screening_v1.1.yaml
#   bash scripts/screen_tasks.sh --indices 55,57
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
"$ROOT/scripts/gpu_guard.sh"
PORT=$("$CLVLA_SIM_PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
mkdir -p "$ROOT/logs"
setsid "$ROOT/scripts/serve_policy.sh" "$PORT" > "$ROOT/logs/policy_server_screening_$(date -u +%Y%m%dT%H%M%SZ)_$PORT.log" 2>&1 < /dev/null &
SPID=$!
trap 'kill -TERM -- "-$SPID" 2>/dev/null || true; wait "$SPID" 2>/dev/null || true' EXIT
for _ in $(seq 1 300); do curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && break; kill -0 "$SPID" || exit 1; sleep 3; done
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$ROOT/scripts/py-sim" "$ROOT/scripts/screen_tasks.py" --port "$PORT" "$@"
