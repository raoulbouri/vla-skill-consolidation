#!/usr/bin/env bash
# Stage 2 smoke test on one GPU: student SDE server, one LIBERO episode, then replay and teacher checks in a fresh process. Usage: bash scripts/opd_stage2.sh [demo] [base_seed]
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CLVLA_GPU=1 CLVLA_USE_GPU=1
source "$ROOT/env.sh"
cd "$ROOT"
scripts/gpu_guard.sh 1 || exit 3
DEMO="${1:-0}"; SEED="${2:-50000}"
STAMP=$(date -u +%Y%m%dT%H%M%SZ); RUN="outputs/opd_stage2/${STAMP}_t035_demo${DEMO}"; mkdir -p "$RUN"
PORT=$("$CLVLA_SIM_PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95
setsid scripts/py-policy -W ignore scripts/opd_stage2_serve.py --port "$PORT" --record-dir "$RUN" --base-seed "$SEED" > "$RUN/server.log" 2>&1 < /dev/null &
SPID=$!
stop() { kill -TERM -- "-$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null; }
trap stop EXIT
for _ in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && break; kill -0 "$SPID" 2>/dev/null || { echo SERVER_DIED; tail -20 "$RUN/server.log"; exit 1; }; sleep 3; done
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/py-sim -W ignore scripts/opd_stage2_rollout.py --port "$PORT" --out "$RUN" --libero-index 35 --demo "$DEMO" > "$RUN/rollout.log" 2>&1 < /dev/null; rc=$?
tail -3 "$RUN/rollout.log" | cut -c1-400; echo rollout_exit $rc
stop; trap - EXIT
[ "$rc" = 0 ] || { tail -30 "$RUN/server.log" | cut -c1-300; exit 1; }
scripts/py-policy -W ignore scripts/opd_stage2_verify.py --run "$RUN" --teacher-delta outputs/opd_stage2/teacher_t035 > "$RUN/verify.log" 2>&1 < /dev/null; rc=$?
echo verify_exit $rc; [ "$rc" = 0 ] || tail -25 "$RUN/verify.log" | cut -c1-400
echo "$RUN"
