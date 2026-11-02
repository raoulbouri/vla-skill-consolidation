#!/usr/bin/env bash
# One serial queue on one GPU: serve each policy, evaluate it, stop the server. Usage: combined_worker.sh <run_dir> <gpu> <jobs_file> [extra eval args]. Job line: policy_id kind run_dir_or_- positions modes
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$1"; GPU="$2"; JOBS="$3"; shift 3
export CLVLA_GPU="$GPU" CLVLA_USE_GPU=1
source "$ROOT/env.sh"
export CLVLA_GPU="$GPU"
cd "$ROOT"
OUT="$RUN/${EVAL_SUBDIR:-stage1}"; mkdir -p "$OUT/workers" "$RUN/logs"
W=$(basename "$JOBS")
while read -r ID KIND PATH_ POS MODES; do
  [ -z "${ID:-}" ] && continue
  scripts/gpu_guard.sh "$GPU" || { echo "GUARD_REFUSED $ID"; exit 3; }
  PORT=$("$CLVLA_SIM_PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])')
  case "$KIND" in
    base) A=(--kind base);;
    teacher) A=(--kind teacher --run-dir "$PATH_");;
    student) A=(--kind student --adapter "$PATH_");;
    teacher_delta) A=(--kind teacher_delta --teacher-delta "$PATH_");;
  esac
  PYTHONPATH="$ROOT/scripts" XLA_PYTHON_CLIENT_MEM_FRACTION="${SERVER_MEM_FRACTION:-.3}" setsid scripts/py-policy -W ignore scripts/combined_serve.py "${A[@]}" --port "$PORT" > "$RUN/logs/server_${ID}.log" 2>&1 < /dev/null &
  SPID=$!
  ok=0; for _ in $(seq 1 200); do curl -sf "http://127.0.0.1:$PORT/healthz" >/dev/null && { ok=1; break; }; kill -0 "$SPID" 2>/dev/null || break; sleep 3; done
  if [ "$ok" = 1 ]; then
    PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/py-sim -W ignore scripts/combined_eval.py --port "$PORT" --policy-id "$ID" --positions "$POS" --modes "$MODES" --out "$OUT/workers/$W.jsonl" "$@" >> "$RUN/logs/eval_$W.log" 2>&1 < /dev/null
    echo "$ID eval_exit $?"
  else
    echo "$ID SERVER_FAILED"; tail -5 "$RUN/logs/server_${ID}.log" | cut -c1-200
  fi
  kill -TERM -- "-$SPID" 2>/dev/null; wait "$SPID" 2>/dev/null
done < "$JOBS"
echo WORKER_DONE
