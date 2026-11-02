#!/usr/bin/env bash
# Serve a trained run on CLVLA_GPU (foreground). Usage: scripts/serve_trained_policy.sh <run_dir> [port] [step]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
RUN="$1"; PORT="${2:-$CLVLA_SERVER_PORT}"; STEP="${3:-}"
"$ROOT/scripts/gpu_guard.sh" "$CLVLA_GPU"
[ -f "$RUN/run_metadata.json" ] || { echo "not a clvla run dir: $RUN"; exit 1; }
export CUDA_VISIBLE_DEVICES="$CLVLA_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_FLAGS="${CLVLA_XLA_FLAGS:-}"
export PYTHONPATH="$CLVLA_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
echo "[serve_trained_policy] run=$RUN step=${STEP:-latest} gpu=$CLVLA_GPU port=$PORT XLA_FLAGS='$XLA_FLAGS'"
exec "$CLVLA_POLICY_PY" "$ROOT/scripts/serve_trained_policy.py" --run-dir "$RUN" --port "$PORT" ${STEP:+--step "$STEP"}
