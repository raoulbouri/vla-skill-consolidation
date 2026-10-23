#!/usr/bin/env bash
# Start OpenPI's official policy server with the pinned pi05_libero checkpoint on CLVLA_GPU (foreground).
# Usage: scripts/serve_policy.sh [port]       Health check: curl -s localhost:<port>/healthz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
PORT="${1:-$CLVLA_SERVER_PORT}"
CKPT="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero"
"$ROOT/scripts/gpu_guard.sh" "$CLVLA_GPU"
[ -f "$CKPT/params/_METADATA" ] || { echo "checkpoint missing: $CKPT (run scripts/validate_assets.py)"; exit 1; }
export CUDA_VISIBLE_DEVICES="$CLVLA_GPU" XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$OPENPI_DIR"
echo "[serve_policy] openpi $(git rev-parse HEAD) config=pi05_libero dir=$CKPT gpu=$CLVLA_GPU port=$PORT"
exec "$CLVLA_POLICY_PY" scripts/serve_policy.py --env LIBERO --port "$PORT" \
  policy:checkpoint --policy.config pi05_libero --policy.dir "$CKPT"
