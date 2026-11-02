#!/usr/bin/env bash
# Start scripts/train_teachers_parallel.sh detached, with the W&B API key taken from STDIN (never from the command
# line, a file or a log). Run it from the machine that holds the key, e.g.:
#   grep '^WANDB_API_KEY=' ~/Developer/wandb.env | cut -d= -f2- | ssh aims 'bash ~/IRL-project/scripts/launch_training_wandb.sh SPEC [SPEC ...]'
# The key is checked with W&B before anything starts; only the log path is printed.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
cd "$ROOT"
IFS= read -r WANDB_API_KEY || true
[ -n "${WANDB_API_KEY:-}" ] || { echo "no W&B key on stdin"; exit 2; }
export WANDB_API_KEY
# Verify through the API (does NOT write ~/.netrc; wandb.init later reads WANDB_API_KEY from the environment).
CUDA_VISIBLE_DEVICES= "$CLVLA_POLICY_PY" -c "import os, wandb; u = wandb.Api(api_key=os.environ['WANDB_API_KEY']).viewer; print('[wandb] key verified for', getattr(u, 'entity', None) or getattr(u, 'username', '?'))" 2>&1 | grep -v -i warn || { echo "W&B key check failed"; exit 3; }
S=$(date -u +%Y%m%dT%H%M%SZ)
LOG="logs/training_$S.log"
setsid nohup bash scripts/train_teachers_parallel.sh "$@" > "$LOG" 2>&1 < /dev/null &
echo "launched; log $ROOT/$LOG"
