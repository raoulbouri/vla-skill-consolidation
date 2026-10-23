#!/usr/bin/env bash
# Refuse to use the shared GPU if another user has compute processes on it. Read-only: never touches other processes.
# Usage: scripts/gpu_guard.sh [gpu_index]   (default $CLVLA_GPU). Exit 0 = free for us, 1 = busy, 2 = no nvidia-smi.
set -uo pipefail
source "$(cd "$(dirname "$0")/.." && pwd)/env.sh"
GPU="${1:-$CLVLA_GPU}"
command -v nvidia-smi >/dev/null || { echo "gpu_guard: nvidia-smi not found"; exit 2; }
UUID=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g {print $2}')
[ -n "$UUID" ] || { echo "gpu_guard: GPU $GPU not found"; exit 2; }
busy=0
while IFS=', ' read -r uuid pid mem; do
  [ "$uuid" = "$UUID" ] || continue
  owner=$(ps -o user= -p "$pid" 2>/dev/null | tr -d ' ')
  if [ "$owner" != "$(whoami)" ]; then
    echo "gpu_guard: GPU $GPU has a compute process of user '${owner:-?}' (pid $pid, $mem); refusing to use it"
    busy=1
  fi
done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader)
[ "$busy" = 0 ] && echo "gpu_guard: GPU $GPU has no other users' compute processes"
exit "$busy"
