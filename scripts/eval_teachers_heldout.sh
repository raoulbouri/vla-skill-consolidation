#!/usr/bin/env bash
# Held-out comparison: original pi05_libero vs each 25-demo teacher, on the SAME 25 initial states per task
# (initial states of the held-out demos demo_25..49; never a training demo, never an official LIBERO init state).
#   setsid nohup bash scripts/eval_teachers_heldout.sh [baseline|teachers|all] > logs/eval_heldout_<stamp>.log 2>&1 < /dev/null &
# Serial (one policy server at a time on GPU $CLVLA_GPU). Resumable: finished evaluations are skipped.
# Then: scripts/py-policy scripts/compare_teachers.py   (paired table + CIs -> outputs/teacher_eval/comparison_n25.*)
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
cd "$ROOT"
WHAT="${1:-all}"
IDX=(78 84 16 56 35 42 28 3 89 63)   # manifest v1.0 order = positions 0..9
COMMON=(--init-source demos:n25 --episodes-per-task 25)
TEACHER="${CLVLA_TEACHER_NAME:-teacher-n25-s500}"   # training run name (spec `name`)
log() { echo "[eval_heldout $(date -u +%H:%M:%SZ)] $*"; }
fail=0

if [ "$WHAT" = baseline ] || [ "$WHAT" = all ]; then
  if ls outputs/baseline/*_pi05_libero_heldout_n25/summary.json >/dev/null 2>&1; then
    log "original checkpoint: held-out eval exists, skipping"
  else
    log "original checkpoint: 10 tasks x 25 held-out init states"
    bash scripts/baseline_eval.sh --kind baseline --name pi05_libero_heldout_n25 "${COMMON[@]}" || { log "baseline eval FAILED"; fail=1; }
  fi
fi
if [ "$WHAT" = teachers ] || [ "$WHAT" = all ]; then
  for pos in "${!IDX[@]}"; do
    t=$(printf %03d "${IDX[$pos]}")
    run=$(ls -d outputs/specialists/*_${TEACHER}-t$t 2>/dev/null | tail -1)
    if [ -z "$run" ] || ! python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['status']=='completed' else 1)" "$run/summary.json" 2>/dev/null; then
      log "t$t: no completed teacher run, skipping"; continue
    fi
    if ls outputs/teacher_eval/*_${TEACHER}-t${t}_heldout/summary.json >/dev/null 2>&1; then
      log "t$t: held-out eval exists, skipping"; continue
    fi
    log "t$t: evaluating $run on position $pos"
    bash scripts/eval_checkpoint.sh "$run" --kind teacher_eval --name "${TEACHER}-t${t}_heldout" --positions "$pos" "${COMMON[@]}" \
      || { log "t$t: eval FAILED"; fail=1; }
  done
fi
[ "$fail" = 0 ] && log "ALL DONE" || { log "DONE with failures"; exit 1; }
