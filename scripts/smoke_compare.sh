#!/usr/bin/env bash
# Phase E smoke comparison: ORIGINAL pi05_libero vs the TRAINED smoke run on the same 3 selection init states (0,1,2)
# of task position 0, same loop/instruction/env setup; videos saved; then an episode-level comparison.
# Smoke passes on infrastructure (no infra errors), NOT on success improving. Never uses init states >= 10.
#   bash scripts/smoke_compare.sh outputs/finetune/<smoke run>
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN="$1"
COMMON=(--kind smoke --split selection --episodes-per-task 3 --positions 0 --save-videos)
bash "$ROOT/scripts/baseline_eval.sh" "${COMMON[@]}" --name smoke_original_t078
bash "$ROOT/scripts/eval_checkpoint.sh" "$RUN" "${COMMON[@]}" --name smoke_trained_t078
A=$(ls -td "$ROOT"/outputs/smoke/*_smoke_original_t078 | head -1)
B=$(ls -td "$ROOT"/outputs/smoke/*_smoke_trained_t078 | head -1)
"$ROOT/scripts/py-policy" "$ROOT/scripts/compare_runs.py" "$A" "$B" || true   # differences are expected after training
echo "original: $A"; echo "trained:  $B"
