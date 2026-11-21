#!/usr/bin/env bash
# Ablation of curriculum task 1 (t063): the full run's recipe from its task-0 student, but no old-task replay and no
# buffer admission (configs/cl_opd_t063_noreplay.yaml, old_tasks.per_update 0); then held-out evaluation.
#   scripts/cl/t063_noreplay.sh
# CL_FROM_RUN: the run whose finished task 0 is reused (default: the run in outputs/cl_opd/LATEST_FULL). Its
# k0_t035 directory is linked, not copied. CL_RUN continues an existing ablation directory instead of starting one.
set -uo pipefail
source "$(dirname "$0")/common.sh"
CFG=configs/cl_opd_t063_noreplay.yaml
if [ -n "${CL_RUN:-}" ]; then
  RUN="$CL_RUN"
else
  FROM="${CL_FROM_RUN:-$(cat outputs/cl_opd/LATEST_FULL 2>/dev/null)}"
  [ -n "$FROM" ] && [ -d "$FROM/k0_t035" ] || { echo "no finished task 0: set CL_FROM_RUN to a run with k0_t035/"; exit 2; }
  RUN="outputs/cl_opd/t063_noreplay_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RUN" && ln -s "../$(basename "$FROM")/k0_t035" "$RUN/k0_t035"
  echo "$RUN" > outputs/cl_opd/LATEST_T063_NOREPLAY
fi
frozen_check START
echo "=== $RUN task 1 (no replay) gpu $GPU $(date -u +%FT%TZ)"
scripts/py-policy scripts/cl_opd_run.py --config $CFG --run "$RUN" --task 1 --gpu "$GPU" --stage train || { echo "T063_NOREPLAY_TRAIN_FAILED"; exit 1; }
scripts/py-policy scripts/cl_opd_run.py --config $CFG --run "$RUN" --task 1 --gpu "$GPU" --stage eval || { echo "T063_NOREPLAY_EVAL_FAILED"; exit 1; }
frozen_check END
print_matrix "$RUN"
echo "T063_NOREPLAY_DONE $(date -u +%FT%TZ)"
