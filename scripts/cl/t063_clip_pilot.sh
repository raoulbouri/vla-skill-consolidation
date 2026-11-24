#!/usr/bin/env bash
# Clipping pilot (follow-up B): rerun curriculum task 1 (t063) from a finished task 0 with per-trajectory gradient
# clipping k=3, to update 200 (configs/cl_opd_t063_clip3_u200.yaml), then the held-out evaluation (t035, t063, t042).
# The unclipped reference is outputs/cl_opd/full_20261108T214604Z/k1_t063 (collapsed near update 148).
#   scripts/cl/t063_clip_pilot.sh
# CL_CONFIG / CL_TAG run another arm from the same start, e.g. the trust-region arm:
#   CL_CONFIG=configs/cl_opd_t063_tr3_u200.yaml CL_TAG=t063_tr3_u200 scripts/cl/t063_clip_pilot.sh
# CL_FROM_TASK0: the finished task-0 directory to start from (needs DONE, student/metrics.json, the final checkpoint
# and teacher_episodes/; default: <run in outputs/cl_opd/LATEST_FULL>/k0_t035). It is linked, never modified; the
# old-task buffer is rebuilt from its teacher_episodes/ by the driver's own buffer stage (CPU).
# CL_RUN continues an existing pilot directory instead of starting one (training resumes from its saved state).
set -uo pipefail
source "$(dirname "$0")/common.sh"
CFG="${CL_CONFIG:-configs/cl_opd_t063_clip3_u200.yaml}"
TAG="${CL_TAG:-t063_clip3_u200}"
if [ -n "${CL_RUN:-}" ]; then
  RUN="$CL_RUN"
else
  FROM="${CL_FROM_TASK0:-$(cat outputs/cl_opd/LATEST_FULL 2>/dev/null)/k0_t035}"
  for f in DONE student/metrics.json teacher_episodes; do
    [ -e "$FROM/$f" ] || { echo "not a finished task 0: $FROM/$f missing (set CL_FROM_TASK0)"; exit 2; }
  done
  RUN="outputs/cl_opd/${TAG}_$(date -u +%Y%m%dT%H%M%SZ)"
  mkdir -p "$RUN" && ln -s "$(realpath "$FROM")" "$RUN/k0_t035"
  echo "$RUN" > outputs/cl_opd/LATEST_T063_PILOT
fi
frozen_check START
echo "=== $RUN task 0 buffer (CPU) $(date -u +%FT%TZ)"
scripts/py-policy scripts/cl_opd_run.py --config "$CFG" --run "$RUN" --task 0 --stage buffer || { echo "T063_CLIP3_BUFFER_FAILED"; exit 1; }
echo "=== $RUN task 1 train, gpu $GPU $(date -u +%FT%TZ)"
scripts/py-policy scripts/cl_opd_run.py --config "$CFG" --run "$RUN" --task 1 --gpu "$GPU" --stage train || { echo "T063_CLIP3_TRAIN_FAILED"; exit 1; }
echo "=== $RUN task 1 eval $(date -u +%FT%TZ)"
scripts/py-policy scripts/cl_opd_run.py --config "$CFG" --run "$RUN" --task 1 --gpu "$GPU" --stage eval || { echo "T063_CLIP3_EVAL_FAILED"; exit 1; }
frozen_check END
print_matrix "$RUN"
echo "T063_CLIP3_DONE $RUN $(date -u +%FT%TZ)"
