#!/usr/bin/env bash
# Curriculum task K of a continual-learning OPD run: Phase 1 (teacher drives) + Phase 2 (student drives) training, buffer
# admission, and the held-out evaluation row (scripts/cl_opd_run.py --stage all). Run K = 0, 1, ..., 4 in order.
#   scripts/cl/run_task.sh K
# Task 0 starts a new run directory and writes it to the pointer file; later tasks read the pointer. Rerunning the same
# K resumes from the saved training state. The driver refuses a run whose config hash differs, so a variant needs its
# own config file and its own pointer.
# Environment (all optional):
#   CL_CONFIG   config file                  (default configs/cl_opd.yaml)
#   CL_POINTER  run pointer file             (default outputs/cl_opd/LATEST_FULL for configs/cl_opd.yaml,
#               else outputs/cl_opd/LATEST_<config name>)
#   CL_RUN      run directory; skips the pointer
#   GPUQ_GPU    GPU index set by a scheduler (default CLVLA_GPU from .env)
set -uo pipefail
K="${1:?usage: scripts/cl/run_task.sh K}"
source "$(dirname "$0")/common.sh"
CFG="${CL_CONFIG:-configs/cl_opd.yaml}"
TAG="$(basename "$CFG" .yaml)"
if [ "$TAG" = cl_opd ]; then TAG=full; fi   # names of the original run: outputs/cl_opd/full_<stamp>, LATEST_FULL
PTR="${CL_POINTER:-outputs/cl_opd/LATEST_${TAG^^}}"
if [ -n "${CL_RUN:-}" ]; then
  RUN="$CL_RUN"
else
  mkdir -p "$(dirname "$PTR")"
  if [ "$K" = 0 ] && [ ! -s "$PTR" ]; then echo "outputs/cl_opd/${TAG}_$(date -u +%Y%m%dT%H%M%SZ)" > "$PTR"; fi
  [ -s "$PTR" ] || { echo "no run pointer $PTR: run task 0 first or set CL_RUN"; exit 2; }
  RUN="$(cat "$PTR")"
fi
mkdir -p "$RUN"
frozen_check START
echo "=== $RUN task $K config $CFG gpu $GPU $(date -u +%FT%TZ)"
scripts/py-policy scripts/cl_opd_run.py --config "$CFG" --run "$RUN" --task "$K" --gpu "$GPU" || { echo "TASK_${K}_FAILED"; exit 1; }
frozen_check END
print_matrix "$RUN"
echo "TASK_${K}_DONE $(date -u +%FT%TZ)"
