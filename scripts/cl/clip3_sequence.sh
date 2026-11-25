#!/usr/bin/env bash
# Clipped curriculum (follow-up B, full sequence): tasks 1-4 (t063, t042, t003, t016) with per-trajectory clipping k=3,
# 400 updates each (configs/cl_opd_clip3.yaml), from the finished unclipped task 0 of outputs/cl_opd/full_20261108T214604Z.
# Each task: train (Phase 1 + 2), buffer admission, held-out evaluation (scripts/cl_opd_run.py --stage all).
#   scripts/cl/clip3_sequence.sh
# Rerunning continues: finished tasks (DONE) are skipped, an unfinished one resumes from its saved state (every 10
# updates). The final training state of every task is kept as student/state_final (CL_KEEP_FINAL_STATE=1).
# To continue a run rebuilt elsewhere (AWS -> xulab): python3 scripts/cl/persist.py restore --from DEST --outputs outputs,
# then rerun this script; the run comes from outputs/cl_opd/LATEST_CLIP3_SEQ (or CL_RUN).
# Environment (all optional):
#   CL_RUN          run directory to continue (default: outputs/cl_opd/LATEST_CLIP3_SEQ, else a new one)
#   CL_FROM_TASK0   finished task-0 directory (default: <run in outputs/cl_opd/LATEST_FULL>/k0_t035); linked, never modified
#   CL_TASKS        task indices to run, in order (default "1 2 3 4"); a task needs the previous one DONE
#   CL_PERSIST_DIR  if set, a persist.py snapshot to this directory after every stage
set -uo pipefail
source "$(dirname "$0")/common.sh"
CFG=configs/cl_opd_clip3.yaml
PTR=outputs/cl_opd/LATEST_CLIP3_SEQ
export CL_KEEP_FINAL_STATE=1
mkdir -p outputs/cl_opd
RUN="${CL_RUN:-$(cat "$PTR" 2>/dev/null)}"
if [ -z "$RUN" ]; then
  RUN="outputs/cl_opd/clip3_seq_$(date -u +%Y%m%dT%H%M%SZ)"
  echo "$RUN" > "$PTR"
fi
mkdir -p "$RUN"
if [ ! -e "$RUN/k0_t035" ]; then
  FROM="${CL_FROM_TASK0:-$(cat outputs/cl_opd/LATEST_FULL 2>/dev/null)/k0_t035}"
  for f in DONE student/metrics.json teacher_episodes; do
    [ -e "$FROM/$f" ] || { echo "not a finished task 0: $FROM/$f missing (set CL_FROM_TASK0)"; exit 2; }
  done
  ln -s "$(realpath "$FROM")" "$RUN/k0_t035"
fi
persist() {
  [ -n "${CL_PERSIST_DIR:-}" ] || return 0
  scripts/py-policy scripts/cl/persist.py snapshot --outputs outputs/cl_opd --run "$(basename "$RUN")" --to "$CL_PERSIST_DIR" --pointer LATEST_CLIP3_SEQ --min-age 0 \
    || echo "PERSIST_FAILED (continuing)"
}
frozen_check START
echo "=== $RUN task 0 buffer (CPU) $(date -u +%FT%TZ)"
scripts/py-policy scripts/cl_opd_run.py --config "$CFG" --run "$RUN" --task 0 --stage buffer || { echo "CLIP3_SEQ_BUFFER0_FAILED"; exit 1; }
persist
for K in ${CL_TASKS:-1 2 3 4}; do
  D=$(ls -d "$RUN"/k${K}_* 2>/dev/null | head -1)
  if [ -n "$D" ] && [ -e "$D/DONE" ]; then echo "=== task $K already done ($D)"; continue; fi
  echo "=== $RUN task $K (train, buffer, eval) gpu $GPU $(date -u +%FT%TZ)"
  scripts/py-policy scripts/cl_opd_run.py --config "$CFG" --run "$RUN" --task "$K" --gpu "$GPU"
  rc=$?
  persist
  [ $rc = 0 ] || { echo "CLIP3_SEQ_TASK_${K}_FAILED $(date -u +%FT%TZ)"; exit 1; }
  print_matrix "$RUN"
  echo "CLIP3_SEQ_TASK_${K}_DONE $(date -u +%FT%TZ)"
done
frozen_check END
echo "CLIP3_SEQ_DONE $RUN $(date -u +%FT%TZ)"
