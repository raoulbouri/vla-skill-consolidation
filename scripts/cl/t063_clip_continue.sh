#!/usr/bin/env bash
# Clipping pilot, continued (follow-up B): the clipped t063 run of scripts/cl/t063_clip_pilot.sh resumed from its last
# saved training state (update 190: optimizer, replay buffer, generators) and trained to update 400
# (configs/cl_opd_t063_clip3_u400.yaml; same recipe, total_updates 400), then the held-out evaluation (t035, t063, t042).
#   CL_FROM_PILOT=<pilot run>/k1_t063 scripts/cl/t063_clip_continue.sh
# CL_FROM_PILOT: the pilot's task-1 directory with student/state/ (state.json, master, opt_state, replay, sampler,
# diag), student/metrics.json and its training CSVs; it is copied into a new run directory, never modified.
# CL_FROM_TASK0: as in t063_clip_pilot.sh (default <run in outputs/cl_opd/LATEST_FULL>/k0_t035).
# Updates 191-200 are trained again and compared with the pilot's rows (REPRO line): equal rows mean the resume is exact.
set -uo pipefail
source "$(dirname "$0")/common.sh"
CFG=configs/cl_opd_t063_clip3_u400.yaml
FROM="${CL_FROM_TASK0:-$(cat outputs/cl_opd/LATEST_FULL 2>/dev/null)/k0_t035}"
PILOT="${CL_FROM_PILOT:?set CL_FROM_PILOT to the k1_t063 directory of the pilot run}"
for f in DONE student/metrics.json teacher_episodes; do
  [ -e "$FROM/$f" ] || { echo "not a finished task 0: $FROM/$f missing (set CL_FROM_TASK0)"; exit 2; }
done
for f in student/state/state.json student/state/master.npz student/state/opt_state.npz student/state/replay.pkl student/state/old_sampler.json student/state/diag.npz student/metrics.json; do
  [ -e "$PILOT/$f" ] || { echo "pilot state incomplete: $PILOT/$f missing"; exit 2; }
done
RUN="outputs/cl_opd/t063_clip3_u400_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN/k1_t063" && ln -s "$(realpath "$FROM")" "$RUN/k0_t035"
cp -a "$PILOT/student" "$RUN/k1_t063/" && { [ ! -d "$PILOT/logs" ] || cp -a "$PILOT/logs" "$RUN/k1_t063/"; } || { echo "T063_CONT_COPY_FAILED"; exit 1; }
rm -rf "$RUN/k1_t063/student/state.tmp" "$RUN/k1_t063/student/state.old"
echo "$RUN" > outputs/cl_opd/LATEST_T063_PILOT
echo "=== $RUN resumes the pilot at update $("$CLVLA_POLICY_PY" -I -c 'import json, sys; print(json.load(open(sys.argv[1]))["update"])' "$RUN/k1_t063/student/state/state.json")"
frozen_check START
echo "=== $RUN task 0 buffer (CPU) $(date -u +%FT%TZ)"
scripts/py-policy scripts/cl_opd_run.py --config $CFG --run "$RUN" --task 0 --stage buffer || { echo "T063_CONT_BUFFER_FAILED"; exit 1; }
echo "=== $RUN task 1 train, gpu $GPU $(date -u +%FT%TZ)"
scripts/py-policy scripts/cl_opd_run.py --config $CFG --run "$RUN" --task 1 --gpu "$GPU" --stage train || { echo "T063_CONT_TRAIN_FAILED"; exit 1; }
"$CLVLA_POLICY_PY" -I - "$PILOT/student" "$RUN/k1_t063/student" <<'PY'
import csv, glob, sys
def rows(d):
    out = {}
    for f in sorted(glob.glob(d + "/training_metrics*.csv")):
        for r in csv.DictReader(open(f)):
            out.setdefault(int(r["update"]), r)  # first occurrence: the pilot's own rows in the copied files
    return out
def new_rows(d):
    out = {}
    for f in sorted(glob.glob(d + "/training_metrics_from_u1[0-9][0-9].csv")):
        if f.endswith("_u100.csv"):
            continue
        for r in csv.DictReader(open(f)):
            out[int(r["update"])] = r
    return out
P, N = rows(sys.argv[1]), new_rows(sys.argv[2])
us = [u for u in range(191, 201) if u in P and u in N]
cols = ("kl_mean", "grad_norm_unclipped", "grad_norm", "rollout_success", "master_norm")
d = max((abs(float(N[u][c]) - float(P[u][c])) / max(abs(float(P[u][c])), 1e-30) for u in us for c in cols), default=float("nan"))
print(f"REPRO updates {us[0] if us else '-'}-{us[-1] if us else '-'} vs pilot: max relative difference {d:.3g} over {','.join(cols)}" + (" (exact)" if d == 0 else ""))
PY
echo "=== $RUN task 1 eval $(date -u +%FT%TZ)"
scripts/py-policy scripts/cl_opd_run.py --config $CFG --run "$RUN" --task 1 --gpu "$GPU" --stage eval || { echo "T063_CONT_EVAL_FAILED"; exit 1; }
frozen_check END
print_matrix "$RUN"
echo "T063_CONT_DONE $RUN $(date -u +%FT%TZ)"
