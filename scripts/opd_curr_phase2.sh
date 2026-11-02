#!/usr/bin/env bash
# Two-phase curriculum, Phase 2 and final evaluation, GPU 1; run only after the update-100 gate is reviewed. Resumes the paused run for updates 101-500 on student-driven episodes, then evaluates on held-out demo_10..49 (ODE and SDE, with geometry) and scores all checkpoints on the development set.
# Usage: opd_curr_phase2.sh <run_dir> <teacher_delta_dir> <previous_student500_adapter>
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="$1"; T="$2"; OLD="$3"
export CLVLA_GPU=1 CLVLA_USE_GPU=1
source env.sh
export CLVLA_GPU=1
scripts/gpu_guard.sh 1 || { echo GUARD_REFUSED; exit 3; }
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_START
MALLOC_ARENA_MAX=2 XLA_PYTHON_CLIENT_MEM_FRACTION=.75 PYTHONPATH=$PWD/scripts:$PWD/tests/ogpo_port nice -n 5 scripts/py-policy -W ignore scripts/opd_lora_train.py --teacher-run "$PWD/$T" --teacher-delta "$T" --name student_curriculum --out "$RUN/student" --position 4 --updates 500 --random-starts --arm-noise 0.04 --profile n10 --teacher-control-updates 100 --teacher-success-only --resume > "$RUN/logs/train_phase2.log" 2>&1 < /dev/null
echo train_exit $?
grep -E '^\[end\]' "$RUN/logs/train_phase2.log" | cut -c1-400
python3 - "$RUN/student/metrics.json" <<'PY' || { echo PHASE2_INCOMPLETE; tail -5 "$RUN/logs/train_phase2.log" | cut -c1-300; echo PHASE2_DONE; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if r.get("completed_updates") == 500 and r.get("stop_reason") is None else 1)
PY
CK="$RUN/student/checkpoints"
printf "base base - 4 ode,sde\ncurr_u000 student $CK/adapter_000.npz 4 ode,sde\n" > "$RUN/jobs_a"
printf "teacher10d teacher_delta $T 4 ode,sde\ncurr_u100 student $CK/adapter_100.npz 4 ode,sde\n" > "$RUN/jobs_b"
printf "curr_u500 student $CK/adapter_500.npz 4 ode,sde\nstudent500_old student $OLD 4 ode,sde\n" > "$RUN/jobs_c"
export EVAL_SUBDIR=evaluation SERVER_MEM_FRACTION=.3
for j in a b c; do bash scripts/combined_worker.sh "$RUN" 1 "$RUN/jobs_$j" --profile n10 --states 40 --door-joint microwave_1_microjoint --geometry > "$RUN/logs/worker_$j.log" 2>&1 & done
wait
cat "$RUN"/logs/worker_?.log | grep -vE 'gpu_guard|WORKER_DONE' | cut -c1-200
python3 scripts/opd_curr_compare.py "$RUN" curr_u500 base,teacher10d,student500_old,curr_u000,curr_u100,curr_u500 > "$RUN/logs/compare.log" 2>&1; echo compare_exit $?
G="$RUN/gate"
printf "student_u500 student $CK/adapter_500.npz opd_diag_rollout.py --out $G/exp1/student_u500 --save-obs --start-seed0 960000 --noise-seed0 310000\n" > "$G/jobs_f"
bash scripts/opd_diag_worker.sh "$G" 1 "$G/jobs_f" > "$G/logs/worker_f.log" 2>&1
printf "student_u500 student $CK/adapter_500.npz opd_diag_takeover.py --teacher-dir $G/exp1/teacher10d --out $G/exp2b/student_u500\n" > "$G/jobs_g"
bash scripts/opd_diag_worker.sh "$G" 1 "$G/jobs_g" > "$G/logs/worker_g.log" 2>&1 &
scripts/gpu_guard.sh 1 && MALLOC_ARENA_MAX=2 XLA_PYTHON_CLIENT_MEM_FRACTION=.3 PYTHONPATH=$PWD/scripts nice -n 5 scripts/py-policy -W ignore scripts/opd_diag_score.py --diag "$G" --teacher-delta "$T" --checkpoints "$CK" --ckpts 0,25,50,75,100,200,300,400,500 --teacher-set teacher10d --student-set student_u500 --compare-ckpt 500 --out-name scores_final > "$G/logs/score_final.log" 2>&1 < /dev/null
echo score_final_exit $?
wait
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_END
echo PHASE2_DONE
