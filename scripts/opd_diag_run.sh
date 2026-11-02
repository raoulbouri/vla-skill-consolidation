#!/usr/bin/env bash
# t35 diagnostics, Experiments 1-3, inference only, GPU 1. Phase A: Experiment 1 rollouts in 3 parallel workers. Phase B, once the teacher and student500 rollouts exist: one takeover worker (student500, then teacher resume) and the scorer in parallel, while student100/base rollouts finish; at most 3 GPU jobs at once (about 8.7 GB per server).
# Usage: opd_diag_run.sh <diag_run_dir> <training_run_dir> <teacher_delta_dir>
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="$1"; SRC="$2"; T="$3"
export CLVLA_GPU=1 CLVLA_USE_GPU=1
source env.sh
export CLVLA_GPU=1
mkdir -p "$RUN/logs"
scripts/gpu_guard.sh 1 || { echo GUARD_REFUSED; exit 3; }
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_START
CK="$SRC/student10d/checkpoints"
E1="$RUN/exp1"; E2="$RUN/exp2b"
printf "teacher10d teacher_delta $T opd_diag_rollout.py --out $E1/teacher10d --save-obs\n" > "$RUN/jobs_a"
printf "student500 student $CK/adapter_500.npz opd_diag_rollout.py --out $E1/student500 --save-obs\n" > "$RUN/jobs_b"
printf "student100 student $CK/adapter_100.npz opd_diag_rollout.py --out $E1/student100\nbase base - opd_diag_rollout.py --out $E1/base\n" > "$RUN/jobs_c"
for j in a b c; do bash scripts/opd_diag_worker.sh "$RUN" 1 "$RUN/jobs_$j" > "$RUN/logs/worker_$j.log" 2>&1 & eval "W$j=\$!"; done
wait "$Wa" "$Wb"
echo PHASE_A_TEACHER_STUDENT500_DONE
printf "student500_takeover student $CK/adapter_500.npz opd_diag_takeover.py --teacher-dir $E1/teacher10d --out $E2/student500\nteacher10d_resume teacher_delta $T opd_diag_takeover.py --teacher-dir $E1/teacher10d --out $E2/teacher10d\n" > "$RUN/jobs_d"
bash scripts/opd_diag_worker.sh "$RUN" 1 "$RUN/jobs_d" > "$RUN/logs/worker_d.log" 2>&1 &
scripts/gpu_guard.sh 1 && MALLOC_ARENA_MAX=2 XLA_PYTHON_CLIENT_MEM_FRACTION=.3 PYTHONPATH=$PWD/scripts nice -n 5 scripts/py-policy -W ignore scripts/opd_diag_score.py --diag "$RUN" --teacher-delta "$T" --checkpoints "$CK" > "$RUN/logs/score.log" 2>&1 < /dev/null
echo score_exit $?
wait
cat "$RUN"/logs/worker_?.log | grep -vE 'gpu_guard|WORKER_DONE' | cut -c1-200
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_END
echo DIAG_DONE
