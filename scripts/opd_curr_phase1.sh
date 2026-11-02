#!/usr/bin/env bash
# Two-phase curriculum, Phase 1 and gate, GPU 1: a fresh Student-10D trains on 100 teacher-driven episodes (successful only), then pauses with its full training state saved; the update-100 gate runs on development starts 960000-960039 and the script stops for review.
# Usage: opd_curr_phase1.sh <run_dir> <teacher_delta_dir>
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="$1"; T="$2"
export CLVLA_GPU=1 CLVLA_USE_GPU=1
source env.sh
export CLVLA_GPU=1
mkdir -p "$RUN/logs"
scripts/gpu_guard.sh 1 || { echo GUARD_REFUSED; exit 3; }
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_START
MALLOC_ARENA_MAX=2 XLA_PYTHON_CLIENT_MEM_FRACTION=.75 PYTHONPATH=$PWD/scripts:$PWD/tests/ogpo_port nice -n 5 scripts/py-policy -W ignore scripts/opd_lora_train.py --teacher-run "$PWD/$T" --teacher-delta "$T" --name student_curriculum --out "$RUN/student" --position 4 --updates 500 --random-starts --arm-noise 0.04 --profile n10 --teacher-control-updates 100 --teacher-success-only --stop-after 100 > "$RUN/logs/train_phase1.log" 2>&1 < /dev/null
echo train_exit $?
grep -E '^\[end\]' "$RUN/logs/train_phase1.log" | cut -c1-400
python3 - "$RUN/student/metrics.json" <<'PY' || { echo PHASE1_INCOMPLETE; tail -5 "$RUN/logs/train_phase1.log" | cut -c1-300; echo PHASE1_DONE; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if r.get("completed_updates") == 100 and str(r.get("stop_reason")).startswith("paused") else 1)
PY
G="$RUN/gate"; CK="$RUN/student/checkpoints"; mkdir -p "$G/logs"
printf "teacher10d teacher_delta $T opd_diag_rollout.py --out $G/exp1/teacher10d --save-obs --start-seed0 960000 --noise-seed0 310000\n" > "$G/jobs_a"
printf "student_u100 student $CK/adapter_100.npz opd_diag_rollout.py --out $G/exp1/student_u100 --save-obs --start-seed0 960000 --noise-seed0 310000\n" > "$G/jobs_b"
for j in a b; do bash scripts/opd_diag_worker.sh "$G" 1 "$G/jobs_$j" > "$G/logs/worker_$j.log" 2>&1 & eval "W$j=\$!"; done
wait "$Wa" "$Wb"
printf "student_u100 student $CK/adapter_100.npz opd_diag_takeover.py --teacher-dir $G/exp1/teacher10d --out $G/exp2b/student_u100\nteacher10d_resume teacher_delta $T opd_diag_takeover.py --teacher-dir $G/exp1/teacher10d --out $G/exp2b/teacher10d_resume\n" > "$G/jobs_d"
bash scripts/opd_diag_worker.sh "$G" 1 "$G/jobs_d" > "$G/logs/worker_d.log" 2>&1 &
scripts/gpu_guard.sh 1 && MALLOC_ARENA_MAX=2 XLA_PYTHON_CLIENT_MEM_FRACTION=.3 PYTHONPATH=$PWD/scripts nice -n 5 scripts/py-policy -W ignore scripts/opd_diag_score.py --diag "$G" --teacher-delta "$T" --checkpoints "$CK" --ckpts 0,25,50,75,100 --teacher-set teacher10d --student-set student_u100 --compare-ckpt 100 > "$G/logs/score.log" 2>&1 < /dev/null
echo score_exit $?
wait
cat "$G"/logs/worker_?.log | grep -vE 'gpu_guard|WORKER_DONE' | cut -c1-200
python3 scripts/opd_curr_gate.py "$G" student_u100 > "$G/logs/gate.log" 2>&1; echo gate_exit $?
python3 -c "import json; r=json.load(open('$G/gate.json')); print('GATE', r['rule'])"
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_END
echo PHASE1_DONE
