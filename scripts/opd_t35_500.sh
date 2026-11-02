#!/usr/bin/env bash
# t35 Student-10D: up to 500 LoRA pathwise OPD updates on rollouts from fresh random resets, then 40-state evaluation only if the run completed. Usage: opd_t35_pilot.sh <run_dir> <teacher_delta_dir> <100_update_adapter>
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="$1"; T="$2"; S100="$3"
export CLVLA_GPU=1 CLVLA_USE_GPU=1
source env.sh
export CLVLA_GPU=1
mkdir -p "$RUN/logs"
scripts/gpu_guard.sh 1 || { echo GUARD_REFUSED; exit 3; }
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_START
MALLOC_ARENA_MAX=2 XLA_PYTHON_CLIENT_MEM_FRACTION=.75 PYTHONPATH=$PWD/scripts:$PWD/tests/ogpo_port nice -n 5 scripts/py-policy -W ignore scripts/opd_lora_train.py --teacher-run "$PWD/$T" --teacher-delta "$T" --name student10d --out "$RUN/student10d" --position 4 --updates 500 --random-starts --arm-noise 0.04 --profile n10 > "$RUN/logs/train.log" 2>&1 < /dev/null
echo train_exit $?
grep -E '^\[end\]' "$RUN/logs/train.log" | cut -c1-400
python3 - "$RUN/student10d/metrics.json" <<'PY' || { echo TRAINING_INCOMPLETE; tail -5 "$RUN/logs/train.log" | cut -c1-300; echo PILOT_DONE; exit 1; }
import json, sys
r = json.load(open(sys.argv[1]))
sys.exit(0 if r.get("completed_updates") == 500 and r.get("stop_reason") is None else 1)
PY
printf "base base - 4 ode,sde\nstudent100 student $S100 4 ode,sde\n" > "$RUN/jobs_a"; printf "teacher10d teacher_delta $T 4 ode,sde\n" > "$RUN/jobs_b"; printf "student500 student $RUN/student10d/checkpoints/adapter_500.npz 4 ode,sde\n" > "$RUN/jobs_c"
export EVAL_SUBDIR=evaluation SERVER_MEM_FRACTION=.3
for j in a b c; do bash scripts/combined_worker.sh "$RUN" 1 "$RUN/jobs_$j" --profile n10 --states 40 --door-joint microwave_1_microjoint > "$RUN/logs/worker_$j.log" 2>&1 & done
wait
cat "$RUN"/logs/worker_?.log | grep -vE 'gpu_guard|WORKER_DONE' | cut -c1-200
python3 scripts/opd_t35_compare.py "$RUN" base,teacher10d,student100,student500 student500 > "$RUN/logs/compare.log" 2>&1; echo compare_exit $?
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_END
echo PILOT_DONE
