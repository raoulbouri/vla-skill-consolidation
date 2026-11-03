#!/usr/bin/env bash
# Default teacher-to-student distillation (pathwise OPD, LoRA student) on one LIBERO-90 task; the recipe that matched Teacher-10D on t35 (26/40 vs 24/40 ODE on demo_10..49, docs/OPD_EXPERIMENT_REGISTRY.md section 8).
#  Phase 1, updates 1-100: the teacher drives the robot (successful episodes only, 8 observations chosen by phase); the student samples its own flow chains on them. The full training state is saved at update 100.
#  Phase 2, updates 101-400: the student drives, 4 episodes per update (8 observations each, 1 noise draw) plus 8 observations replayed from a per-phase buffer
#   (1000 per phase; seeded with Phase 1's teacher-driven observations, which are never evicted; student observations fill the rest by reservoir sampling); stops early if robot-door contact falls below 5% of the last 100 episodes.
#  Evaluation: teacher, final student and base on the 40 held-out states demo_10..49 (ODE and SDE, geometry), then comparison.json and side_by_side.csv (similar rule: upper 95% limit of teacher - student ODE success <= 0.15).
# Usage: opd_distill.sh <run_dir> <teacher_delta_dir>
# Env: STAGE=all|train|eval (default all); GPU (default 1); TRAIN_MEM_FRACTION (default .75); SHARED_GPU_OK=1 lets training start on a GPU other users are on (evaluation never does).
# Needs about 120 GiB host RAM and 21 GB GPU memory for training, 29 GB for evaluation (three policy servers); about 1 h for Phase 1, 5.5 h for Phase 2, 30 min for evaluation on an RTX 5090.
set -uo pipefail
cd "$(dirname "$0")/.."
RUN="$1"; T="$2"
GPU="${GPU:-1}"; STAGE="${STAGE:-all}"; UPDATES=400; WARM=100
export CLVLA_GPU=$GPU CLVLA_USE_GPU=1
source env.sh
export CLVLA_GPU=$GPU
mkdir -p "$RUN/logs"
S="$RUN/student"
done_updates() { python3 -c "import json; print(json.load(open('$S/metrics.json')).get('completed_updates', 0))" 2>/dev/null || echo 0; }
train() {
  MALLOC_ARENA_MAX=2 XLA_PYTHON_CLIENT_MEM_FRACTION="${TRAIN_MEM_FRACTION:-.75}" PYTHONPATH=$PWD/scripts:$PWD/tests/ogpo_port nice -n 5 scripts/py-policy -W ignore scripts/opd_lora_train.py --teacher-run "$PWD/$T" --teacher-delta "$T" --name student --out "$S" --position 4 --updates $UPDATES --random-starts --arm-noise 0.04 --profile n10 --teacher-control-updates $WARM --teacher-success-only "$@"
}
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_START
if [ "$STAGE" != eval ]; then
  if ! scripts/gpu_guard.sh "$GPU"; then
    [ "${SHARED_GPU_OK:-0}" = 1 ] || { echo GUARD_REFUSED; exit 3; }
    echo "SHARED_GPU_OK: training starts on shared GPU $GPU ($(date -u +%FT%TZ))"
  fi
  if [ ! -f "$S/state/state.json" ]; then
    train --stop-after $WARM > "$RUN/logs/train_phase1.log" 2>&1 < /dev/null
    echo phase1_exit $?
    [ "$(done_updates)" = $WARM ] || { echo PHASE1_INCOMPLETE; tail -5 "$RUN/logs/train_phase1.log" | cut -c1-300; exit 1; }
  fi
  train --resume --rollouts-per-update 4 --chains-per-obs 1 --replay-per-update 8 --replay-capacity 1000 --stop-contact-rate 0.05 --stop-window 100 --stop-min-episodes 200 > "$RUN/logs/train_phase2.log" 2>&1 < /dev/null
  echo phase2_exit $?
  grep -E '^\[end\]' "$RUN/logs/train_phase2.log" | cut -c1-400
  [ "$(done_updates)" -gt $WARM ] || { echo PHASE2_NO_PROGRESS; exit 1; }
  [ "$STAGE" = train ] && { echo TRAIN_DONE; exit 0; }
fi
U=$(done_updates); F=$(printf "%03d" "$U")
[ -f "$S/checkpoints/adapter_$F.npz" ] || { echo "NO_CHECKPOINT adapter_$F.npz"; exit 1; }
scripts/gpu_guard.sh "$GPU" || { echo "EVAL_REFUSED: GPU $GPU has other users"; exit 4; }
echo "teacher teacher_delta $T 4 ode,sde" > "$RUN/jobs_a"
echo "student student $S/checkpoints/adapter_$F.npz 4 ode,sde" > "$RUN/jobs_b"
echo "base base - 4 ode,sde" > "$RUN/jobs_c"
export EVAL_SUBDIR=evaluation SERVER_MEM_FRACTION=.3
for j in a b c; do bash scripts/combined_worker.sh "$RUN" "$GPU" "$RUN/jobs_$j" --profile n10 --states 40 --door-joint microwave_1_microjoint --geometry > "$RUN/logs/worker_$j.log" 2>&1 & done
wait
cat "$RUN"/logs/worker_?.log | grep -vE 'gpu_guard|WORKER_DONE' | cut -c1-200
python3 scripts/opd_curr_compare.py "$RUN" student teacher,student,base teacher > "$RUN/logs/compare.log" 2>&1; echo compare_exit $?
python3 -c "import json; c=json.load(open('$RUN/comparison.json')); print('student update $U', {k: c['cells'][k]['successes'] for k in c['cells']}, 'similar', c['similar_rule'])"
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_END
echo DISTILL_DONE
