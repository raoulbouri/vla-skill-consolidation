#!/usr/bin/env bash
# Train one run per (spec, manifest task), each from the ORIGINAL pi05_libero checkpoint, ONE RUN PER GPU, so with
# two GPUs two runs train at once.
#
#   setsid nohup bash scripts/train_teachers_parallel.sh SPEC [SPEC ...] > logs/training_<stamp>.log 2>&1 < /dev/null &
#
# The queue is task-major: (t78, spec1), (t78, spec2), (t84, spec1), ..., so the variants of one task train side by
# side. Specs with a `wandb:` block need WANDB_API_KEY in the environment: start through
# scripts/launch_training_wandb.sh, which reads the key from stdin.
#
#   CLVLA_TRAIN_GPUS       GPUs to use, one run each (default "0 1")
#   CLVLA_MEM_FRACTION     hard JAX memory cap per run (default 0.92)
#   CLVLA_VRAM_BUFFER_MIB  free memory that must remain on a GPU before a run starts there (default 0)
#   CLVLA_TRAIN_XLA_FLAGS  XLA flags for training (default --xla_gpu_autotune_level=0: aims' XLA segfaults in autotuning)
#
# Safety and resumability:
#   * the GPU guard runs before every launch (refuses a GPU with another user's compute process; never touches it);
#   * a (spec, task) with a completed run is skipped, and one whose trainer is still alive (e.g. started by an earlier
#     launcher) is never launched twice;
#   * up to 3 passes over the queue: a failed run (e.g. the intermittent cuInit failure on aims) has its run directory
#     renamed *_FAILED and is retried in the next pass. The final line lists anything still not completed.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
cd "$ROOT"
SPECS=("$@"); [ "${#SPECS[@]}" -gt 0 ] || { echo "usage: $0 SPEC [SPEC ...]"; exit 2; }
read -r -a GPUS <<< "${CLVLA_TRAIN_GPUS:-0 1}"
FRAC="${CLVLA_MEM_FRACTION:-0.92}"
BUFFER_MIB="${CLVLA_VRAM_BUFFER_MIB:-0}"
XLA_FLAGS_TRAIN="${CLVLA_TRAIN_XLA_FLAGS---xla_gpu_autotune_level=0}"
IDX=(78 84 16 56 35 42 28 3 89 63)
PASSES=3
log() { echo "[training $(date -u +%H:%M:%SZ)] $*"; }

spec_name() { python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['name'])" "$1"; }
spec_profile() { python3 -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1])).get('data_split_profile') or '')" "$1"; }
task_name() { python3 -c "import json,sys; print(next(r['name'] for r in json.load(open(sys.argv[1]))['libero_90'] if r['idx']==int(sys.argv[2])))" configs/libero90_tasks.json "$1"; }
done_run() {  # done_run <spec name> <idx>: a completed run exists
  for s in outputs/specialists/*_${1}-t$(printf %03d "$2")/summary.json; do
    [ -f "$s" ] && python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status')=='completed' else 1)" "$s" && return 0
  done
  return 1
}
running_run() {  # running_run <spec path> <idx>: a trainer for this spec+task is alive
  pgrep -u "$(whoami)" -f "train_specialists.py --spec $1 --indices $2 --execute" >/dev/null
}
set_aside_failed() {  # set_aside_failed <spec name> <idx>: rename unfinished/failed run dirs so they are never confused
  for d in outputs/specialists/*_${1}-t$(printf %03d "$2"); do
    [ -d "$d" ] || continue
    if [ ! -f "$d/summary.json" ] || ! python3 -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get('status')=='completed' else 1)" "$d/summary.json"; then
      mv "$d" "${d}_FAILED" && log "t$2 $1: set aside $(basename "$d") -> _FAILED"
    fi
  done
}
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$1"; }
total_mib() { nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$1"; }

declare -A GPU_PID GPU_JOB
reap() {
  for g in "${!GPU_PID[@]}"; do
    p=${GPU_PID[$g]}
    if ! kill -0 "$p" 2>/dev/null; then
      wait "$p"; rc=$?
      read -r j_idx j_name <<< "${GPU_JOB[$g]}"
      log "t$j_idx $j_name (GPU $g): finished rc=$rc"
      [ "$rc" = 0 ] || set_aside_failed "$j_name" "$j_idx"
      unset "GPU_PID[$g]" "GPU_JOB[$g]"
    fi
  done
}
pick_gpu() {  # a GPU without our run, accepted by the guard, with enough free memory
  for g in "${GPUS[@]}"; do
    [ -n "${GPU_PID[$g]:-}" ] && continue
    "$ROOT/scripts/gpu_guard.sh" "$g" >/dev/null 2>&1 || continue
    need=$(python3 -c "print(int(float('$FRAC') * $(total_mib "$g")) + $BUFFER_MIB)")
    [ "$(free_mib "$g")" -ge "$need" ] || continue
    echo "$g"; return 0
  done
  return 1
}

log "start: specs ${SPECS[*]}; GPUs ${GPUS[*]}; mem fraction $FRAC; buffer ${BUFFER_MIB} MiB; XLA_FLAGS '$XLA_FLAGS_TRAIN'; W&B key ${WANDB_API_KEY:+present}"
for pass in $(seq 1 $PASSES); do
  launched=0
  for idx in "${IDX[@]}"; do
    for SPEC in "${SPECS[@]}"; do
      NAME=$(spec_name "$SPEC"); PROFILE=$(spec_profile "$SPEC")
      done_run "$NAME" "$idx" && continue
      if running_run "$SPEC" "$idx"; then log "t$idx $NAME: already training elsewhere (pass $pass), skipping"; continue; fi
      lrm="$CLVLA_PROCESSED_ROOT/libero90_aligned256_v1/lerobot/$(task_name "$idx")${PROFILE:+.$PROFILE}.lerobot.json"
      [ -f "$lrm" ] || { log "t$idx $NAME: dataset $lrm missing, skipping"; continue; }
      while :; do reap; g=$(pick_gpu) && break; sleep 30; done
      set_aside_failed "$NAME" "$idx"
      tl="logs/train_${NAME}_t$(printf %03d "$idx")_$(date -u +%Y%m%dT%H%M%SZ).log"
      CLVLA_GPU="$g" CLVLA_USE_GPU=1 XLA_FLAGS="$XLA_FLAGS_TRAIN" CLVLA_MAX_CONCURRENT_TRAINING="${#GPUS[@]}" \
        XLA_PYTHON_CLIENT_MEM_FRACTION="$FRAC" \
        nice -n 10 "$ROOT/scripts/py-policy" scripts/train_specialists.py --spec "$SPEC" --indices "$idx" --execute \
        > "$tl" 2>&1 < /dev/null &
      GPU_PID[$g]=$!; GPU_JOB[$g]="$idx $NAME"; launched=$((launched + 1))
      log "t$idx $NAME: launched on GPU $g (pid $!, pass $pass) -> $tl"
      sleep 60   # let it claim its GPU before picking the next one
    done
  done
  while [ "${#GPU_PID[@]}" -gt 0 ]; do reap; sleep 30; done
  # wait for trainers started by another launcher before deciding whether anything is left
  while pgrep -u "$(whoami)" -f "train_specialists.py --spec" >/dev/null; do sleep 30; done
  [ "$launched" = 0 ] && break
done
missing=()
for idx in "${IDX[@]}"; do for SPEC in "${SPECS[@]}"; do
  NAME=$(spec_name "$SPEC"); done_run "$NAME" "$idx" || missing+=("t$idx:$NAME")
done; done
if [ "${#missing[@]}" -gt 0 ]; then log "DONE with runs not completed: ${missing[*]}"; exit 1; fi
log "ALL DONE"
