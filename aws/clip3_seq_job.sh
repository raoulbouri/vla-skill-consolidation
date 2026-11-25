#!/usr/bin/env bash
# SageMaker job: the clipped curriculum (scripts/cl/clip3_sequence.sh, configs/cl_opd_clip3.yaml) on one GPU, run through
# aws/entrypoint.sh. Built for managed spot and for running out of time: everything needed to continue is persisted to
# the SageMaker checkpoint directory (synced to S3) while it trains, and a job started with the same checkpoint S3 URI
# (a spot restart, or a new job with the same --run-id) restores it and continues where the last one stopped.
# 1. checks: GPU, host RAM, disk, MuJoCo rendering through the NVIDIA EGL driver;
# 2. inputs, all public: base pi0.5 (OpenPI, GCS), the five teachers and the sequence inputs (task-0 result of
#    outputs/cl_opd/full_20261108T214604Z, demonstrations of the five tasks) from Hugging Face, sha256-checked;
# 3. restore from /opt/ml/checkpoints/cl_opd if a previous job persisted there (scripts/cl/persist.py restore);
# 4. the sequence, with scripts/cl/persist.py snapshot every 2 minutes and after every stage: all run files except the
#    diagnostic batches, and the full training state (master, optimizer, replay, sampler) of each task in two
#    alternating checksummed slots, refreshed every 10 updates. On SIGTERM (spot interruption, MaxRuntime) the trainer
#    saves its state, then a last snapshot is taken; at worst the previous slot (10 updates earlier) is used.
# 5. /opt/ml/model/results.tar.gz: metrics, logs, evaluation records, the success matrix (no weights, no batches).
set -uo pipefail
cd /opt/vla
source scripts/cl/common.sh
REPO="${CLVLA_HF_REPO:-raoulb091602/vla-skill-consolidation}"
SUMS=checksums/vla-skill-consolidation_cl-clip3-seq-inputs_20261125.SHA256SUMS
CFG=configs/cl_opd_clip3.yaml
PTR=outputs/cl_opd/LATEST_CLIP3_SEQ
CK=/opt/ml/checkpoints
MON=/opt/ml/model/monitor.csv
step() { echo; echo "=== [$(date -u +%FT%TZ)] $*"; }
snapshot() {
  local run; run="$(cat "$PTR" 2>/dev/null)"
  [ -n "$run" ] && [ -d "$run" ] || return 0
  scripts/py-policy scripts/cl/persist.py snapshot --outputs outputs/cl_opd --run "$(basename "$run")" --to "$CK" --pointer LATEST_CLIP3_SEQ --min-age "${1:-120}" \
    || echo "[persist] snapshot failed"
}
pack() {
  local run; run="$(cat "$PTR" 2>/dev/null)"
  if [ -n "$run" ] && [ -d "$run" ]; then
    tar czf /opt/ml/model/results.tar.gz --exclude='*.npz' --exclude='batches' --exclude='teacher_episodes' \
      --exclude='buffer' --exclude='k0_t035' --exclude='state*' "$run" && echo "results: $(du -h /opt/ml/model/results.tar.gz | cut -f1)"
  fi
}
SEQ_PID=""
on_term() {   # spot interruption or MaxRuntime: let the trainer save its state, then persist it
  echo "=== [$(date -u +%FT%TZ)] SIGTERM: stopping the sequence"
  [ -n "$SEQ_PID" ] && kill -TERM -- "-$SEQ_PID" 2>/dev/null
  for _ in $(seq 75); do pgrep -f opd_lora_train.py >/dev/null || break; sleep 1; done
  snapshot 0; pack
  echo "=== [$(date -u +%FT%TZ)] persisted after SIGTERM"
  exit 143
}
trap on_term TERM

step "machine (clipped sequence)"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
free -g | head -2; nproc; df -h /opt/ml "$VLA_DATA" "$CK" | tail -n +1
( echo "time,host_used_gib,host_avail_gib,gpu_used_mib"
  while :; do echo "$(date -u +%T),$(free -g | awk '/Mem:/{print $3","$7}'),$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)"; sleep 30; done ) > "$MON" &

step "EGL rendering through the NVIDIA driver"
scripts/py-sim -c "
import mujoco
from OpenGL import GL
ctx = mujoco.GLContext(64, 64); ctx.make_current()
v = GL.glGetString(GL.GL_VENDOR).decode(); r = GL.glGetString(GL.GL_RENDERER).decode()
print('GL vendor:', v, '| renderer:', r)
assert 'NVIDIA' in v, 'MuJoCo is not rendering on the NVIDIA GPU'
" || { echo "EGL_CHECK_FAILED"; exit 4; }

step "base checkpoint (public OpenPI bucket)"
scripts/py-policy -c "from openpi.shared import download; print(download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero'))" || { echo "BASE_DOWNLOAD_FAILED"; exit 5; }

step "teachers of all five tasks (Hugging Face, sha256-checked)"
scripts/py-policy scripts/fetch_cl_assets.py --config "$CFG" || { echo "TEACHERS_FAILED"; exit 5; }

step "sequence inputs (Hugging Face, sha256-checked)"
scripts/py-policy -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('$REPO', local_dir=os.environ['VLA_DATA'], allow_patterns=['$SUMS', 'outputs/cl_opd/full_20261108T214604Z/k0_t035/*', 'processed/libero90_aligned256_v1/hdf5/*'])
" && (cd "$VLA_DATA" && sha256sum -c --quiet "$SUMS") || { echo "INPUTS_FAILED"; exit 5; }

step "restore from the checkpoint directory"
if [ -d "$CK/cl_opd" ]; then
  scripts/py-policy scripts/cl/persist.py status --from "$CK" | head -60
  scripts/py-policy scripts/cl/persist.py restore --from "$CK" --outputs outputs || { echo "RESTORE_FAILED"; exit 6; }
  echo "continuing $(cat "$PTR" 2>/dev/null)"
else
  echo "nothing persisted: a new run"
fi

step "clipped sequence"
( while :; do sleep 120; snapshot 120; done ) &
export CL_FROM_TASK0="$VLA_DATA/outputs/cl_opd/full_20261108T214604Z/k0_t035" CL_PERSIST_DIR="$CK"
setsid bash scripts/cl/clip3_sequence.sh &
SEQ_PID=$!
wait "$SEQ_PID"
rc=$?
snapshot 0; pack
step "sequence exit $rc"
exit $rc
