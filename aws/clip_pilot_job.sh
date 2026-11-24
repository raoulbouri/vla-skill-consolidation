#!/usr/bin/env bash
# SageMaker job: a t063 pilot on one GPU, run through aws/entrypoint.sh. The argument picks it:
#   clip      (default) the clipping pilot, 200 updates (scripts/cl/t063_clip_pilot.sh)
#   tr        the trust-region arm, 200 updates (the same launcher with configs/cl_opd_t063_tr3_u200.yaml)
#   continue  the clipping pilot resumed from its update-190 state and trained to 400 (scripts/cl/t063_clip_continue.sh);
#             needs the k1_t063 directory of the pilot run as the input channel "pilot" (/opt/ml/input/data/pilot/k1_t063)
# 1. checks: GPU, host RAM, disk, and that MuJoCo renders through the NVIDIA EGL driver (fails fast otherwise);
# 2. inputs, all public: base pi0.5 checkpoint (OpenPI, GCS), the t035 and t063 teachers and the pilot inputs (task-0
#    result of outputs/cl_opd/full_20261108T214604Z, demonstrations of t035/t063/t042) from Hugging Face, sha256-checked;
# 3. the pilot; host RAM and GPU memory are sampled every 30 s for sizing later jobs;
# 4. /opt/ml/model/results.tar.gz: metrics, logs, evaluation records, the success matrix (no weights, no batches).
set -uo pipefail
cd /opt/vla
MODE="${1:-clip}"
case "$MODE" in clip|tr|continue) ;; *) echo "unknown mode $MODE (clip, tr, continue)"; exit 2 ;; esac
source scripts/cl/common.sh
REPO="${CLVLA_HF_REPO:-raoulb091602/vla-skill-consolidation}"
SUMS=checksums/vla-skill-consolidation_cl-clip-pilot-inputs_20261121.SHA256SUMS
MON=/opt/ml/model/monitor.csv
step() { echo; echo "=== [$(date -u +%FT%TZ)] $*"; }
pack() {
  local run; run="$(cat outputs/cl_opd/LATEST_T063_PILOT 2>/dev/null)"
  if [ -n "$run" ] && [ -d "$run" ]; then
    tar czf /opt/ml/model/results.tar.gz --exclude='*.npz' --exclude='batches' --exclude='teacher_episodes' \
      --exclude='buffer' --exclude='k0_t035' --exclude='state*' "$run" && echo "results: $(du -h /opt/ml/model/results.tar.gz | cut -f1)"
  fi
}
trap pack EXIT

step "machine (mode $MODE)"
nvidia-smi --query-gpu=name,memory.total,memory.used,driver_version --format=csv
free -g | head -2; nproc; df -h /opt/ml "$VLA_DATA" | tail -n +1
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

step "teachers t035, t063 (Hugging Face, sha256-checked)"
scripts/py-policy scripts/fetch_cl_assets.py --config configs/cl_opd_t063_clip3_u200.yaml --tasks t035,t063 || { echo "TEACHERS_FAILED"; exit 5; }

step "pilot inputs (Hugging Face, sha256-checked)"
scripts/py-policy -c "
import os
from huggingface_hub import snapshot_download
snapshot_download('$REPO', local_dir=os.environ['VLA_DATA'], allow_patterns=['$SUMS', 'outputs/cl_opd/full_20261108T214604Z/k0_t035/*', 'processed/libero90_aligned256_v1/hdf5/*'])
" && (cd "$VLA_DATA" && sha256sum -c --quiet "$SUMS") || { echo "INPUTS_FAILED"; exit 5; }

step "pilot ($MODE)"
export CL_FROM_TASK0="$VLA_DATA/outputs/cl_opd/full_20261108T214604Z/k0_t035"
case "$MODE" in
  clip) bash scripts/cl/t063_clip_pilot.sh ;;
  tr) CL_CONFIG=configs/cl_opd_t063_tr3_u200.yaml CL_TAG=t063_tr3_u200 bash scripts/cl/t063_clip_pilot.sh ;;
  continue) CL_FROM_PILOT=/opt/ml/input/data/pilot/k1_t063 bash scripts/cl/t063_clip_continue.sh ;;
esac
rc=$?
step "pilot exit $rc"
exit $rc
