#!/usr/bin/env bash
# Container entrypoint: writes the runtime .env (all data on the job's storage volume), links outputs/ to a work
# directory on that volume, then runs the command it is given, e.g.
#   /opt/vla/aws/entrypoint.sh bash aws/clip_pilot_job.sh
set -euo pipefail
cd /opt/vla
DATA="${VLA_DATA:-/opt/ml/input/data/vla}"
# outputs/ is NOT the SageMaker checkpoint directory: SageMaker puts temporary *.sagemaker-uploading files into the
# directories it syncs, which breaks code that lists a directory it has just written (the buffer writer, the
# trainer's state save). The job script mirrors the small result files there instead (aws/clip_pilot_job.sh).
WORK="${VLA_WORK:-/opt/ml/input/data/work}"
mkdir -p "$DATA" "$WORK/outputs" /opt/ml/checkpoints/outputs /opt/ml/model
cat > .env <<ENV
OPENPI_DATA_HOME=$DATA/openpi
LIBERO_DATA_ROOT=$DATA/libero
CLVLA_OUTPUT_ROOT=$WORK/outputs
CLVLA_PROCESSED_ROOT=$DATA/processed
HF_LEROBOT_HOME=$DATA/lerobot
UV_CACHE_DIR=$DATA/uv
UV_PYTHON_INSTALL_DIR=/opt/uv-python
HF_HOME=$DATA/hf
CLVLA_GPU=0
CLVLA_SERVER_PORT=8000
ENV
export VLA_DATA="$DATA"
# outputs/ must be at the repository root; never replace a non-empty directory.
if [ -d outputs ] && [ ! -L outputs ]; then rmdir outputs || { echo "non-empty outputs/ in the image; refusing" >&2; exit 2; }; fi
[ -e outputs ] || ln -s "$WORK/outputs" outputs
if [ "${1:-}" = train ]; then shift; fi   # SageMaker passes "train" when no explicit arguments are set
[ "$#" -gt 0 ] || { echo "pass the command to run, e.g. bash aws/clip_pilot_job.sh" >&2; exit 2; }
exec "$@"
