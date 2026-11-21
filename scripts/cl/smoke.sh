#!/usr/bin/env bash
# The whole continual-learning loop on all 5 tasks with configs/cl_opd_smoke.yaml (3 teacher-driven + 3 student-driven
# updates per task, 3 evaluation states per cell), one task after another. Run it once after setup, before a real run.
#   scripts/cl/smoke.sh            (GPU: GPUQ_GPU if set, else CLVLA_GPU from .env)
set -uo pipefail
source "$(dirname "$0")/common.sh"
RUN="outputs/cl_opd/smoke_$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN" && echo "$RUN" > outputs/cl_opd/LATEST_SMOKE
frozen_check START
for K in 0 1 2 3 4; do
  echo "=== task $K $(date -u +%FT%TZ)"
  scripts/py-policy scripts/cl_opd_run.py --config configs/cl_opd_smoke.yaml --run "$RUN" --task "$K" --gpu "$GPU" || { echo "TASK_${K}_FAILED"; exit 1; }
done
scripts/py-policy scripts/cl_opd_wandb.py --run "$RUN" --dry-run
du -sh "$RUN"; ls "$RUN"
frozen_check END
echo SMOKE_DONE "$RUN"
