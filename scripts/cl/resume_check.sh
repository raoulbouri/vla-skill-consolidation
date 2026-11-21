#!/usr/bin/env bash
# Exact-resume check on task 1 of the latest smoke run (old-task buffer + LoRA teachers): 4 updates straight vs. stop
# after 3 and resume; the two must agree bitwise. Run scripts/cl/smoke.sh first.
#   scripts/cl/resume_check.sh     (GPU: GPUQ_GPU if set, else CLVLA_GPU from .env)
set -uo pipefail
source "$(dirname "$0")/common.sh"
[ -s outputs/cl_opd/LATEST_SMOKE ] || { echo "no smoke run: run scripts/cl/smoke.sh first"; exit 2; }
RUN="$(cat outputs/cl_opd/LATEST_SMOKE)"
scripts/py-policy scripts/cl_opd_resume_check.py --config configs/cl_opd_smoke.yaml --run "$RUN" --task 1 --out "$RUN/resume_check" --gpu "$GPU"
