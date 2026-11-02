#!/usr/bin/env bash
# Stage 3: one teacher-only update on xulab GPU 1. Prints checkpoint-file hashes before and after.
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CLVLA_GPU=1 CLVLA_USE_GPU=1
source "$ROOT/env.sh"
cd "$ROOT"
scripts/gpu_guard.sh 1 || exit 3
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_START
B="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/params"
H0=$(find "$B" -type f -exec sha256sum {} + | sort -k2 | sha256sum | cut -c1-32); echo "BASE_PARAMS_FILES $H0"
STAMP=$(date -u +%Y%m%dT%H%M%SZ); OUT="outputs/opd_stage3/${STAMP}_t035"; mkdir -p "$OUT"; echo "$OUT"
export XLA_PYTHON_CLIENT_MEM_FRACTION=.95 PYTHONPATH="$ROOT/scripts:$ROOT/tests/ogpo_port"
scripts/py-policy -W ignore scripts/opd_stage3.py --stage2 outputs/opd_stage2/20261101T230405Z_t035_demo0 --teacher-delta outputs/opd_stage2/teacher_t035 --out "$OUT" > "$OUT/run.log" 2>&1 < /dev/null; rc=$?
echo "exit $rc"; grep -E '^\[(gate|routing|optimizer|update|evaluation|end|stop)\]' "$OUT/run.log" | cut -c1-1800; [ "$rc" = 0 ] || tail -25 "$OUT/run.log" | cut -c1-300
H1=$(find "$B" -type f -exec sha256sum {} + | sort -k2 | sha256sum | cut -c1-32); echo "BASE_PARAMS_FILES_AFTER $H1"
sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet && echo FROZEN_OK_END
