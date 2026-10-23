#!/usr/bin/env bash
# Canonical checkpoint baseline, then an exact re-execution from its saved snapshots, then an episode-level comparison.
# Usage: setsid nohup bash scripts/baseline_and_repro.sh 10 > logs/baseline_and_repro_<stamp>.log 2>&1 < /dev/null &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
N="${1:-10}"
bash "$ROOT/scripts/baseline_eval.sh" --episodes-per-task "$N" --name pi05_libero_official_det || exit 1
RUN=$(ls -td "$ROOT"/outputs/baseline/*_pi05_libero_official_det | head -1)
bash "$ROOT/scripts/baseline_eval.sh" --from-run "$RUN" --kind smoke --name repro_of_$(basename "$RUN") || exit 1
REPRO=$(ls -td "$ROOT"/outputs/smoke/*_repro_of_* | head -1)
"$ROOT/scripts/py-policy" "$ROOT/scripts/compare_runs.py" "$RUN" "$REPRO" | tee "$REPRO/compare_with_original.txt"
