#!/usr/bin/env bash
# Export every indexed teacher (outputs/checkpoints/INDEX.json) as a delta against the pi05_libero base: the changed or added leaves plus BF16 sha256 of every leaf (scripts/opd_export_teacher_delta.py), with the run's train_spec.yaml and summary.json. CPU only, one teacher at a time, resumable (finished teachers are skipped).
# Usage: export_teacher_deltas.sh <out_root>   ->  <out_root>/<variant>/t<idx>_<task>/{delta.npz,leaf_sha.json,train_spec.yaml,summary.json}
set -uo pipefail
cd "$(dirname "$0")/.."
OUT="$1"
source env.sh
BASE="$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero/params"
mkdir -p "$OUT"
python3 -c "
import json
for e in json.load(open('outputs/checkpoints/INDEX.json')):
    print(e['variant'], 't%03d_%s' % (e['libero_90_index'], e['task']), e['run_dir'], e['checkpoint_dir'])
" | while read -r V NAME RUN CK; do
  D="$OUT/$V/$NAME"
  [ -s "$D/leaf_sha.json" ] && { echo "skip $V $NAME"; continue; }
  mkdir -p "$D"
  nice -n 10 scripts/py-policy -W ignore scripts/opd_export_teacher_delta.py "$CK/params" "$BASE" "$D" < /dev/null && cp "$RUN/train_spec.yaml" "$RUN/summary.json" "$D/" && echo "done $V $NAME $(du -sh "$D" | cut -f1)" || { echo "FAILED $V $NAME"; rm -f "$D/delta.npz" "$D/leaf_sha.json"; }
done
echo EXPORT_DONE
