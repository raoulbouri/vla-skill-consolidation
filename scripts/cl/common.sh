# Sourced by the scripts in scripts/cl/. Repository root, env.sh, the GPU, and the frozen-module check; no paths are
# hardcoded: everything is relative to the repository root or comes from .env.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$ROOT/env.sh"
cd "$ROOT"
GPU="${GPUQ_GPU:-$CLVLA_GPU}"   # gpuq (or any scheduler) may set GPUQ_GPU; otherwise CLVLA_GPU from .env
frozen_check() {   # the OGPO modules of Tasks 5-9 must be byte-identical to their recorded hashes
  sha256sum -c tests/ogpo_port/frozen_task5_9.sha256 --quiet || { echo "FROZEN_MODULES_CHANGED: restore them before running"; exit 3; }
  echo "FROZEN_OK_$1"
}
print_matrix() {
  "$CLVLA_POLICY_PY" -I -c 'import json, sys; m = json.load(open(sys.argv[1])); print("MATRIX", json.dumps({k: m.get(k) for k in ("S", "final_avg", "nbt_avg", "fwt")}))' "$1/matrix.json"
}
