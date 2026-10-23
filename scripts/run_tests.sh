#!/usr/bin/env bash
# Staged test runner. Each stage runs in the environment it needs; a summary table is printed at the end.
#   bash scripts/run_tests.sh cpu   # unit tests (both envs) + environment imports + real-data preprocessing; no GPU
#   bash scripts/run_tests.sh sim   # cpu + simulator stage (EGL rendering on CLVLA_GPU)
#   bash scripts/run_tests.sh all   # sim + checkpoint inference + policy server + end-to-end episode (GPU)
# Logs: logs/tests/<stamp>/<stage>.log
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LEVEL="${1:-cpu}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"; LOGDIR="$ROOT/logs/tests/$STAMP"; mkdir -p "$LOGDIR"
# hydra-core in OpenPI's pinned LIBERO requirements lacks omegaconf; its auto-loaded pytest plugin would crash the sim env.
SIMPY=(env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 "$ROOT/scripts/py-sim" -m pytest -p no:cacheprovider -rA)
POLPY=("$ROOT/scripts/py-policy" -m pytest -p no:cacheprovider -rA)
declare -a NAMES RESULTS

stage() {  # stage <name> <cmd...>
  local name="$1"; shift
  echo "=== [$name] $*"
  ( cd "$ROOT" && "$@" ) > "$LOGDIR/$name.log" 2>&1
  local rc=$?
  local line; line=$(grep -E "^(=+ )?[0-9]+ (passed|failed|skipped|error)|no tests ran" "$LOGDIR/$name.log" | tail -1)
  NAMES+=("$name"); RESULTS+=("rc=$rc  ${line//=/}")
  echo "    rc=$rc  ${line//=/}"
}

stage unit-policy-env      "${POLPY[@]}" tests/unit
stage unit-sim-env         "${SIMPY[@]}" tests/unit
stage env-policy           "${POLPY[@]}" tests/integration/test_environment.py
stage env-sim              "${SIMPY[@]}" tests/integration/test_environment.py
stage data-real            "${POLPY[@]}" -s tests/integration/test_data_real.py
if [ "$LEVEL" = sim ] || [ "$LEVEL" = all ]; then
  stage simulator          "${SIMPY[@]}" -s tests/integration/test_simulator.py
fi
if [ "$LEVEL" = all ]; then
  if "$ROOT/scripts/gpu_guard.sh"; then
    stage checkpoint-gpu   env CLVLA_USE_GPU=1 "${POLPY[@]}" -s tests/integration/test_checkpoint.py tests/integration/test_environment.py
    stage server-e2e-gpu   "${SIMPY[@]}" -s tests/integration/test_server.py tests/integration/test_end_to_end.py
  else
    NAMES+=("gpu-stages"); RESULTS+=("NOT RUN: GPU busy (see gpu_guard output)")
  fi
fi
echo; echo "=== summary ($LEVEL) — logs in $LOGDIR"
fail=0
for i in "${!NAMES[@]}"; do printf "  %-16s %s\n" "${NAMES[$i]}" "${RESULTS[$i]}"; [[ "${RESULTS[$i]}" == rc=0* ]] || fail=1; done
exit $fail
