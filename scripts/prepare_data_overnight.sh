#!/usr/bin/env bash
# Serial, detached data preparation for the 10 manifest tasks (protocol v1.0 order):
#   per task: re-render (pre-action aligned, 256 px)  ->  LeRobot train/val conversion;   finally: seal the version manifest
#   and verify every LeRobot dataset. CLVLA_SPLIT_PROFILE=n25 converts only the 25/25 profile.
# Resumable: tasks whose outputs already exist are skipped. Stops at the first failure. Never overwrites: a leftover
# *.partial file or a half-written LeRobot directory from a crash must be inspected and removed by hand before a rerun.
#   setsid nohup bash scripts/prepare_data_overnight.sh > logs/prepare_data_<stamp>.log 2>&1 < /dev/null &
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source "$ROOT/env.sh"
cd "$ROOT"
VERSION=libero90_aligned256_v1
IDX=(78 84 16 56 35 42 28 3 89 63)
PROFILE="${CLVLA_SPLIT_PROFILE:-}"   # e.g. n25: convert ONLY that split profile (default: the protocol 0-44 / 45-49 split)
SUFFIX="${PROFILE:+.$PROFILE}"
PROFILE_ARGS=(); [ -n "$PROFILE" ] && PROFILE_ARGS=(--split-profile "$PROFILE")
log() { echo "[prepare_data $(date -u +%H:%M:%SZ)] $*"; }
trap 'log "FAILED (exit $?) near line $LINENO; fix the cause, remove any leftover *.partial / half-written dataset for the failing task, then rerun this script (finished tasks are skipped)"' ERR

# EGL rendering uses only GPU $CLVLA_GPU (small footprint); refuse to start if another user has compute jobs on it.
"$ROOT/scripts/gpu_guard.sh"

name_of() {
  "$CLVLA_POLICY_PY" -I -c 'import json,sys; print(next(r["name"] for r in json.load(open(sys.argv[1]))["libero_90"] if r["idx"]==int(sys.argv[2])))' \
    "$ROOT/configs/libero90_tasks.json" "$1"
}

log "start: ${#IDX[@]} tasks, version $VERSION, processed root $CLVLA_PROCESSED_ROOT, lerobot home $HF_LEROBOT_HOME"
for idx in "${IDX[@]}"; do
  name="$(name_of "$idx")"
  report="$CLVLA_PROCESSED_ROOT/$VERSION/hdf5/$name.report.json"
  lrm="$CLVLA_PROCESSED_ROOT/$VERSION/lerobot/$name$SUFFIX.lerobot.json"
  if [ -f "$report" ]; then
    log "t$idx: re-render already done, skipping"
  else
    log "t$idx: re-rendering"
    nice -n 10 "$ROOT/scripts/py-sim" scripts/rerender_demos.py --indices "$idx" --execute
  fi
  if [ -f "$lrm" ]; then
    log "t$idx: conversion already done, skipping"
  else
    log "t$idx: converting to LeRobot (split ${PROFILE:-protocol 0-44 / 45-49})"
    nice -n 10 "$ROOT/scripts/py-policy" scripts/convert_to_lerobot.py --indices "$idx" "${PROFILE_ARGS[@]}" --execute
  fi
  log "t$idx: done"
done
log "sealing version manifest"
nice -n 10 "$ROOT/scripts/py-sim" scripts/rerender_demos.py --finalize
log "verifying LeRobot datasets (${PROFILE:-protocol})"
nice -n 10 "$ROOT/scripts/py-policy" scripts/verify_lerobot_datasets.py "${PROFILE_ARGS[@]/--split-profile/--profile}"
log "ALL DONE"
