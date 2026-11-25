"""Copy a continual-learning run to a persistent directory while it trains, and rebuild a run directory from that copy.
Standard library only, so it runs anywhere (the AWS job, xulab, a laptop).

  python3 scripts/cl/persist.py snapshot --outputs outputs/cl_opd --run NAME --to DEST [--pointer LATEST_X] [--min-age 120]
  python3 scripts/cl/persist.py restore  --from DEST --outputs outputs/cl_opd
  python3 scripts/cl/persist.py status   --from DEST

Layout of DEST (the SageMaker checkpoint directory on AWS, which SageMaker syncs to S3; any directory elsewhere):
  DEST/cl_opd/<pointer file>                    e.g. LATEST_CLIP3_SEQ, the run's name
  DEST/cl_opd/<run>/...                         every file of the run, at the same relative path, except:
      symlinks (task 0 links to a finished run that is fetched separately), */student/batches/ (diagnostic batches),
      the trainer's live state directories (state, state.tmp, state.old, state_final) and SageMaker's own temp files;
  DEST/cl_opd/<run>/<kK_task>/student/state_slots/{a,b}/
      the full training state (master, optimizer, replay, old-task sampler, ...) of that task, in two alternating slots.
      A slot is valid only with its COMPLETE.json (the state.json it copied, and the size and sha256 of every file),
      which is written last and removed first, so an interrupted copy or upload never replaces the previous good one.
      The state saved at the end of a task (state_final, kept by CL_KEEP_FINAL_STATE=1) goes to a slot the same way;
      once a task is TRAINED (and its live state is gone) only its newest slot is kept.

Large files (over 64 MB) are copied only when they have not changed for --min-age seconds, so a checkpoint that is
still being written is not copied half-way; later snapshots copy any file whose size or mtime differs.
restore copies every file back, then puts the newest valid slot of each unfinished task back as student/state, so
scripts/cl_opd_run.py resumes that task from the saved update (it continues any task whose state.json exists).
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

LIVE_STATE = ("state", "state.tmp", "state.old", "state_final")
SKIP_DIRS = {"batches"}
SLOTS = ("a", "b")
BIG = 64 << 20


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sagemaker_temp(name: str) -> bool:
    return ".sagemaker-" in name


def same(a: Path, b: Path) -> bool:
    try:
        sa, sb = a.stat(), b.stat()
    except FileNotFoundError:
        return False
    return sa.st_size == sb.st_size and int(sa.st_mtime) == int(sb.st_mtime)


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)  # in place: a rename would show up as a delete and a new file in the synced directory


def run_files(run: Path):
    """Files of a run to persist: (path relative to the run, absolute path)."""
    for d, dirs, files in os.walk(run):
        d = Path(d)
        keep = []
        for x in dirs:
            p = d / x
            if p.is_symlink() or x in SKIP_DIRS or x == "state_slots":
                continue
            if d.name == "student" and x in LIVE_STATE:
                continue
            keep.append(x)
        dirs[:] = keep
        for f in files:
            p = d / f
            if p.is_symlink() or sagemaker_temp(f):
                continue
            yield p.relative_to(run), p


def slot_info(slot: Path) -> dict | None:
    f = slot / "COMPLETE.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (OSError, ValueError):
        return None


def slot_valid(slot: Path, info: dict, check_sha: bool) -> bool:
    for name, (size, digest) in info["files"].items():
        p = slot / name
        if not p.exists() or p.stat().st_size != size or (check_sha and sha256(p) != digest):
            return False
    return True


def snapshot_state(src: Path, slots: Path, log) -> None:
    """Copy one live state directory (student/state or student/state_final) into the older slot."""
    sj = src / "state.json"
    try:
        before = sj.read_text()
        update = json.loads(before)["update"]
    except (OSError, ValueError, KeyError):
        return  # being swapped by the trainer right now: the next snapshot gets it
    infos = {s: slot_info(slots / s) for s in SLOTS}
    if any(i is not None and i["state"] == before for i in infos.values()):
        return  # this state is already in a slot
    # the slot to overwrite: an empty one, else the one with the older update
    target = min(SLOTS, key=lambda s: -1 if infos[s] is None else infos[s]["update"])
    dst = slots / target
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "COMPLETE.json").unlink(missing_ok=True)
    files = {}
    try:
        names = sorted(p.name for p in src.iterdir() if p.is_file())
        for name in names:
            copy(src / name, dst / name)
            files[name] = [(dst / name).stat().st_size, sha256(dst / name)]
        after = sj.read_text()
    except OSError as e:
        log(f"[persist] state copy of {src} interrupted ({type(e).__name__}); slot {target} left invalid")
        return
    if after != before:
        log(f"[persist] {src} changed during the copy; slot {target} left invalid, next snapshot retries")
        return
    for stale in [p for p in dst.iterdir() if p.name not in files and p.name != "COMPLETE.json"]:
        stale.unlink()
    (dst / "COMPLETE.json").write_text(json.dumps({"update": update, "state": before, "source": src.name, "files": files, "time": time.time()}))
    log(f"[persist] {src.parent.parent.name} state at update {update} -> slot {target}")


def snapshot(outputs: Path, run_name: str, dest: Path, pointer: str | None, min_age: float, log=print) -> None:
    outputs.mkdir(parents=True, exist_ok=True)
    with open(outputs / ".persist.lock", "w") as lock:  # one snapshot at a time (the periodic loop and the per-task call)
        fcntl.flock(lock, fcntl.LOCK_EX)
        _snapshot(outputs, run_name, dest, pointer, min_age, log)


def _snapshot(outputs: Path, run_name: str, dest: Path, pointer: str | None, min_age: float, log) -> None:
    run = outputs / run_name
    out = dest / "cl_opd" / run_name
    now = time.time()
    n = 0
    for rel, p in run_files(run):
        try:
            st = p.stat()
        except FileNotFoundError:
            continue
        if st.st_size > BIG and now - st.st_mtime < min_age:
            continue
        q = out / rel
        if not same(p, q):
            try:
                copy(p, q)
                n += 1
            except FileNotFoundError:
                continue
    for student in run.glob("k*_*/student"):
        if student.parent.is_symlink():
            continue
        slots = out / student.parent.name / "student" / "state_slots"
        for live in ("state", "state_final"):
            if (student / live).is_dir():
                snapshot_state(student / live, slots, log)
        if (student.parent / "TRAINED").exists() and not (student / "state").is_dir():
            infos = {s: slot_info(slots / s) for s in SLOTS}
            valid = [s for s in SLOTS if infos[s] is not None]
            if len(valid) == 2:  # the task is finished: keep only its newest (final) state
                shutil.rmtree(slots / min(valid, key=lambda s: infos[s]["update"]))
    if pointer and (outputs / pointer).exists():
        if not same(outputs / pointer, dest / "cl_opd" / pointer):
            copy(outputs / pointer, dest / "cl_opd" / pointer)
    (dest / "cl_opd" / f"PERSIST_{run_name}.json").write_text(json.dumps({"time": now, "run": run_name, "copied": n, "tasks": status(dest / "cl_opd" / run_name)}, indent=1))


def status(run_copy: Path) -> dict:
    s = {}
    for k in sorted(run_copy.glob("k*_*")):
        slots = {x: slot_info(k / "student" / "state_slots" / x) for x in SLOTS}
        s[k.name] = {"done": (k / "DONE").exists(), "trained": (k / "TRAINED").exists(),
                     "state_updates": {x: (i["update"], i["source"]) for x, i in slots.items() if i is not None},
                     "checkpoints": sorted(p.name for p in (k / "student" / "checkpoints").glob("*.npz"))}
    return s


def restore(src_root: Path, outputs: Path, log=print) -> None:
    """Rebuild runs under outputs/ from DEST/cl_opd. Existing files that differ are overwritten only when the persisted
    copy is newer; the state of an unfinished task is restored only when it is newer than the local one."""
    base = src_root / "cl_opd"
    assert base.is_dir(), f"nothing persisted under {base}"
    outputs.mkdir(parents=True, exist_ok=True)
    for d, dirs, files in os.walk(base):
        d = Path(d)
        dirs[:] = [x for x in dirs if x != "state_slots"]
        for f in files:
            if sagemaker_temp(f) or f.startswith(("PERSIST_", ".persist")):
                continue
            p = d / f
            q = outputs / p.relative_to(base)
            if q.exists() and (same(p, q) or q.stat().st_mtime > p.stat().st_mtime):
                continue
            copy(p, q)
    for slots in base.glob("*/k*_*/student/state_slots"):
        student = outputs / slots.relative_to(base).parent
        if (student.parent / "TRAINED").exists():
            continue  # finished task: the driver does not need its state (it stays in the slots)
        cands = sorted(((i["update"], s) for s in SLOTS if (i := slot_info(slots / s)) is not None), reverse=True)
        local = student / "state" / "state.json"
        have = json.loads(local.read_text())["update"] if local.exists() else -1
        for update, s in cands:
            if update <= have:
                break
            info = slot_info(slots / s)
            if not slot_valid(slots / s, info, check_sha=True):
                log(f"[restore] {slots / s} fails its checksums; trying the other slot")
                continue
            tmp = student / "state.restore"
            shutil.rmtree(tmp, ignore_errors=True)
            tmp.mkdir(parents=True)
            for name in info["files"]:
                shutil.copy2(slots / s / name, tmp / name)
            shutil.rmtree(student / "state", ignore_errors=True)
            tmp.rename(student / "state")
            log(f"[restore] {student.parent.name}: state at update {update} (slot {s})")
            break


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("snapshot")
    a.add_argument("--outputs", required=True, type=Path, help="the outputs/cl_opd directory")
    a.add_argument("--run", required=True, help="run directory name under --outputs")
    a.add_argument("--to", required=True, type=Path)
    a.add_argument("--pointer", help="pointer file name under --outputs to copy too")
    a.add_argument("--min-age", type=float, default=120.0)
    a.add_argument("--loop", type=float, default=0.0, help="repeat every N seconds (0: once)")
    r = sub.add_parser("restore")
    r.add_argument("--from", dest="src", required=True, type=Path)
    r.add_argument("--outputs", required=True, type=Path, help="the outputs directory (cl_opd/ is created inside)")
    s = sub.add_parser("status")
    s.add_argument("--from", dest="src", required=True, type=Path)
    x = ap.parse_args()
    if x.cmd == "snapshot":
        while True:
            try:
                snapshot(x.outputs, x.run, x.to, x.pointer, x.min_age)
            except Exception as e:  # noqa: BLE001  a failed pass must not stop the loop; the next one retries
                print(f"[persist] snapshot failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
                if not x.loop:
                    raise
            if not x.loop:
                break
            time.sleep(x.loop)
    elif x.cmd == "restore":
        restore(x.src, x.outputs / "cl_opd")
    else:
        for run in sorted(p for p in (x.src / "cl_opd").iterdir() if p.is_dir()):
            print(run.name, json.dumps(status(run), indent=1))


if __name__ == "__main__":
    main()
