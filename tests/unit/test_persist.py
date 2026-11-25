"""scripts/cl/persist.py: persisting a run while it trains and rebuilding it elsewhere. Standard library only, CPU."""

from __future__ import annotations

import importlib.util
import json
import os
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("persist", ROOT / "scripts" / "cl" / "persist.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)

RUN = "clip3_seq_20261125T000000Z"


def write_state(student: Path, update: int, payload: bytes = b"m") -> None:
    st = student / "state"
    st.mkdir(parents=True, exist_ok=True)
    (st / "master.npz").write_bytes(payload * 1000 + bytes([update % 256]))
    (st / "opt_state.npz").write_bytes(b"o" * 2000 + bytes([update % 256]))
    (st / "state.json").write_text(json.dumps({"update": update, "master_sha": f"x{update}"}))


@pytest.fixture
def run(tmp_path):
    out = tmp_path / "outputs" / "cl_opd"
    r = out / RUN
    finished = tmp_path / "elsewhere" / "k0_t035"
    finished.mkdir(parents=True)
    (finished / "DONE").write_text("ok\n")
    r.mkdir(parents=True)
    (r / "k0_t035").symlink_to(finished)
    (r / "config.json").write_text("{}")
    (r / "buffer").mkdir()
    (r / "buffer" / "t035.npy").write_bytes(b"b" * 100)
    s = r / "k1_t063" / "student"
    (s / "checkpoints").mkdir(parents=True)
    (s / "checkpoints" / "adapter_100.npz").write_bytes(b"c" * 500)
    (s / "batches").mkdir()
    (s / "batches" / "u001.npz").write_bytes(b"x")
    (s / "training_metrics.csv").write_text("update\n1\n")
    (s / "state.tmp").mkdir()
    (s / "state.tmp" / "junk").write_text("partial")
    (s / "metrics.json.sagemaker-uploading").write_text("")
    write_state(s, 10)
    (out / "LATEST_CLIP3_SEQ").write_text(f"outputs/cl_opd/{RUN}\n")
    return out


def snap(out, dest, **kw):
    P.snapshot(out, RUN, dest, "LATEST_CLIP3_SEQ", kw.get("min_age", 0.0), log=lambda m: None)


def test_snapshot_layout(run, tmp_path):
    dest = tmp_path / "ckpt"
    snap(run, dest)
    c = dest / "cl_opd" / RUN
    assert (c / "config.json").exists() and (c / "buffer" / "t035.npy").exists()
    assert (c / "k1_t063" / "student" / "checkpoints" / "adapter_100.npz").exists()
    assert not (c / "k0_t035").exists(), "the linked task 0 is fetched separately"
    assert not (c / "k1_t063" / "student" / "batches").exists()
    assert not (c / "k1_t063" / "student" / "state").exists() and not (c / "k1_t063" / "student" / "state.tmp").exists()
    assert not list(c.rglob("*.sagemaker-*"))
    info = P.slot_info(c / "k1_t063" / "student" / "state_slots" / "a")
    assert info["update"] == 10 and set(info["files"]) == {"master.npz", "opt_state.npz", "state.json"}
    assert (dest / "cl_opd" / "LATEST_CLIP3_SEQ").read_text() == f"outputs/cl_opd/{RUN}\n"
    assert json.loads((dest / "cl_opd" / f"PERSIST_{RUN}.json").read_text())["tasks"]["k1_t063"]["state_updates"] == {"a": [10, "state"]}


def test_slots_alternate_and_skip_repeats(run, tmp_path):
    dest = tmp_path / "ckpt"
    s = run / RUN / "k1_t063" / "student"
    slots = dest / "cl_opd" / RUN / "k1_t063" / "student" / "state_slots"
    snap(run, dest)
    snap(run, dest)  # same state: nothing rewritten
    assert P.slot_info(slots / "b") is None
    write_state(s, 20)
    snap(run, dest)
    write_state(s, 30)
    snap(run, dest)
    assert P.slot_info(slots / "a")["update"] == 30 and P.slot_info(slots / "b")["update"] == 20


def test_big_files_wait_for_min_age(run, tmp_path, monkeypatch):
    monkeypatch.setattr(P, "BIG", 100)
    dest = tmp_path / "ckpt"
    ck = run / RUN / "k1_t063" / "student" / "checkpoints" / "adapter_100.npz"
    snap(run, dest, min_age=3600)
    assert not (dest / "cl_opd" / RUN / "k1_t063" / "student" / "checkpoints" / "adapter_100.npz").exists()
    old = time.time() - 7200
    os.utime(ck, (old, old))
    snap(run, dest, min_age=3600)
    assert (dest / "cl_opd" / RUN / "k1_t063" / "student" / "checkpoints" / "adapter_100.npz").read_bytes() == ck.read_bytes()


def test_state_changing_during_copy_leaves_slot_invalid(run, tmp_path, monkeypatch):
    dest = tmp_path / "ckpt"
    s = run / RUN / "k1_t063" / "student"
    real = P.copy

    def racing(src, dst):
        real(src, dst)
        if src.name == "master.npz":
            write_state(s, 20)  # the trainer swaps in a new state mid-copy

    monkeypatch.setattr(P, "copy", racing)
    snap(run, dest)
    assert P.slot_info(dest / "cl_opd" / RUN / "k1_t063" / "student" / "state_slots" / "a") is None


def test_restore_rebuilds_and_resumes_newest_valid_slot(run, tmp_path):
    dest = tmp_path / "ckpt"
    s = run / RUN / "k1_t063" / "student"
    snap(run, dest)
    write_state(s, 20, payload=b"n")
    snap(run, dest)
    new = tmp_path / "xulab" / "outputs" / "cl_opd"
    P.restore(dest, new, log=lambda m: None)
    st = new / RUN / "k1_t063" / "student" / "state"
    assert json.loads((st / "state.json").read_text())["update"] == 20
    assert (st / "master.npz").read_bytes() == (s / "state" / "master.npz").read_bytes()
    assert (new / RUN / "buffer" / "t035.npy").exists() and (new / "LATEST_CLIP3_SEQ").exists()
    assert not (new / RUN / "k1_t063" / "student" / "state_slots").exists()
    # a corrupted newest slot falls back to the older one
    slots = dest / "cl_opd" / RUN / "k1_t063" / "student" / "state_slots"
    newest = max("ab", key=lambda x: P.slot_info(slots / x)["update"])
    (slots / newest / "master.npz").write_bytes(b"z" * 1001)  # same size, wrong content
    new2 = tmp_path / "other" / "cl_opd"
    P.restore(dest, new2, log=lambda m: None)
    assert json.loads((new2 / RUN / "k1_t063" / "student" / "state" / "state.json").read_text())["update"] == 10


def test_restore_keeps_newer_local_state(run, tmp_path):
    dest = tmp_path / "ckpt"
    snap(run, dest)
    P.restore(dest, run, log=lambda m: None)  # local state is at 10 already: untouched
    assert json.loads((run / RUN / "k1_t063" / "student" / "state" / "state.json").read_text())["update"] == 10


def test_finished_task_keeps_only_final_slot_and_is_not_restored(run, tmp_path):
    dest = tmp_path / "ckpt"
    s = run / RUN / "k1_t063" / "student"
    slots = dest / "cl_opd" / RUN / "k1_t063" / "student" / "state_slots"
    snap(run, dest)
    write_state(s, 400)
    snap(run, dest)
    (s / "state").rename(s / "state_final")  # cl_opd_run.py with CL_KEEP_FINAL_STATE=1
    (s.parent / "TRAINED").write_text("{}")
    snap(run, dest)
    kept = [x for x in "ab" if P.slot_info(slots / x) is not None]
    assert len(kept) == 1 and P.slot_info(slots / kept[0])["update"] == 400 and not (slots / ("b" if kept == ["a"] else "a")).exists()
    new = tmp_path / "x" / "cl_opd"
    P.restore(dest, new, log=lambda m: None)
    assert (new / RUN / "k1_t063" / "TRAINED").exists() and not (new / RUN / "k1_t063" / "student" / "state").exists()
