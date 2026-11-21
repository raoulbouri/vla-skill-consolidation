"""Continual-learning OPD helpers (clvla.cl_opd) and the driver's plan: NumPy only, no GPU, no simulator."""

from __future__ import annotations

import copy
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from clvla.cl_opd import buffer as B
from clvla.cl_opd import config as C
from clvla.cl_opd import matrix as MX
from clvla.cl_opd.phases import PHASES, label, label_by_time, within_phase_time
from clvla.cl_opd.sampler import OldTaskSampler, split

ROOT = Path(__file__).resolve().parents[2]


# phases

def test_label_rules_and_sticky_manipulation():
    dist = [0.5, 0.2, 0.1, 0.05, 0.05, 0.05, 0.3]
    con = [0, 0, 0, 1, 1, 0, 0]
    disp = [0, 0, 0, 0, 0.06, 0.01, 0.0]  # moved at step 4, then back below the threshold: stays manipulation
    assert label(dist, con, disp) == ["approach", "approach", "precontact", "contact", "manipulation", "manipulation", "manipulation"]


def test_time_fallback_and_within_phase_time():
    ph = label_by_time(8)
    assert ph == ["approach"] * 2 + ["precontact"] * 2 + ["contact"] * 2 + ["manipulation"] * 2
    t = within_phase_time(["a", "a", "a", "b", "c", "c"])
    assert t.tolist() == [0.0, 0.5, 1.0, 0.5, 0.0, 1.0]


# buffer ranking

def episodes(n_ep=6, seed=0):
    """Synthetic episodes with phases of very different lengths (contact is short), as in manipulation."""
    r = np.random.default_rng(seed)
    cands = []
    for e in range(n_ep):
        lens = {"approach": int(r.integers(30, 80)), "precontact": int(r.integers(5, 15)), "contact": int(r.integers(2, 6)), "manipulation": int(r.integers(10, 40))}
        ph = [p for p in PHASES for _ in range(lens[p])]
        t = within_phase_time(ph)
        cands += [{"episode": f"ep{e}", "step": i, "phase": p, "t": float(t[i])} for i, p in enumerate(ph)]
    return cands


def test_priority_order_prefix_is_phase_balanced():
    cands = episodes()
    order = B.priority_order(cands, seed=3)
    assert sorted(order) == list(range(len(cands))), "order must be a permutation"
    avail = {p: sum(c["phase"] == p for c in cands) for p in PHASES}
    for n in (4, 8, 20, 40, 100):
        cnt = {p: sum(cands[i]["phase"] == p for i in order[:n]) for p in PHASES}
        live = [p for p in PHASES if cnt[p] < avail[p]]  # phases not yet exhausted
        if live:
            assert max(cnt[p] for p in live) - min(cnt[p] for p in live) <= 1, (n, cnt)


def test_priority_order_spreads_over_episodes_and_keeps_gap():
    cands = episodes()
    order = B.priority_order(cands, seed=3, min_gap=2)
    first = order[:24]  # 6 per phase: every episode should appear within each phase's first 6 picks
    for p in PHASES:
        eps = {cands[i]["episode"] for i in first if cands[i]["phase"] == p}
        assert len(eps) == 6, (p, eps)
    # gap rule: with 40 steps per phase, each episode has 20 gap-respecting states, so its first 10 picks per phase obey it
    long = []
    for e in range(3):
        ph = [p for p in PHASES for _ in range(40)]
        t = within_phase_time(ph)
        long += [{"episode": f"ep{e}", "step": i, "phase": p, "t": float(t[i])} for i, p in enumerate(ph)]
    by_ep = {}
    for i in B.priority_order(long, seed=1, min_gap=2):
        by_ep.setdefault((long[i]["episode"], long[i]["phase"]), []).append(long[i]["step"])
    for (e, p), steps in by_ep.items():
        first = steps[:10]
        assert all(abs(a - b) >= 2 for i, a in enumerate(first) for b in first[:i]), (e, p, first)
        assert max(first) - min(first) >= 30, ("picks should spread over the phase", e, p, first)


def test_priority_order_deterministic():
    cands = episodes()
    assert B.priority_order(cands, seed=5) == B.priority_order(cands, seed=5)
    assert B.priority_order(cands, seed=5) != B.priority_order(cands, seed=6)


# fixed-total buffer

def task_arrays(n, tag):
    rng = np.random.default_rng(tag)
    return {"image": rng.integers(0, 255, (n, 8, 8, 3), dtype=np.uint8), "wrist_image": rng.integers(0, 255, (n, 8, 8, 3), dtype=np.uint8), "state": rng.standard_normal((n, 8)).astype(np.float32), "action": rng.standard_normal((n, 10, 7)).astype(np.float32)}


def ranked(tag, n_ep=6):
    cands = episodes(n_ep, tag)
    order = B.priority_order(cands, seed=tag)
    rows = [{**cands[i], "rank": r} for r, i in enumerate(order)]
    return rows, task_arrays(len(rows), tag)


def test_allocate_equal_shares_and_passes_on_unused():
    assert B.allocate(100, {"a": 1000, "b": 1000, "c": 1000}) == {"a": 34, "b": 33, "c": 33}
    assert B.allocate(100, {"a": 10, "b": 1000}) == {"a": 10, "b": 90}
    assert sum(B.allocate(7, {"a": 3, "b": 3}).values()) == 6


def test_buffer_fixed_total_nested_and_hashed(tmp_path):
    buf = B.FixedTotalBuffer(tmp_path / "buf", total=120)
    full = {}
    for k, t in enumerate(("t1", "t2", "t3")):
        rows, arrs = ranked(k + 1)
        full[t] = (rows, arrs)
        shares = buf.admit(t, rows, arrs, {"prompt": f"do {t}"})
        assert buf.stored() <= 120 and sum(shares.values()) == 120
        assert buf.verify()
        for u in buf.state["tasks"]:  # every stored task is exactly a prefix of its ranked list
            got = B.read_rows(buf.root, u)
            assert got == full[u][0][: len(got)]
            np.testing.assert_array_equal(B.load_arrays(buf.root, u)["state"], full[u][1]["state"][: len(got)])
    assert buf.state["shares"] == {"t1": 40, "t2": 40, "t3": 40}
    idx = buf.phase_index()
    for t in idx:  # 40 per task = 10 per phase
        assert {p: len(v) for p, v in idx[t].items()} == {p: 10 for p in PHASES}
    with pytest.raises(AssertionError):
        buf.admit("t1", *ranked(9), {"prompt": "again"})
    # tampering is detected
    a = np.load(buf.root / "t2" / "state.npy")
    a[0, 0] += 1
    np.save(buf.root / "t2" / "state.npy", a)
    assert not buf.verify()


def test_ranked_from_records(tmp_path):
    rec = tmp_path / "rec"
    rec.mkdir()
    for e in range(3):
        n = 20
        ph = ["approach"] * 10 + ["precontact"] * 3 + ["contact"] * 2 + ["opening"] * 4 + ["late"]
        a = task_arrays(n, e)
        np.savez_compressed(rec / f"u{e:03d}_r0_d{e}.npz", update=e, demo=e, prompt="open the microwave", step=np.arange(n), phase=np.array(ph), image=a["image"], wrist_image=a["wrist_image"], state=a["state"], physical=a["action"], phase_source="geometry")
    rows, arrs, meta = B.ranked_from_records(rec, seed=1)
    assert len(rows) == 60 and arrs["image"].shape == (60, 8, 8, 3) and meta["prompt"] == "open the microwave"
    assert {r["phase"] for r in rows[:4]} == set(PHASES), "opening and late map to manipulation; the first 4 cover all phases"
    # arrays follow the rank order
    src = {f"u{e:03d}_r0_d{e}": task_arrays(20, e) for e in range(3)}
    for r, row in enumerate(rows[:10]):
        np.testing.assert_array_equal(arrs["state"][r], src[row["episode"]]["state"][row["step"]])


# sampler

def phase_index(sizes):
    return {t: {p: list(range(i * 100, i * 100 + n)) for i, (p, n) in enumerate(zip(PHASES, ns))} for t, ns in sizes.items()}


def test_split_rotates_remainder():
    assert split(10, 4, 0) == [3, 3, 2, 2] and split(10, 4, 1) == [2, 3, 3, 2] and sum(split(7, 3, 5)) == 7


def test_sampler_counts_coverage_and_resume():
    idx = phase_index({"a": [20, 5, 3, 12], "b": [15, 4, 2, 9]})
    s = OldTaskSampler(idx, seed=1)
    plans = [s.plan(40, u) for u in range(30)]
    for pl in plans:
        assert len(pl) == 40
        assert sum(t == "a" for t, _, _ in pl) == 20
        for t in ("a", "b"):
            per = [sum(1 for t_, p_, _ in pl if t_ == t and p_ == p) for p in PHASES]
            assert max(per) - min(per) <= 1
    # without replacement: a cell repeats a row only after using all of its rows
    seq = [r for pl in plans for t, p, r in pl if (t, p) == ("b", "contact")]
    for i in range(0, len(seq) - 1, 2):
        assert sorted(seq[i : i + 2]) == idx["b"]["contact"]
    # resume: restoring the state reproduces the continuation exactly
    s1 = OldTaskSampler(idx, seed=1)
    for u in range(10):
        s1.plan(40, u)
    s2 = OldTaskSampler(idx, seed=1, state=json.loads(json.dumps(s1.get_state())))
    assert [s1.plan(40, u) for u in range(10, 20)] == [s2.plan(40, u) for u in range(10, 20)]


def test_sampler_no_old_tasks():
    assert OldTaskSampler({}, seed=0).plan(40, 0) == []


# matrix

def eval_rows(table, pos_of):
    rows = []
    for (pol, task), (x, n) in table.items():
        rows += [{"policy_id": pol, "position": pos_of[task], "mode": "ode", "state": s, "success": s < x} for s in range(n)]
    return rows


def test_matrix_metrics():
    ids, pos = ["t1", "t2", "t3"], {"t1": 4, "t2": 1, "t3": 5}
    tbl = {("stage0", "t1"): (20, 40), ("stage0", "t2"): (4, 40),
           ("stage1", "t1"): (16, 40), ("stage1", "t2"): (30, 40), ("stage1", "t3"): (2, 40),
           ("stage2", "t1"): (10, 40), ("stage2", "t2"): (24, 40), ("stage2", "t3"): (36, 40),
           ("base", "t1"): (0, 40), ("base", "t2"): (2, 40), ("base", "t3"): (0, 40)}
    pol = {"stage0": 0, "stage1": 1, "stage2": 2, "base": "base"}
    m = MX.summarize(ids, MX.cells(eval_rows(tbl, pos), pol, {v: k for k, v in pos.items()}))
    assert m["S"][0] == [0.5, 0.1, None] and m["last_stage"] == 2
    assert m["final_avg"] == pytest.approx((0.25 + 0.6 + 0.9) / 3)
    assert m["nbt"] == pytest.approx({"t1": 0.25, "t2": 0.15})
    assert m["nbt_norm"]["t1"] == pytest.approx(0.5)
    assert m["fwt"] == pytest.approx({"t2": 0.05, "t3": 0.05})
    lo, hi = m["ci95"][0][0]
    assert lo < 0.5 < hi


def test_matrix_partial_run():
    ids, pos = ["t1", "t2"], {"t1": 4, "t2": 1}
    m = MX.summarize(ids, MX.cells(eval_rows({("stage0", "t1"): (10, 40)}, pos), {"stage0": 0}, {4: "t1", 1: "t2"}))
    assert m["last_stage"] == 0 and m["final_avg"] == 0.25 and m["nbt"] == {}


# config

def test_repo_config_loads():
    c = C.load(ROOT / "configs" / "cl_opd.yaml")
    assert c["tasks"][0]["id"] == "t035" and c["student"] == "full"
    assert c["per_task"]["warm_updates"] == 100 and c["per_task"]["total_updates"] == 400


@pytest.mark.parametrize("edit", [
    lambda c: c["per_task"].update(warm_updates=400),
    lambda c: c["tasks"].append(dict(c["tasks"][0])),
    lambda c: c["tasks"][0].update(variant="dense"),
    lambda c: c["old_tasks"].update(weight=1.0),
])
def test_bad_configs_rejected(edit):
    c = yaml.safe_load((ROOT / "configs" / "cl_opd.yaml").read_text())
    edit(c)
    with pytest.raises(AssertionError):
        C.check(c)


def test_config_sha_changes_with_any_choice():
    c = C.load(ROOT / "configs" / "cl_opd.yaml")
    d = copy.deepcopy(c)
    d["old_tasks"]["per_update"] = 41
    assert C.sha(c) != C.sha(d) and C.sha(c) == C.sha(copy.deepcopy(c))


# driver plan (dry run: prints commands, touches nothing)

def dry(task, cfg=None):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "cl_opd_run.py"), "--config", str(cfg or ROOT / "configs" / "cl_opd.yaml"), "--run", "/nonexistent/cl_dry", "--task", str(task), "--dry-run"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return r.stdout


def test_driver_task0_plan():
    out = dry(0)
    p1, p2 = [l for l in out.splitlines() if "opd_lora_train.py" in l]
    assert "--stop-after 100" in p1 and "--resume" not in p1 and "--init-master" not in p1 and "--old-buffer" not in p1
    assert "--resume" in p2 and "--replay-per-update 8" in p2 and "--rollouts-per-update 4" in p2 and "--stop-contact-rate 0.05" in p2
    assert "--student-mode full" in p1 and "--geometry microwave" in p1 and "--record-dir" in p1 and "--state-every 10" in p1
    assert "base base -" in out and "stage0 student" in out


def test_driver_later_task_plan():
    out = dry(1)
    p1, p2 = [l for l in out.splitlines() if "opd_lora_train.py" in l]
    for p in (p1, p2):
        assert "--init-master" in p and "--old-per-update 40" in p and "--old-weight 0.5" in p and "--old-buffer" in p
    assert "--geometry none" in p2 and "--stop-contact-rate" not in p2, "no contact stop rule without the microwave probe"
    stage = [l for l in out.splitlines() if l.strip().startswith("stage1 student")][0]
    assert stage.split()[3] == "4,9,5", "evaluates tasks 0..1 and the next task"
    assert "base base" not in out


def test_driver_reads_choices_from_config(tmp_path):
    c = yaml.safe_load((ROOT / "configs" / "cl_opd.yaml").read_text())
    c["per_task"].update(warm_updates=50, total_updates=200, state_every=25, checkpoint_updates=[50, 200])
    c["old_tasks"]["per_update"] = 16
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(c))
    out = dry(1, f)
    assert "--stop-after 50" in out and "--updates 200" in out and "--state-every 25" in out and "--old-per-update 16" in out


def test_wandb_run_names():
    sys.path.insert(0, str(ROOT / "scripts"))
    import cl_opd_wandb as W

    c = C.load(ROOT / "configs" / "cl_opd.yaml")
    assert W.run_name(c, 0) == "cl-opd-c-25demos-s400-t035"
