"""Per-trajectory gradient clipping (clvla.cl_opd.traj_clip) and its trainer integration. CPU only.
The trainer tests import scripts/opd_lora_train.py (JAX on the CPU) and are skipped where that import is unavailable."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from clvla.cl_opd import config as C
from clvla.cl_opd.traj_clip import TrajClip, loss_weights, scales, source_class, thresholds

ROOT = Path(__file__).resolve().parents[2]
TRAINER_BEFORE_STAGE2 = "2dde6ac5f0063364c41f75b3aeae2733028dce991c9eb7c3e6620d6c72dcb5bd"  # the trainer of every historical run


def batch(rng, n_fresh=32, n_rep=8, n_old=40, dim=(50, 30), spike=None):
    """Random per-trajectory gradients in two parameter groups; spike = (index, factor) multiplies one of them."""
    cls = ["fresh"] * n_fresh + ["replay"] * n_rep + ["old"] * n_old
    mu = {"expert": rng.normal(size=dim[0]), "heads": rng.normal(size=dim[1])}  # a shared direction, as real batches have (Stage 1: cos(new, old) ~ 0.7)
    g = [{n: mu[n] + rng.normal(size=v.size) for n, v in mu.items()} for _ in cls]
    if spike:
        i, f = spike
        g[i] = {k: v * f for k, v in g[i].items()}
    return cls, g


def flatv(d):
    return np.concatenate([d["expert"], d["heads"]])


def run_clip(cls, g, k, w):
    c = TrajClip(k, w)
    for i, (cl, gi) in enumerate(zip(cls, g)):
        c.record(i, 0, cl, {n: float(v @ v) for n, v in gi.items()})
    c.plan()
    return c


# the rule

def test_source_class_layout():
    assert [source_class(j, 3, 5) for j in range(7)] == ["fresh"] * 3 + ["replay"] * 2 + ["old"] * 2


def test_loss_weights_match_the_trainer_objective():
    cls = ["fresh"] * 32 + ["replay"] * 8 + ["old"] * 40
    a = loss_weights(cls, 0.5)
    c = np.array(cls)
    assert np.allclose(a[c != "old"], 0.5 / 40) and np.allclose(a[c == "old"], 0.5 / 40)
    assert np.isclose(a[c != "old"].sum(), 0.5) and np.isclose(a[c == "old"].sum(), 0.5)
    # fresh and replay share ONE new-task mean: equal weight per trajectory, proportional to their counts (not 1/2 each)
    assert np.isclose(a[c == "fresh"].sum() / a[c == "replay"].sum(), 32 / 8)
    a2 = loss_weights(["fresh"] * 16 + ["old"] * 80, 0.2)
    assert np.allclose(a2[:16], 0.8 / 16) and np.allclose(a2[16:], 0.2 / 80)
    a0 = loss_weights(["fresh"] * 32 + ["replay"] * 8, None)
    assert np.allclose(a0, 1 / 40)


def test_thresholds_are_class_wise_medians():
    n = np.array([1, 2, 3, 10, 20, 30, 100.0])
    cls = np.array(["fresh"] * 3 + ["replay"] * 3 + ["old"])
    t = thresholds(n, cls, 3)
    assert np.allclose(t, [6, 6, 6, 60, 60, 60, 300])
    assert np.all(np.isinf(thresholds(n, cls, 0)))
    s = scales(n, t)
    assert np.all(s <= 1) and np.allclose(s, 1)  # nobody above 3x their own median


def test_two_pass_correction_equals_direct_clipped_sum():
    rng = np.random.default_rng(0)
    cls, g = batch(rng, spike=(5, 800.0))
    c = run_clip(cls, g, 3.0, 0.5)
    G = np.stack([flatv(x) for x in g])
    g_u = c.a @ G
    d = sum(c.coef(i) * G[i] for i in c.todo)
    direct = (c.a * c.s) @ G
    assert np.allclose(g_u + d, direct)
    assert 5 in c.todo
    assert np.isclose(np.linalg.norm(c.s[5] * G[5]), c.tau[5])  # clipped to its threshold, not dropped
    # loss weights untouched: unclipped trajectories keep a_i exactly
    keep = c.s == 1
    assert np.allclose((c.a * c.s)[keep], c.a[keep])


def test_spike_loses_its_dominance():
    rng = np.random.default_rng(1)
    cls, g = batch(rng, spike=(3, 1000.0))
    c = run_clip(cls, g, 3.0, 0.5)
    G = np.stack([flatv(x) for x in g])
    for b in (c.a, c.a * c.s):
        gg = b @ G
        infl = b * (G @ gg) / (gg @ gg)
        bound = b * np.linalg.norm(G, axis=1) / np.linalg.norm(gg)
        assert np.all(infl <= bound + 1e-12)  # the logged bound is an upper bound on the influence
    gu, gc = c.a @ G, (c.a * c.s) @ G
    assert c.a[3] * (G[3] @ gu) / (gu @ gu) > 0.95  # partly along the shared direction, so below the real spikes' 0.998
    assert (c.a * c.s)[3] * (G[3] @ gc) / (gc @ gc) < 0.1
    st = c.stats(np.linalg.norm(gu), np.linalg.norm(gc))
    assert st["clip_n_fresh"] >= 1 and st["clip_frac_old"] == 0.0 and st["clip_infl_bound_clipped"] < 0.1


def test_class_wise_threshold_avoids_the_phase1_trap():
    """Phase 1: fresh teacher-driven trajectories have ~5x the norm of the old-task ones. A batch-wide median (set by the
    80 old trajectories) would clip every fresh one; the class-wise rule clips none."""
    rng = np.random.default_rng(2)
    cls = ["fresh"] * 16 + ["old"] * 80
    n = np.r_[rng.uniform(4000, 6000, 16), rng.uniform(800, 1200, 80)]
    assert not np.any(scales(n, thresholds(n, np.array(cls), 3)) < 1)
    assert np.all(scales(n, np.full(96, 3 * np.median(n)))[:16] < 1)


def test_k_zero_is_a_no_op():
    rng = np.random.default_rng(3)
    cls, g = batch(rng, spike=(0, 1e4))
    c = run_clip(cls, g, 0.0, 0.5)
    assert c.todo == [] and np.all(c.s == 1)


# trainer integration

def _strip(text):
    out, skip = [], False
    for line in text.split("\n"):
        if line.strip() == "# traj-clip begin":
            assert not skip, "nested traj-clip block"
            skip = True
            continue
        if line.strip() == "# traj-clip end":
            assert skip, "unbalanced traj-clip block"
            skip = False
            continue
        if not skip:
            out.append(line)
    assert not skip
    return "\n".join(out)


def test_trainer_without_clip_blocks_is_the_historical_trainer():
    """Every clipping line sits in a marked block; without them the file is byte-identical to the trainer of the earlier
    runs, and every block is inert unless --traj-clip-k > 0 (default 0)."""
    src = (ROOT / "scripts" / "opd_lora_train.py").read_text()
    assert hashlib.sha256(_strip(src).encode()).hexdigest() == TRAINER_BEFORE_STAGE2
    assert '"--traj-clip-k", type=float, default=0.0' in src


@pytest.fixture(scope="module")
def OT():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "tests" / "ogpo_port"))
    try:
        import opd_lora_train
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"trainer import unavailable here: {type(e).__name__}: {e}")
    return opd_lora_train


def test_trainer_pass2_and_finish(OT):
    import jax
    import jax.numpy as jnp

    rng = np.random.default_rng(4)
    keys = ["expert/a", "expert/b", "heads/c"]
    group = lambda k: k.split("/")[0]  # noqa: E731
    shapes = {"expert/a": (6, 5), "expert/b": (7,), "heads/c": (4, 3)}
    cls = ["fresh"] * 12 + ["replay"] * 4 + ["old"] * 20
    mu = {k: rng.normal(size=shapes[k]) for k in keys}
    grads = [{k: (mu[k] + rng.normal(size=shapes[k])).astype(np.float32) for k in keys} for _ in cls]
    grads[2] = {k: v * 500 for k, v in grads[2].items()}  # spike in fresh
    grads[30] = {k: v * 40 for k, v in grads[30].items()}  # spike in old
    leaf_stats = jax.jit(lambda g: {k: (jnp.vdot(g[k].reshape(-1).astype(jnp.float32), g[k].reshape(-1).astype(jnp.float32)), jnp.sum(g[k].astype(jnp.float32))) for k in keys})
    leaf_dots = jax.jit(lambda x, y: {k: jnp.vdot(x[k].reshape(-1).astype(jnp.float32), y[k].reshape(-1).astype(jnp.float32)) for k in keys})
    clip = TrajClip(3.0, 0.5)
    acc = {k: jnp.zeros(shapes[k], jnp.float32) for k in keys}
    acc_old = {k: jnp.zeros(shapes[k], jnp.float32) for k in keys}
    for i, (c, gi) in enumerate(zip(cls, grads)):
        gh = {k: jnp.asarray(v) for k, v in gi.items()}
        fp, sq = OT._leaf_check(leaf_stats(gh), keys, group)
        clip.record(i, 0, c, sq, fp)
        tgt = acc_old if c == "old" else acc
        for k in keys:
            tgt[k] = tgt[k] + gh[k]
    calls = []

    def recompute(j, m):
        calls.append(j)
        return {k: jnp.asarray(v) for k, v in grads[j].items()}

    d, info = OT.clip_pass2(clip, recompute, leaf_stats, leaf_dots, keys, group, True, acc, acc_old)
    assert set(clip.todo) == {2, 30} and set(calls) >= {2, 30}
    n_new, n_old = 16, 20
    g_u = {k: np.asarray(0.5 * acc[k] / n_new + 0.5 * acc_old[k] / n_old, np.float32) for k in keys}
    dot = lambda x, y: sum(float(np.sum(np.asarray(x[k], np.float64) * np.asarray(y[k], np.float64))) for k in keys)  # noqa: E731

    def by(x):
        o = {}
        for k in keys:
            o[group(k)] = o.get(group(k), 0.0) + float(np.sum(np.asarray(x[k], np.float64) ** 2))
        return {n: v**0.5 for n, v in o.items()}

    gn_u, gr_u = dot(g_u, g_u) ** 0.5, by(g_u)
    g_c, gn, gr, row = OT.clip_finish(clip, g_u, gn_u, gr_u, d, info, dot, by, keys)
    # the clipped update is exactly sum_i a_i s_i g_i, in both parameter groups
    want = {k: sum(clip.a[i] * clip.s[i] * grads[i][k].astype(np.float64) for i in range(len(cls))) for k in keys}
    # FP32: pass 1 sums the spike unclipped and pass 2 subtracts most of it again, so the rounding of the spike's terms
    # (~1e-7 of |a_i g_i|) stays in the result; measured here relative to the clipped update and on real spikes by the validator
    err = sum(float(np.sum((g_c[k] - want[k]) ** 2)) for k in keys) ** 0.5 / sum(float(np.sum(want[k] ** 2)) for k in keys) ** 0.5
    assert err < 1e-4, err
    assert np.isclose(gn, dot(want, want) ** 0.5, rtol=1e-5)
    assert set(gr) == {"expert", "heads"} and row["clip_correction_expert"] > 0 and row["clip_correction_heads"] > 0
    assert row["clip_recompute_bitwise"] and row["clip_recompute_norm_rel_max"] == 0.0
    assert row["clip_n_fresh"] == 1 and row["clip_n_old"] == 1 and row["clip_n_replay"] == 0
    # exact influences of the top trajectory, before and after
    G = {i: np.concatenate([grads[i][k].reshape(-1).astype(np.float64) for k in keys]) for i in range(len(cls))}
    vu = np.concatenate([np.asarray(g_u[k], np.float64).reshape(-1) for k in keys])
    vc = np.concatenate([want[k].reshape(-1) for k in keys])
    t = clip.top(False)
    assert t == 2
    assert np.isclose(row["clip_top_unclipped_infl_before"], clip.a[t] * G[t] @ vu / (vu @ vu), rtol=1e-4)
    assert np.isclose(row["clip_top_unclipped_infl_after"], clip.a[t] * clip.s[t] * G[t] @ vc / (vc @ vc), rtol=1e-4)
    assert row["clip_top_unclipped_infl_before"] > 0.9 > 0.2 > row["clip_top_unclipped_infl_after"]
    cos = vu @ vc / np.linalg.norm(vu) / np.linalg.norm(vc)
    assert np.isclose(row["clip_angle_deg"], np.degrees(np.arccos(cos)), atol=1e-3)
    # a recomputation that differs is reported
    grads[2]["heads/c"][0, 0] += 1.0
    _, info2 = OT.clip_pass2(clip, recompute, leaf_stats, leaf_dots, keys, group, True, acc, acc_old)
    assert not all(c["bitwise"] for c in info2["checks"])


# driver and config

def _dry(task, cfg):
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "cl_opd_run.py"), "--config", str(cfg), "--run", "/nonexistent/cl_dry", "--task", str(task), "--dry-run"], capture_output=True, text=True, env=env, cwd=ROOT)
    assert r.returncode == 0, r.stderr
    return [l for l in r.stdout.splitlines() if "opd_lora_train.py" in l]


def test_driver_passes_traj_clip_only_when_configured(tmp_path):
    for name in ("cl_opd.yaml", "cl_opd_t063_noreplay.yaml"):
        assert all("--traj-clip-k" not in l for l in _dry(1, ROOT / "configs" / name)), "existing configs must not enable clipping"
    c = yaml.safe_load((ROOT / "configs" / "cl_opd.yaml").read_text())
    c["per_task"]["traj_clip_k"] = 3.0
    f = tmp_path / "c.yaml"
    f.write_text(yaml.safe_dump(c))
    p1, p2 = _dry(1, f)
    assert "--traj-clip-k 3.0" in p1 and "--traj-clip-k 3.0" in p2


def test_config_rejects_bad_traj_clip_k():
    c = yaml.safe_load((ROOT / "configs" / "cl_opd.yaml").read_text())
    c["per_task"]["traj_clip_k"] = 0.5
    with pytest.raises(AssertionError):
        C.check(c)
    c["per_task"]["traj_clip_k"] = 3.0
    C.check(c)
