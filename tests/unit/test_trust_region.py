"""Action-space trust region by step backtracking (clvla.cl_opd.trust_region) and its trainer helper. CPU only.
The trainer test imports scripts/opd_lora_train.py (JAX on the CPU) and is skipped where that import is unavailable."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

from clvla.cl_opd.trust_region import TrustRegion, relative_change

ROOT = Path(__file__).resolve().parents[2]


def test_relative_change():
    v = np.ones((10, 3, 4))
    assert relative_change(v, v) == 0.0
    assert np.isclose(relative_change(v, 1.5 * v), 0.5)
    assert np.isclose(relative_change(2 * v, v), 0.5)


def test_warmup_only_records():
    tr = TrustRegion(3.0)
    for i in range(5):
        s, row = tr.search(lambda s: (100.0 * s, 100.0 * s))
        assert s == 1.0 and row["tr_threshold"] is None and row["tr_halvings"] == 0
    assert tr.history == [100.0] * 5 and tr.threshold() == 300.0


def test_backtracking_halves_until_inside():
    tr = TrustRegion(3.0, history=[1.0] * 20)  # threshold 3
    calls = []

    def rel(s):
        calls.append(s)
        return 20.0 * s, 40.0 * s  # r(1) = 20: halve to 1/8 (2.5 <= 3)

    s, row = tr.search(rel)
    assert s == 0.125 and calls == [1.0, 0.5, 0.25, 0.125]
    assert row["tr_rel_full"] == 20.0 and row["tr_rel"] == 2.5 and row["tr_rel_max"] == 5.0 and row["tr_halvings"] == 3 and not row["tr_capped"]
    assert tr.history[-1] == 2.5  # the applied r enters the running median


def test_inside_step_is_untouched():
    tr = TrustRegion(3.0, history=[1.0] * 20)
    s, row = tr.search(lambda s: (2.9 * s, 3.5 * s))
    assert s == 1.0 and row["tr_halvings"] == 0 and row["tr_threshold"] == 3.0


def test_capped_after_max_halvings():
    tr = TrustRegion(3.0, history=[1.0] * 20)
    s, row = tr.search(lambda s: (1e6, 1e6))
    assert s == 0.5**6 and row["tr_capped"] and row["tr_halvings"] == 6


def test_window_median():
    tr = TrustRegion(2.0, window=3, history=[100.0, 100.0, 1.0, 2.0, 3.0])
    assert tr.threshold() == 4.0


def test_state_round_trip_and_mismatch():
    a = TrustRegion(3.0, history=[1.0, 2.0])
    b = TrustRegion(3.0)
    b.set_state(a.get_state())
    assert b.history == [1.0, 2.0] and b.threshold() is None
    with pytest.raises(AssertionError):
        TrustRegion(2.0).set_state(a.get_state())
    with pytest.raises(AssertionError):
        TrustRegion(0.5)


@pytest.fixture(scope="module")
def OT():
    sys.path.insert(0, str(ROOT / "scripts"))
    sys.path.insert(0, str(ROOT / "tests" / "ogpo_port"))
    try:
        import opd_lora_train
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"trainer import unavailable here: {type(e).__name__}: {e}")
    return opd_lora_train


def test_trainer_tr_step_linear_model(OT):
    """Velocity v = w * x (elementwise): r(s) = s * rms(D x) / rms(w x), so the applied step is exactly master + s D."""
    rng = np.random.default_rng(0)
    keys = ["w"]
    master = {"w": np.ones(6, np.float32)}
    new_np = {"w": np.ones(6, np.float32) * 1.4}  # r(1) = 0.4
    chains = [rng.normal(size=(1, 11, 6)).astype(np.float32) for _ in range(3)]
    events = [{"request": None}] * 3
    vel = lambda th, ob, sm, sk, x, t: th["w"] * x  # noqa: E731
    times = list(np.linspace(1.0, 0.1, 10))
    tr = TrustRegion(3.0, history=[0.05] * 20)  # threshold 0.15: 0.4 -> 0.2 -> 0.1
    applied, row = OT.tr_step(tr, master, new_np, events, chains, master, lambda m: m, lambda ev: (None, None, None), vel, times, keys)
    assert row["tr_scale"] == 0.25 and row["tr_observations"] == 3
    assert np.isclose(row["tr_rel_full"], 0.4) and np.isclose(row["tr_rel"], 0.1)
    assert np.allclose(applied["w"], 1.1) and applied["w"].dtype == np.float32
    tr2 = TrustRegion(3.0)  # warmup: the proposed master itself, not a copy
    applied2, row2 = OT.tr_step(tr2, master, new_np, events, chains, master, lambda m: m, lambda ev: (None, None, None), vel, times, keys)
    assert applied2 is new_np and row2["tr_scale"] == 1.0


def test_driver_passes_the_flag_only_when_set(tmp_path):
    """Dry run of the driver for task 1: --trust-region-k appears with per_task.trust_region_k and not without it."""
    c = yaml.safe_load((ROOT / "configs" / "cl_opd.yaml").read_text())
    outs = {}
    for name, k in (("off", None), ("on", 3.0)):
        cc = {**c, "name": f"tr-test-{name}", "per_task": {**c["per_task"], **({"trust_region_k": k} if k else {})}}
        cfg = tmp_path / f"{name}.yaml"
        cfg.write_text(yaml.safe_dump(cc))
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "cl_opd_run.py"), "--config", str(cfg), "--run", str(tmp_path / f"run_{name}"), "--task", "1", "--dry-run"],
                           cwd=ROOT, env={**os.environ, "DRYRUN": "1"}, capture_output=True, text=True)
        if r.returncode != 0 and "teacher" in (r.stdout + r.stderr).lower() and "not" in (r.stdout + r.stderr).lower():
            pytest.skip("teachers not fetched here")
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
        outs[name] = r.stdout
    assert "--trust-region-k 3.0" in outs["on"] and "--trust-region-k" not in outs["off"]
