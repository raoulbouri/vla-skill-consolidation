"""Continual-learning OPD on real artifacts, CPU only (policy env): t35 batches from the default-recipe run, model shapes,
and (slow, ~10 GB RAM) the full student built from the real base weights. Skipped where an artifact is missing.
  scripts/py-policy -m pytest -p no:cacheprovider -rA tests/integration/test_cl_opd_real.py [-m 'not slow']
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from clvla.cl_opd import buffer as B
from clvla.cl_opd.phases import PHASES, label_t35

ROOT = Path(__file__).resolve().parents[2]
T35 = ROOT / "outputs/opd_t35_curriculum/20261103T103057Z_phase2_replay_from_u100/student/batches"
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "tests" / "ogpo_port"))


def teacher_batches(n=100):
    fs = sorted(T35.glob("update_*.npz"))[:n]
    if not fs:
        pytest.skip("t35 default-recipe batches not present")
    return [np.load(f) for f in fs]


def test_event_phases_match_t35_geometry_phases():
    """The buffer's event labeller on real snapshots agrees with the phases the t35 trainer recorded."""
    agree = total = 0
    for z in teacher_batches():
        for j, ph in enumerate(z["phases"]):
            if f"snap_{j}" not in z.files:
                continue
            got = label_t35(z[f"snap_{j}"][None])[0]
            want = B.TRAINER_TO_BUFFER[str(ph)]
            agree += got == want
            total += 1
    assert total >= 700 and agree == total, f"{agree}/{total}"


def test_buffer_from_real_teacher_observations(tmp_path):
    """Real images and states through ranking and admission: phase balance, bytes per transition, verify."""
    rec = tmp_path / "rec"
    rec.mkdir()
    for u, z in enumerate(teacher_batches()):
        n, demo = len(z["phases"]), int(np.atleast_1d(z["demo"])[0])
        np.savez_compressed(rec / f"u{u:03d}_r0_d{demo}.npz", update=u, demo=demo, prompt="open the microwave", step=np.asarray(z["events"]), phase=np.asarray(z["phases"]),
                            image=np.stack([z[f"image_{j}"] for j in range(n)]), wrist_image=np.stack([z[f"wrist_{j}"] for j in range(n)]), state=np.stack([z[f"state_{j}"] for j in range(n)]),
                            physical=np.stack([z[f"physical_{j}"] for j in range(n)]), phase_source="geometry")
    rows, arrs, meta = B.ranked_from_records(rec, seed=11)
    buf = B.FixedTotalBuffer(tmp_path / "buf", total=400)
    buf.admit("t035", rows, arrs, meta)
    assert buf.verify() and buf.stored() == 400
    counts = {p: len(v) for p, v in buf.phase_index()["t035"].items()}
    assert max(counts.values()) - min(counts.values()) <= 1, counts
    per = buf.bytes_on_disk() / 400
    print(f"\n[buffer] {len(rows)} candidates from {meta['episodes']} episodes; phases {counts}; {per / 1e3:.0f} kB per transition")
    assert 250e3 < per < 400e3, "two 224x224 RGB images plus state and action"


def test_full_student_trainable_set():
    """Shapes only: the full student trains exactly the action expert (430M with adaRMS) and the heads, never the VLM."""
    pytest.importorskip("openpi")
    from opd_lora_student import is_expert, is_trainable_full, shapes

    sh = shapes(full=True)
    tr = {k: v for k, v in sh.items() if is_trainable_full(k)}
    assert not any(k.startswith("PaliGemma/img") for k in tr)
    assert all(is_expert(k) or not k.startswith("PaliGemma") for k in tr)
    assert int(sum(np.prod(v) for v in tr.values())) == 430_098_464
    assert not any("lora" in k for k in sh)


def test_trainer_and_driver_parse_on_cpu():
    """The edited trainer imports and parses its arguments (no model loaded); the simulator script compiles under py3.8."""
    env = {**os.environ, "PYTHONPATH": f"{ROOT / 'scripts'}:{ROOT / 'tests' / 'ogpo_port'}:{os.environ.get('PYTHONPATH', '')}"}
    r = subprocess.run([sys.executable, "scripts/opd_lora_train.py", "--help"], cwd=ROOT, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    for flag in ("--student-mode", "--init-master", "--old-buffer", "--old-per-update", "--record-dir", "--state-every", "--geometry"):
        assert flag in r.stdout
    sim_py = os.environ.get("CLVLA_SIM_PY")
    if sim_py:
        r = subprocess.run([sim_py, "-m", "py_compile", "scripts/opd_lora_collect.py"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


@pytest.mark.slow
def test_full_student_starts_equal_to_base():
    """Real weights on CPU: the full student's master is the base expert in FP32, and the built student's leaves equal the
    base model's bit for bit, so step 0 is the base function."""
    pytest.importorskip("openpi")
    from flax import nnx, traverse_util

    from clvla.config import load_experiment_config
    from clvla.ogpo.pi05_sde_policy import model_of
    from clvla.policy import load_official_policy
    from opd_lora_student import build_student_model, initial_full, is_full_adapter, is_trainable_full

    conf = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    base = model_of(policy)
    master = initial_full(str(Path(conf.checkpoint.local_dir) / "params"))
    assert is_full_adapter(master) and all(v.dtype == np.float32 for v in master.values())
    student = build_student_model(base, master)
    fb = traverse_util.flatten_dict(nnx.split(base)[1].to_pure_dict(), sep="/")
    fs = traverse_util.flatten_dict(nnx.split(student)[1].to_pure_dict(), sep="/")
    assert set(fb) == set(fs)
    diff = [k for k in fb if not np.array_equal(np.asarray(fb[k]), np.asarray(fs[k])) or np.asarray(fb[k]).dtype != np.asarray(fs[k]).dtype]
    assert not diff, diff[:5]
    assert sum(is_trainable_full(k) for k in fs) == len(master)
    print(json.dumps({"leaves": len(fs), "trainable_leaves": len(master)}))
