"""Re-render planning, alignment audit, manifests and overwrite refusal (no simulator)."""

import numpy as np
import pytest

from clvla.rerender import (
    EXPECTED_OFFSET,
    AlignmentAmbiguity,
    ReconstructionError,
    Thresholds,
    alignment_plan,
    build_processed_manifest,
    check_split_isolation,
    choose_offset,
    estimate_disk_bytes,
    offset_residuals,
    processed_paths,
    refuse_existing,
    stable_content,
    validate_rerendered,
)


def test_alignment_plan_pairs_each_action_with_its_pre_action_state():
    p = alignment_plan(5)
    assert p["pairs"] == [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]  # obs for actions[k] = render(states[k]), incl. k=0
    assert p["validation"] == [(1, 0), (2, 1), (3, 2), (4, 3)]  # render(states[k]) must match HDF5 obs[k-1]
    assert p["replay"] == [(0, 0, 1), (1, 1, 2), (2, 2, 3), (3, 3, 4)]  # step actions[k] from states[k] -> states[k+1]
    with pytest.raises(ValueError):
        alignment_plan(1)


def _synthetic_trajectory(T=40, noise=1e-4, seed=0):
    """Smooth 8-D 'true' states s[0..T]; HDF5 stores post-action obs[k] = s[k+1]; render(states[k]) = s[k] + noise."""
    rng = np.random.default_rng(seed)
    s = np.cumsum(rng.normal(scale=0.02, size=(T + 1, 8)), axis=0)
    hdf5_obs = s[1:]
    rendered = s[:-1] + rng.normal(scale=noise, size=(T, 8))
    return rendered, hdf5_obs


def test_offset_audit_identifies_the_post_action_storage():
    rendered, hdf5 = _synthetic_trajectory()
    med = offset_residuals(rendered, hdf5)
    assert set(med) == {-1, 0, 1}
    assert min(med, key=med.get) == EXPECTED_OFFSET
    assert choose_offset(med, Thresholds()) == 1


def test_offset_audit_refuses_when_data_are_aligned_differently():
    rendered, hdf5 = _synthetic_trajectory()
    with pytest.raises(AlignmentAmbiguity, match="best offset is 0"):
        choose_offset(offset_residuals(rendered, rendered.copy()), Thresholds())  # already pre-action aligned


def test_offset_audit_refuses_ambiguous_or_large_residuals():
    with pytest.raises(AlignmentAmbiguity, match="margin"):
        choose_offset({-1: 0.0012, 0: 0.0015, 1: 0.001}, Thresholds())
    with pytest.raises(AlignmentAmbiguity, match="aligned residual"):
        choose_offset({-1: 0.5, 0: 0.2, 1: 0.01}, Thresholds())
    with pytest.raises(AlignmentAmbiguity, match="not measurable"):
        choose_offset({0: 0.1}, Thresholds())


def test_thresholds_reject_unknown_keys():
    assert Thresholds.from_dict({"max_replay_qpos_median": 0.02}).max_replay_qpos_median == 0.02
    assert Thresholds().require_final_success is False  # informational by default
    with pytest.raises(ValueError, match="unknown"):
        Thresholds.from_dict({"max_replay_eror": 0.02})


def _arrays(T=6, res=256):
    return {
        "actions": np.zeros((T, 7)),
        "states": np.zeros((T, 77)),
        "rewards": np.zeros(T, np.uint8),
        "dones": np.zeros(T, np.uint8),
        "obs/state": np.zeros((T, 8), np.float32),
        "obs/ee_pos": np.zeros((T, 3)),
        "obs/ee_quat": np.zeros((T, 4)),
        "obs/gripper_qpos": np.zeros((T, 2)),
        "obs/joint_pos": np.zeros((T, 7)),
        "obs/agentview_rgb": np.zeros((T, res, res, 3), np.uint8),
        "obs/eye_in_hand_rgb": np.zeros((T, res, res, 3), np.uint8),
    }


def test_rerendered_arrays_validation():
    validate_rerendered(_arrays(), 6, 256)
    bad = _arrays()
    bad["obs/state"][2, 3] = np.nan
    with pytest.raises(ReconstructionError, match="non-finite"):
        validate_rerendered(bad, 6, 256)
    with pytest.raises(ReconstructionError, match="agentview"):
        validate_rerendered(_arrays(res=128), 6, 256)
    short = _arrays()
    short["actions"] = np.zeros((5, 7))
    with pytest.raises(ReconstructionError, match="actions"):
        validate_rerendered(short, 6, 256)


def test_overwrite_refusal_includes_partial_files(tmp_path):
    out = tmp_path / "x.hdf5"
    refuse_existing(out)  # nothing there
    (tmp_path / "x.hdf5.partial").write_bytes(b"")
    with pytest.raises(FileExistsError, match="partial"):
        refuse_existing(out)
    out.write_bytes(b"")
    with pytest.raises(FileExistsError):
        refuse_existing(out)


def test_paths_are_versioned():
    p = processed_paths("/root", "libero90_aligned256_v1", "KITCHEN_SCENE7_open_the_microwave")
    assert str(p["hdf5"]) == "/root/libero90_aligned256_v1/hdf5/KITCHEN_SCENE7_open_the_microwave_demo.hdf5"
    assert str(p["manifest"]) == "/root/libero90_aligned256_v1/manifest.json"


def _report(idx, created="2026-10-25T00:00:00Z"):
    return {
        "libero_90_index": idx,
        "name": f"T{idx}",
        "output_sha256": "a" * 64,
        "created_utc": created,
        "elapsed_s": 12.3,
        "hostname": "h",
        "demos": {"demo_0": {"T": 5, "created_utc": created}},
    }


def test_processed_manifest_is_deterministic_and_ignores_volatile_fields():
    th = Thresholds()
    a = build_processed_manifest("v1", [_report(78), _report(3)], {"rev": "x"}, th, {"c": 1})
    b = build_processed_manifest(
        "v1", [_report(3, "2027-01-01"), _report(78, "2027-01-01")], {"rev": "x"}, th, {"c": 1}
    )
    assert [t["libero_90_index"] for t in a["tasks"]] == [3, 78]
    assert a["content_sha256"] == b["content_sha256"]
    c = build_processed_manifest(
        "v1", [_report(3), dict(_report(78), output_sha256="b" * 64)], {"rev": "x"}, th, {"c": 1}
    )
    assert c["content_sha256"] != a["content_sha256"]
    with pytest.raises(ValueError, match="duplicate"):
        build_processed_manifest("v1", [_report(3), _report(3)], {}, th, {})
    assert "created_utc" not in stable_content(_report(1))["demos"]["demo_0"]


def test_split_isolation():
    keys = [f"demo_{i}" for i in range(50)]
    check_split_isolation(keys[:45], keys[45:], keys)
    with pytest.raises(ValueError, match="leakage"):
        check_split_isolation(keys[:46], keys[45:], keys)
    with pytest.raises(ValueError, match="non-empty"):
        check_split_isolation(keys, [], keys)
    with pytest.raises(ValueError, match="not in the file"):
        check_split_isolation(keys[:45], ["demo_99"], keys)
    with pytest.raises(ValueError, match="duplicate"):
        check_split_isolation(keys[:45] + ["demo_0"], keys[45:], keys)


def test_disk_estimate():
    assert estimate_disk_bytes(1000, 256) == 1000 * (2 * 256 * 256 * 3 + 1024)


def test_replay_is_flagged_not_gated_by_default():
    from clvla.rerender import Thresholds, replay_summary

    th = Thresholds()
    assert th.gate_on_replay is False and th.max_replay_qpos_median == 0.01
    assert Thresholds.from_dict({"gate_on_replay": True}).gate_on_replay is True
    s = replay_summary([0.001, 0.002, 0.5], [3.0, 9.0, 1.0])
    assert s["qpos_median"] == 0.002 and s["qvel_max"] == 9.0 and s["qpos_max"] == 0.5
    assert replay_summary([], []) == {
        k: 0.0 for k in ("qpos_median", "qpos_p95", "qpos_max", "qvel_median", "qvel_p95", "qvel_max")
    }
