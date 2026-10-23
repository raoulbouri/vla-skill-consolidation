"""Stage 6: real demonstrations through the intended preprocessing path (policy env, CPU; needs data + checkpoint assets)."""

import glob
import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openpi", reason="policy env only")

from clvla.data import episode_state, episode_to_openpi_frames, load_episode, validate_demo_file  # noqa: E402
from clvla.policy import openpi_training_transforms  # noqa: E402

pytestmark = [pytest.mark.policy_env, pytest.mark.data, pytest.mark.checkpoint]


@pytest.fixture(scope="module")
def chain(cfg):
    if not Path(cfg.checkpoint.local_dir, "params").is_dir():
        pytest.skip("checkpoint not downloaded")
    return openpi_training_transforms(cfg.checkpoint.openpi_config, cfg.checkpoint.local_dir)


def test_real_demo_through_openpi_training_transforms(cfg, manifest, chain):
    transform, dc, norm_stats = chain
    t = manifest.tasks[0]
    path = Path(cfg.data.root) / t.demo_file
    idx = validate_demo_file(path, expected_instruction=t.instruction, expected_demos=50)
    key = idx.split(cfg.splits.train_demo_indices)[0]
    ep = load_episode(path, key)
    frames = episode_to_openpi_frames(ep, t.instruction, cfg.policy.action_horizon)
    outs = [transform(dict(f)) for f in frames[:: max(1, len(frames) // 20)]]
    o = outs[0]
    assert o["image"]["base_0_rgb"].shape == (224, 224, 3) and o["image"]["left_wrist_0_rgb"].shape == (224, 224, 3)
    assert o["actions"].shape == (cfg.policy.action_horizon, 32) and o["state"].shape == (32,)
    assert o["tokenized_prompt"].shape[0] > 0 and bool(o["tokenized_prompt_mask"].any())
    acts = np.stack([x["actions"][:, :7] for x in outs])
    states = np.stack([x["state"][:8] for x in outs])
    assert np.all(np.isfinite(acts)) and np.all(np.isfinite(states))
    # Quantile-normalised values of in-distribution data should mostly fall in [-1, 1].
    frac_a = float(np.mean(np.abs(acts) <= 1.05))
    frac_s = float(np.mean(np.abs(states) <= 1.05))
    print(
        f"\n{t.task_id} {key}: {len(frames)} frames; normalised actions in [-1.05,1.05]: {frac_a:.3f}, states: {frac_s:.3f}"
    )
    assert frac_a > 0.9 and frac_s > 0.8


def test_raw_pretraining_hdf5_matches_bundled_norm_stats(cfg):
    """Independent convention check: statistics of the raw official HDF5 for the suites pi05_libero was trained on must
    resemble the checkpoint's bundled norm stats (gripper sign, axis-angle state, units). Not exact: OpenPI's data
    are OpenVLA's no-op-filtered 256px re-renders of these demos."""
    import h5py

    ns_path = Path(cfg.checkpoint.local_dir) / "assets" / cfg.checkpoint.norm_stats_asset_id / "norm_stats.json"
    if not ns_path.is_file():
        pytest.skip("checkpoint not downloaded")
    ns = json.loads(ns_path.read_text())["norm_stats"]
    files = []
    for suite in ("libero_spatial", "libero_object", "libero_goal", "libero_10"):
        files += sorted(glob.glob(str(Path(cfg.data.root) / suite / "*.hdf5")))[:3]
    if len(files) < 12:
        pytest.skip("pretraining-suite HDF5 files not present under LIBERO_DATA_ROOT")
    A, S = [], []
    for f in files:
        with h5py.File(f, "r") as h:
            for k in list(h["data"].keys())[:10]:
                g = h["data"][k]
                A.append(g["actions"][()])
                S.append(
                    episode_state(
                        {"obs/ee_states": g["obs/ee_states"][()], "obs/gripper_states": g["obs/gripper_states"][()]}
                    )
                )
    A, S = np.concatenate(A), np.concatenate(S)
    a_mean, s_mean = A.mean(0), S.mean(0)
    print("\nraw action mean", np.round(a_mean, 3), "\nckpt action mean", np.round(ns["actions"]["mean"], 3))
    print("raw state mean ", np.round(s_mean, 3), "\nckpt state mean ", np.round(ns["state"]["mean"], 3))
    a_std, s_std = np.array(ns["actions"]["std"]), np.array(ns["state"]["std"])
    assert np.all(np.abs(a_mean - ns["actions"]["mean"]) < 0.5 * a_std + 0.02), "action convention mismatch"
    assert np.all(np.abs(s_mean - ns["state"]["mean"]) < 0.5 * s_std + 0.02), "state convention mismatch"
    assert set(np.unique(A[:, 6])) <= {-1.0, 1.0}  # binary gripper command, same sign convention as the model
