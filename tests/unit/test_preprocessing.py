"""Preprocessing must match OpenPI's LIBERO conventions exactly (no invented transformations)."""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest

from clvla.data import episode_to_openpi_frames, load_episode
from clvla.policy import build_observation, quat2axisangle, state_from_env_obs

ROOT = Path(__file__).resolve().parents[2]
MAIN_PY = ROOT / "third_party/openpi/examples/libero/main.py"


def _openpi_main_functions():
    """Extract module constants + `_quat2axisangle` from OpenPI's LIBERO client without importing it (it needs libero)."""
    tree = ast.parse(MAIN_PY.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_quat2axisangle"]
    keep += [
        n for n in tree.body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "").startswith("LIBERO_")
    ]
    ns = {"np": np, "math": math}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(MAIN_PY), "exec"), ns)
    return ns


@pytest.fixture(scope="module")
def openpi_main():
    if not MAIN_PY.is_file():
        pytest.skip("OpenPI checkout missing (run scripts/setup_envs.sh)")
    return _openpi_main_functions()


def test_constants_match_openpi_client(openpi_main, experiment_raw):
    proto = experiment_raw["protocol"]
    assert openpi_main["LIBERO_DUMMY_ACTION"] == proto["simulator"]["dummy_action"]
    assert openpi_main["LIBERO_ENV_RESOLUTION"] == proto["simulator"]["camera_resolution"]
    src = MAIN_PY.read_text()
    assert "resize_size: int = 224" in src and "replan_steps: int = 5" in src and "num_steps_wait: int = 10" in src
    assert "max_steps = 400  # longest training demo has 373 steps" in src  # libero_90 branch
    assert "seed: int = 7" in src


def test_quat2axisangle_identical_to_openpi(openpi_main):
    rng = np.random.default_rng(0)
    quats = rng.normal(size=(500, 4))
    quats /= np.linalg.norm(quats, axis=1, keepdims=True)
    quats = np.vstack([quats, [[0, 0, 0, 1.0]], [[0, 0, 0, 1.0000001]], [[1.0, 0, 0, 0]]])
    for q in quats:
        ours = quat2axisangle(q.copy())
        theirs = openpi_main["_quat2axisangle"](q.copy())
        assert np.array_equal(ours, theirs), (q, ours, theirs)


def test_quat2axisangle_does_not_mutate_input():
    q = np.array([0.0, 0.0, 0.0, 1.5])
    quat2axisangle(q)
    assert q[3] == 1.5


def test_build_observation_matches_openpi_client_steps():
    from openpi_client import image_tools

    rng = np.random.default_rng(1)
    obs = {
        "agentview_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": rng.normal(size=3),
        "robot0_eef_quat": np.array([0.99, 0.05, 0.1, 0.02]) / np.linalg.norm([0.99, 0.05, 0.1, 0.02]),
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
    }
    req = build_observation(obs, "open the microwave", 224)
    expect = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224)
    )
    assert np.array_equal(req["observation/image"], expect)
    assert req["observation/image"].shape == (224, 224, 3) and req["observation/image"].dtype == np.uint8
    assert req["observation/wrist_image"].shape == (224, 224, 3)
    assert req["observation/state"].shape == (8,)
    assert np.allclose(req["observation/state"], state_from_env_obs(obs))
    assert req["prompt"] == "open the microwave"


# ---- OpenPI's real transform chain (policy env only) ----------------------------------------------------------------


@pytest.fixture(scope="module")
def pi05_libero_data_config():
    pytest.importorskip("openpi", reason="policy env only (OpenPI not importable)")
    from clvla.policy import openpi_configs

    return openpi_configs("pi05_libero")


@pytest.mark.policy_env
def test_openpi_libero_config_has_no_extra_action_transform(pi05_libero_data_config):
    tc, dc = pi05_libero_data_config
    names = [type(t).__name__ for t in dc.data_transforms.inputs]
    assert names == ["LiberoInputs"], names  # extra_delta_transform=False -> no DeltaActions
    assert [type(t).__name__ for t in dc.data_transforms.outputs] == ["LiberoOutputs"]
    assert dc.use_quantile_norm is True and dc.asset_id == "physical-intelligence/libero"
    assert tc.model.action_horizon == 10


@pytest.mark.policy_env
def test_demo_frames_flow_through_openpi_training_chain(pi05_libero_data_config, demo_file):
    from openpi import transforms as T

    tc, dc = pi05_libero_data_config
    ep = load_episode(demo_file, "demo_0")
    frame = episode_to_openpi_frames(ep, "open the microwave", horizon=10, timesteps=[2])[0]

    repacked = T.compose(dc.repack_transforms.inputs)(dict(frame))
    request_keys = set(
        build_observation(
            {
                "agentview_image": np.zeros((8, 8, 3), np.uint8),
                "robot0_eye_in_hand_image": np.zeros((8, 8, 3), np.uint8),
                "robot0_eef_pos": np.zeros(3),
                "robot0_eef_quat": np.array([0, 0, 0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2),
            },
            "x",
            8,
        )
    )
    assert set(repacked) == request_keys | {"actions"}  # training samples and eval requests share one key space

    rng = np.random.default_rng(0)
    q01 = {"state": rng.uniform(-1, 0, 8), "actions": rng.uniform(-1, 0, 7)}
    q99 = {k: v + rng.uniform(0.5, 2, v.shape) for k, v in q01.items()}
    stats = {k: T.NormStats(mean=np.zeros_like(q01[k]), std=np.ones_like(q01[k]), q01=q01[k], q99=q99[k]) for k in q01}
    pre_norm = T.compose([*dc.repack_transforms.inputs, *dc.data_transforms.inputs])(dict(frame))
    normed = T.compose(
        [
            *dc.repack_transforms.inputs,
            *dc.data_transforms.inputs,
            T.Normalize(stats, use_quantiles=dc.use_quantile_norm),
        ]
    )(dict(frame))
    for k, raw in (("state", frame["state"]), ("actions", frame["actions"])):
        assert np.allclose(pre_norm[k], raw)  # LiberoInputs passes state/actions through untouched
        expect = (raw - q01[k]) / (q99[k] - q01[k] + 1e-6) * 2.0 - 1.0
        assert np.allclose(normed[k], expect, atol=1e-5), k

    full = T.compose(
        [
            *dc.repack_transforms.inputs,
            *dc.data_transforms.inputs,
            T.Normalize(stats, use_quantiles=dc.use_quantile_norm),
            *dc.model_transforms.inputs,
        ]
    )(dict(frame))
    assert set(full["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert full["image"]["base_0_rgb"].shape == (224, 224, 3)
    assert bool(full["image_mask"]["right_wrist_0_rgb"]) is False  # pi0/pi05 mask the missing right wrist camera
    assert full["actions"].shape == (10, tc.model.action_dim)
    assert np.allclose(full["actions"][:, 7:], 0) and full["state"].shape[-1] == tc.model.action_dim
    assert "tokenized_prompt" in full


def test_committed_norm_stats_match_libero_conventions(repo_root):
    """Bundled stats must describe LIBERO's conventions: gripper action in [-1, 1], 2 gripper qpos, axis-angle ~pi."""
    import os

    ck = os.environ.get("OPENPI_DATA_HOME")
    p = (
        Path(ck or "/nonexistent")
        / "openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json"
    )
    if not p.is_file():
        pytest.skip("checkpoint not downloaded")
    s = json.loads(p.read_text())["norm_stats"]
    assert len(s["state"]["mean"]) == 8 and len(s["actions"]["mean"]) == 7
    assert s["actions"]["q01"][6] < -0.9 and s["actions"]["q99"][6] > 0.9  # gripper -1 open / +1 close
    assert 2.5 < s["state"]["mean"][3] < 3.5  # axis-angle x near pi for a downward-facing gripper
    assert s["state"]["mean"][6] > 0 > s["state"]["mean"][7]  # two finger joints, opposite signs
