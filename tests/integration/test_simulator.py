"""Stage 2: simulator. Official LIBERO-90 envs for the manifest tasks (sim env, EGL on CLVLA_GPU)."""

import numpy as np
import pytest

pytest.importorskip("libero", reason="simulator env only")

from clvla.data import episode_state, load_episode  # noqa: E402
from clvla.evaluation import make_libero_env  # noqa: E402
from clvla.policy import build_observation, state_from_env_obs  # noqa: E402
from clvla.tasks import resolve_tasks  # noqa: E402

pytestmark = pytest.mark.sim


@pytest.fixture(scope="module")
def resolved(manifest):
    return resolve_tasks(manifest)


def _init_states(rt):
    import torch

    return torch.load(str(rt.init_states_path))


def test_all_ten_manifest_tasks_resolve_in_official_libero90(resolved, manifest):
    assert [r.spec.libero_90_index for r in resolved] == [t.libero_90_index for t in manifest.tasks]
    for r in resolved:
        assert r.libero_task.language == r.spec.instruction
        assert r.bddl_path.is_file() and r.init_states_path.is_file()
        assert _init_states(r).shape[0] == 50


def test_reset_observe_and_safe_actions(resolved, cfg):
    rt = resolved[0]
    env = make_libero_env(rt.bddl_path, cfg.simulator.camera_resolution, cfg.simulator.env_seed)
    try:
        env.reset()
        obs = env.set_init_state(_init_states(rt)[cfg.splits.test_init_states[0]])
        res = cfg.simulator.camera_resolution
        assert obs["agentview_image"].shape == (res, res, 3) and obs["agentview_image"].dtype == np.uint8
        assert obs["robot0_eye_in_hand_image"].shape == (res, res, 3)
        assert obs["robot0_eef_pos"].shape == (3,) and obs["robot0_eef_quat"].shape == (4,)
        assert obs["robot0_gripper_qpos"].shape == (2,)
        req = build_observation(obs, rt.spec.instruction, cfg.policy.resize_size)
        assert req["observation/image"].shape == (224, 224, 3) and req["observation/state"].shape == (8,)
        start = obs["robot0_eef_pos"].copy()
        for _ in range(cfg.simulator.num_steps_wait):
            obs, reward, done, info = env.step(cfg.simulator.dummy_action)
        for _ in range(5):  # zero-motion OSC deltas, gripper open: safe
            obs, reward, done, info = env.step([0.0] * 6 + [-1.0])
        assert not done
        assert np.all(np.isfinite(state_from_env_obs(obs)))
        assert np.linalg.norm(obs["robot0_eef_pos"] - start) < 0.05  # zero deltas keep the arm (almost) still
    finally:
        env.close()


@pytest.mark.slow
def test_all_ten_manifest_tasks_instantiate_and_step(resolved, cfg):
    for rt in resolved:
        env = make_libero_env(rt.bddl_path, cfg.simulator.camera_resolution, cfg.simulator.env_seed)
        try:
            env.reset()
            env.set_init_state(_init_states(rt)[cfg.splits.test_init_states[0]])
            obs, _, done, _ = env.step(cfg.simulator.dummy_action)
            assert obs["agentview_image"].shape[:2] == (256, 256), rt.spec.task_id
            assert not done, f"{rt.spec.task_id}: task reports success at its initial state"
        finally:
            env.close()


@pytest.mark.data
def test_demo_conventions_match_simulator(resolved, cfg):
    """Restore demo MuJoCo states in the official env and compare with the HDF5 fields we feed the model.

    Verifies the 8-D state layout + quaternion->axis-angle formula and the image orientation, and pins down a LIBERO
    data quirk: LIBERO's scripts/create_dataset.py stores the observation returned by env.step(actions[t]) as obs[t],
    so HDF5 obs[t] is the state AFTER actions[t] (= sim(states[t+1])), not before it as in the evaluation loop.
    """
    from pathlib import Path

    rt = resolved[0]
    ep = load_episode(Path(cfg.data.root) / rt.spec.demo_file, "demo_0")
    S, T = episode_state(ep), len(ep["actions"])
    env = make_libero_env(rt.bddl_path, cfg.data.image_size, cfg.simulator.env_seed)  # render at the HDF5 size
    try:
        env.reset()
        aligned, same_index, img_err, img_err_flipped = [], [], [], []
        for t in range(1, T, max(1, T // 12)):
            obs = env.set_init_state(ep["states"][t])
            e = state_from_env_obs(obs)
            aligned.append(np.abs(e - S[t - 1]).max())  # obs[t-1] was recorded right after reaching states[t]
            same_index.append(np.abs(e - S[t]).max())
            img = obs["agentview_image"].astype(int)
            img_err.append(np.abs(img - ep["obs/agentview_rgb"][t - 1].astype(int)).mean())
            img_err_flipped.append(np.abs(img[::-1, ::-1] - ep["obs/agentview_rgb"][t - 1].astype(int)).mean())
        print(
            f"\nstate |sim(states[t]) - obs[t-1]| max {max(aligned):.5f}; |sim(states[t]) - obs[t]| median "
            f"{np.median(same_index):.5f}; image mean abs err {np.mean(img_err):.2f} (rotated: {np.mean(img_err_flipped):.2f})"
        )
        assert max(aligned) < 3e-3, aligned  # state formula + layout identical up to sim noise
        assert np.median(same_index) > 3 * np.median(aligned)  # the one-step offset is real
        assert max(img_err) < 12 and np.mean(img_err) < 0.5 * np.mean(img_err_flipped)  # same orientation as env obs
    finally:
        env.close()
