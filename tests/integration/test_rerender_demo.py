"""Re-render one real demo of task 78 in the official env and run every reconstruction audit (sim env, data)."""

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("libero", reason="simulator env only")

from clvla.data import DEMO_SCHEMA, episode_state  # noqa: E402
from clvla.evaluation import make_libero_env  # noqa: E402
from clvla.rerender import Thresholds, rerender_episode  # noqa: E402
from clvla.tasks import resolve_tasks  # noqa: E402

pytestmark = [pytest.mark.sim, pytest.mark.data, pytest.mark.slow]


def test_rerender_one_demo_aligned_and_reconstructed(cfg, manifest):
    import h5py

    rt = resolve_tasks(manifest)[0]
    assert rt.spec.libero_90_index == 78
    with h5py.File(str(Path(cfg.data.root) / rt.spec.demo_file), "r") as f:
        g = f["data"]["demo_0"]
        ep = {k: g[k][()] for k in DEMO_SCHEMA}
        init = g.attrs.get("init_state")
    env = make_libero_env(rt.bddl_path, 256, cfg.simulator.env_seed)
    env_small = make_libero_env(rt.bddl_path, 128, cfg.simulator.env_seed)
    try:
        arrays, rep = rerender_episode(env, env_small, ep, init, Thresholds(image_check_every=5))
    finally:
        env.close()
        env_small.close()
    T = len(ep["actions"])
    print("\n", rep)
    assert rep["offset"] == 1 and rep["offset_median_residuals"]["1"] < rep["offset_median_residuals"]["0"] / 3
    assert "suspect" in rep["replay"] and rep["replay"]["qpos_median"] < 0.05  # recorded diagnostics, not a gate
    assert "success" in rep["final"]
    assert arrays["obs/agentview_rgb"].shape == (T, 256, 256, 3) and arrays["obs/eye_in_hand_rgb"].dtype == np.uint8
    assert np.array_equal(arrays["actions"], ep["actions"]) and np.array_equal(arrays["states"], ep["states"])
    assert np.array_equal(arrays["dones"], ep["dones"]) and np.array_equal(arrays["rewards"], ep["rewards"])
    # pre-action alignment: the re-rendered state at k+1 equals the HDF5 (post-action) state at k
    assert np.abs(arrays["obs/state"][1:] - episode_state(ep)[:-1]).max() < 3e-3
    assert rep["images"]["mae_mean"] < 0.5 * rep["images"]["mae_mean_rotated"]
