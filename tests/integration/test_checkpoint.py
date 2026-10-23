"""Stage 3: load the exact pi05_libero checkpoint and run inference (policy env, GPU: CLVLA_USE_GPU=1)."""

import json
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openpi", reason="policy env only")

pytestmark = [pytest.mark.policy_env, pytest.mark.checkpoint, pytest.mark.gpu, pytest.mark.slow]


@pytest.fixture(scope="module")
def loaded(cfg):
    import jax

    if not any(d.platform == "gpu" for d in jax.devices()):
        pytest.skip("no GPU visible (run with CLVLA_USE_GPU=1 after scripts/gpu_guard.sh)")
    from clvla.policy import load_official_policy

    t0 = time.time()
    policy, report = load_official_policy(
        cfg.checkpoint.openpi_config,
        cfg.checkpoint.local_dir,
        expected_asset_id=cfg.checkpoint.norm_stats_asset_id,
        state_dim=cfg.policy.state_dim,
        action_dim=cfg.policy.action_dim,
        action_horizon=cfg.policy.action_horizon,
    )
    return policy, report, time.time() - t0


def test_compat_report(loaded, cfg):
    _, report, load_s = loaded
    print(f"\nloaded {report} in {load_s:.1f}s")
    assert report.config_name == "pi05_libero" and report.asset_id == "physical-intelligence/libero"
    assert report.use_quantile_norm and report.action_horizon == 10 and "PI05" in report.model_type


def test_policy_uses_bundled_norm_stats(loaded, cfg):
    from openpi import transforms as T

    policy = loaded[0]
    normalizers = [t for t in policy._input_transform.transforms if isinstance(t, T.Normalize)]
    assert len(normalizers) == 1 and normalizers[0].use_quantiles
    bundled = json.loads(
        (Path(cfg.checkpoint.local_dir) / "assets" / cfg.checkpoint.norm_stats_asset_id / "norm_stats.json").read_text()
    )["norm_stats"]
    for key in ("state", "actions"):
        for f in ("q01", "q99"):
            assert np.allclose(getattr(normalizers[0].norm_stats[key], f), bundled[key][f]), (key, f)


def test_inference_on_valid_observations(loaded, cfg, manifest):
    from openpi.policies.libero_policy import make_libero_example
    from openpi_client import image_tools

    from clvla.data import episode_state, load_episode

    policy = loaded[0]
    out = policy.infer(make_libero_example())
    assert out["actions"].shape == (cfg.policy.action_horizon, cfg.policy.action_dim)

    # A real observation from a manifest task, preprocessed exactly as the eval client does (rotate, resize_with_pad).
    t = manifest.tasks[0]
    ep = load_episode(Path(cfg.data.root) / t.demo_file, "demo_0")

    def prep(img):
        return image_tools.convert_to_uint8(
            image_tools.resize_with_pad(np.ascontiguousarray(img[::-1, ::-1]), 224, 224)
        )

    obs = {
        "observation/image": prep(ep["obs/agentview_rgb"][0]),
        "observation/wrist_image": prep(ep["obs/eye_in_hand_rgb"][0]),
        "observation/state": episode_state(ep)[0],
        "prompt": t.instruction,
    }
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = policy.infer(obs)
        times.append(time.perf_counter() - t0)
    a = np.asarray(out["actions"])
    assert a.shape == (10, 7) and np.all(np.isfinite(a))
    assert np.all(np.abs(a[:, :6]) < 3.0)  # within a few units of LIBERO's [-1, 1] OSC range after unnormalisation
    print(f"\ninference latency (s): first {times[0]:.3f} (includes JIT), then {times[1]:.3f}, {times[2]:.3f}")
    print("first predicted action:", np.round(a[0], 3), "| demo action:", np.round(ep["actions"][0], 3))
    assert times[2] < 2.0
