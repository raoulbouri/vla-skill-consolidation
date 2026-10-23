"""Stage 4: official OpenPI websocket server <-> client round trip (sim env runs the client; server on CLVLA_GPU)."""

import numpy as np
import pytest

pytest.importorskip("openpi_client", reason="needs openpi_client")

from clvla.policy import RemotePolicy, build_observation  # noqa: E402

pytestmark = [pytest.mark.server, pytest.mark.gpu, pytest.mark.checkpoint, pytest.mark.slow]


def test_one_request_response(policy_server, cfg):
    client = RemotePolicy(policy_server["host"], policy_server["port"], cfg.policy.action_dim, cfg.policy.replan_steps)
    meta = client.server_metadata
    rng = np.random.default_rng(0)
    raw = {
        "agentview_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eye_in_hand_image": rng.integers(0, 255, (256, 256, 3), dtype=np.uint8),
        "robot0_eef_pos": np.array([-0.05, 0.0, 1.0]),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0]),
        "robot0_gripper_qpos": np.array([0.03, -0.03]),
    }
    actions = client(build_observation(raw, "open the microwave", cfg.policy.resize_size))
    actions = client(build_observation(raw, "open the microwave", cfg.policy.resize_size))
    print(
        f"\nserver startup {policy_server['startup_s']}s; metadata {meta}; latencies {np.round(client.latencies_s, 3)}"
    )
    assert actions.shape == (cfg.policy.action_horizon, cfg.policy.action_dim)
    assert np.all(np.isfinite(actions))
