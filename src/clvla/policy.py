"""Thin adapter around the official OpenPI policy. Nothing here reimplements the model or its transforms.

Two halves, one per environment:
  * sim env   (py3.8): `build_observation` turns a raw LIBERO env observation into the request dict that OpenPI's
                       LIBERO client sends (identical to third_party/openpi/examples/libero/main.py), and
                       `RemotePolicy` sends it to the official websocket policy server.
  * policy env (py3.11): `load_official_policy` calls openpi.policies.policy_config.create_trained_policy, which
                       loads the checkpoint's bundled norm stats; `check_checkpoint_compat` verifies config/checkpoint
                       agreement first and fails with actionable messages.
"""

from __future__ import annotations

import dataclasses
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from clvla.assets import check_norm_stats, load_norm_stats_json


class CompatibilityError(RuntimeError):
    pass


# ---------------------------------------------------------------------------------------------------------------------
# Observation preprocessing (verbatim semantics of OpenPI examples/libero/main.py at the pinned commit)


def quat2axisangle(quat: np.ndarray) -> np.ndarray:
    """Copy of OpenPI main.py `_quat2axisangle` (itself copied from robosuite). Quaternion is (x, y, z, w)."""
    quat = np.array(quat, dtype=np.float64)  # main.py clips in place; copy so the caller's obs is untouched
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def state_from_env_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))


def build_observation(obs: Dict[str, np.ndarray], instruction: str, resize_size: int) -> Dict[str, Any]:
    """Raw LIBERO env obs -> OpenPI request: rotate 180, resize_with_pad to resize_size, uint8, 8-D state, prompt."""
    from openpi_client import image_tools

    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize_size, resize_size))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize_size, resize_size))
    return {
        "observation/image": img,
        "observation/wrist_image": wrist,
        "observation/state": state_from_env_obs(obs),
        "prompt": str(instruction),
    }


def check_action_chunk(actions: Any, action_dim: int, min_len: int) -> np.ndarray:
    a = np.asarray(actions)
    if a.ndim != 2 or a.shape[1] != action_dim:
        raise CompatibilityError(f"policy returned actions of shape {a.shape}, expected (H, {action_dim})")
    if a.shape[0] < min_len:
        raise CompatibilityError(f"policy returned {a.shape[0]} actions; need >= replan_steps={min_len}")
    if not np.all(np.isfinite(a)):
        raise CompatibilityError("policy returned non-finite actions")
    return a


class RemotePolicy:
    """Client for OpenPI's official websocket policy server (scripts/serve_policy.py). Records inference timings."""

    def __init__(self, host: str, port: int, action_dim: int, min_chunk: int):
        from openpi_client import websocket_client_policy

        self._client = websocket_client_policy.WebsocketClientPolicy(host, port)
        self.action_dim, self.min_chunk = action_dim, min_chunk
        self.latencies_s: List[float] = []

    @property
    def server_metadata(self) -> Dict[str, Any]:
        return dict(self._client.get_server_metadata() or {})

    def __call__(self, request: Dict[str, Any]) -> np.ndarray:
        t0 = time.perf_counter()
        out = self._client.infer(request)
        self.latencies_s.append(time.perf_counter() - t0)
        return check_action_chunk(out["actions"], self.action_dim, self.min_chunk)


# ---------------------------------------------------------------------------------------------------------------------
# Checkpoint / config compatibility (policy env)


@dataclasses.dataclass(frozen=True)
class CompatReport:
    config_name: str
    checkpoint_dir: str
    asset_id: str
    model_type: str
    action_horizon: int
    model_action_dim: int
    use_quantile_norm: bool


def check_checkpoint_compat(
    train_config: Any,
    data_config: Any,
    checkpoint_dir: str,
    *,
    expected_config: str,
    expected_asset_id: str,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
) -> CompatReport:
    """Check that an OpenPI TrainConfig + DataConfig match the checkpoint on disk and the frozen protocol.

    Duck-typed (needs only .name, .model.{action_horizon,action_dim,model_type}, data_config.{asset_id,
    use_quantile_norm}) so it is unit-testable without OpenPI.
    """
    ckpt = Path(checkpoint_dir)
    errs = []
    if train_config.name != expected_config:
        errs.append(
            f"OpenPI config is {train_config.name!r} but the protocol requires {expected_config!r} "
            "(do not substitute pi05_droid / pi0 configs)"
        )
    if not (ckpt / "params").is_dir():
        errs.append(f"{ckpt}/params missing: not an OpenPI JAX checkpoint directory (or download incomplete)")
    if data_config.asset_id != expected_asset_id:
        errs.append(f"config asset_id {data_config.asset_id!r} != expected norm-stats asset {expected_asset_id!r}")
    ns = ckpt / "assets" / str(data_config.asset_id) / "norm_stats.json"
    if not ns.is_file():
        errs.append(f"bundled norm stats not found at {ns}; the checkpoint's own stats are required (never recompute)")
    else:
        errs.extend(
            check_norm_stats(
                load_norm_stats_json(ns), state_dim, action_dim, quantiles=bool(data_config.use_quantile_norm)
            )
        )
    model = train_config.model
    if model.action_horizon != action_horizon:
        errs.append(f"model action_horizon {model.action_horizon} != protocol {action_horizon}")
    if model.action_dim < action_dim:
        errs.append(f"model action_dim {model.action_dim} < LIBERO action dim {action_dim}")
    if errs:
        raise CompatibilityError("checkpoint/config incompatible:\n  - " + "\n  - ".join(errs))
    return CompatReport(
        config_name=train_config.name,
        checkpoint_dir=str(ckpt),
        asset_id=str(data_config.asset_id),
        model_type=str(model.model_type),
        action_horizon=model.action_horizon,
        model_action_dim=model.action_dim,
        use_quantile_norm=bool(data_config.use_quantile_norm),
    )


def openpi_configs(config_name: str):
    """(TrainConfig, DataConfig) from the pinned OpenPI. Policy env only."""
    from openpi.training import config as _config

    tc = _config.get_config(config_name)
    return tc, tc.data.create(tc.assets_dirs, tc.model)


def load_official_policy(
    config_name: str,
    checkpoint_dir: str,
    *,
    expected_asset_id: str,
    state_dim: int,
    action_dim: int,
    action_horizon: int,
    sample_kwargs: Optional[Dict[str, Any]] = None,
):
    """Compat-check, then load with OpenPI's create_trained_policy (norm stats from the checkpoint's assets/)."""
    from openpi.policies import policy_config

    tc, dc = openpi_configs(config_name)
    report = check_checkpoint_compat(
        tc,
        dc,
        checkpoint_dir,
        expected_config=config_name,
        expected_asset_id=expected_asset_id,
        state_dim=state_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
    )
    policy = policy_config.create_trained_policy(tc, checkpoint_dir, sample_kwargs=sample_kwargs)
    return policy, report


def openpi_training_transforms(config_name: str, checkpoint_dir: str):
    """The exact input transform chain OpenPI applies to training samples (data_loader.transform_dataset), with the
    checkpoint's bundled norm stats. Used to push real demos through the intended preprocessing path."""
    from openpi import transforms as _transforms
    from openpi.training import checkpoints as _checkpoints

    _, dc = openpi_configs(config_name)
    norm_stats = _checkpoints.load_norm_stats(Path(checkpoint_dir) / "assets", dc.asset_id)
    return (
        _transforms.compose(
            [
                *dc.repack_transforms.inputs,
                *dc.data_transforms.inputs,
                _transforms.Normalize(norm_stats, use_quantiles=dc.use_quantile_norm),
                *dc.model_transforms.inputs,
            ]
        ),
        dc,
        norm_stats,
    )
