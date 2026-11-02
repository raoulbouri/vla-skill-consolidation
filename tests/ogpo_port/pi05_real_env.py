"""Shared real-model fixture logic for Task 7+ (Tasks 5-6 keep their own frozen copies).
Loads pi05_libero via clvla.policy.load_official_policy and turns three real frames of the aligned 256 px HDF5 into model
Observations through the LIBERO evaluation path (build_observation + the policy's own input transform)."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ((78, "demo_0", 40), (35, "demo_3", 20), (42, "demo_5", 60))


def load_real_env():
    import h5py
    import jax
    import jax.numpy as jnp

    from clvla.config import load_experiment_config
    from clvla.policy import build_observation, load_official_policy
    from clvla.tasks import load_manifest

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    manifest = load_manifest(cfg.resolve(cfg.task_manifest))
    policy, _ = load_official_policy(
        cfg.checkpoint.openpi_config, cfg.checkpoint.local_dir, expected_asset_id=cfg.checkpoint.norm_stats_asset_id,
        state_dim=cfg.policy.state_dim, action_dim=cfg.policy.action_dim, action_horizon=cfg.policy.action_horizon)
    model = next(v for v in vars(policy).values() if hasattr(v, "embed_prefix") and hasattr(v, "embed_suffix"))
    proc = Path(os.environ["CLVLA_PROCESSED_ROOT"]) / "libero90_aligned256_v1" / "hdf5"
    by_idx = {t.libero_90_index: t for t in manifest.tasks}
    dicts = []
    for idx, demo, t in FRAMES:
        task = by_idx[idx]
        with h5py.File(proc / f"{task.name}_demo.hdf5", "r") as f:
            g = f["data"][demo]["obs"]
            raw = {"agentview_image": g["agentview_rgb"][t], "robot0_eye_in_hand_image": g["eye_in_hand_rgb"][t],
                   "robot0_eef_pos": g["ee_pos"][t], "robot0_eef_quat": g["ee_quat"][t],
                   "robot0_gripper_qpos": g["gripper_qpos"][t]}
        dicts.append(policy._input_transform(build_observation(raw, task.instruction, cfg.policy.resize_size)))
    keep = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")

    def observation(ids):
        from openpi.models import model as _model

        ds = [{k: dicts[i][k] for k in keep} for i in ids]
        return _model.Observation.from_dict(jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *ds))

    return {"cfg": cfg, "policy": policy, "model": model, "observation": observation}
