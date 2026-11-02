"""Reload a trained run's checkpoint with OpenPI's create_trained_policy and run inference on held-out data.

Checks: params load into the run's exact LoRA config; saved norm stats equal the base checkpoint's; inference on a
VALIDATION demo frame (preprocessed exactly like the evaluation client) returns finite (10, 7) actions; also prints the
first predicted action next to the demo action (informational). Writes <run>/reload_check.json.

  CLVLA_USE_GPU=1 scripts/py-policy scripts/check_trained_policy.py --run-dir outputs/finetune/<run> [--step N]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from clvla.assets import build_lock, checkpoint_fingerprint
from clvla.config import load_experiment_config
from clvla.policy import check_action_chunk

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args(argv)
    run_dir = Path(args.run_dir)
    out_path = run_dir / "reload_check.json"
    if out_path.exists():
        raise SystemExit(f"{out_path} exists; refusing to overwrite")
    guard = subprocess.run([str(ROOT / "scripts/gpu_guard.sh")], capture_output=True, text=True)
    if guard.returncode != 0:
        raise SystemExit(f"GPU guard refused: {guard.stdout.strip()}")

    import h5py
    import yaml
    from openpi_client import image_tools

    from clvla.openpi_training import ASSET_ID, load_trained_policy, norm_stats_equal

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    t0 = time.time()
    policy, step, ckpt = load_trained_policy(run_dir, cfg.checkpoint.local_dir, args.step)
    load_s = time.time() - t0
    saved = json.loads((ckpt / "assets" / ASSET_ID / "norm_stats.json").read_text())["norm_stats"]
    bundled = json.loads((Path(cfg.checkpoint.local_dir) / "assets" / ASSET_ID / "norm_stats.json").read_text())["norm_stats"]
    if not norm_stats_equal(saved, bundled):
        raise SystemExit("trained checkpoint norm stats differ from the base checkpoint's")

    # One validation-demo frame from the aligned processed file, preprocessed like the eval client.
    dcfg = yaml.safe_load((ROOT / meta["train_spec"]["data_config"]).read_text())
    lm = json.loads(Path(meta["datasets"]["lerobot_manifest"]).read_text())
    hdf5 = Path(os.environ[dcfg["output"]["processed_root_env"]]) / dcfg["version"] / lm["source"]["processed_file"]
    val_key = lm["splits"]["val"]["demo_keys"][0]
    with h5py.File(str(hdf5), "r") as f:
        g = f["data"][val_key]
        raw_av, raw_wr = g["obs/agentview_rgb"][0], g["obs/eye_in_hand_rgb"][0]
        state, demo_action = g["obs/state"][0], g["actions"][0]

    def prep(img):
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(img[::-1, ::-1]), 224, 224))

    obs = {"observation/image": prep(raw_av), "observation/wrist_image": prep(raw_wr),
           "observation/state": np.asarray(state, np.float64), "prompt": meta["task"]["instruction"]}
    times, actions = [], None
    for _ in range(3):
        t1 = time.perf_counter()
        actions = check_action_chunk(policy.infer(obs)["actions"], cfg.policy.action_dim, cfg.policy.replan_steps)
        times.append(time.perf_counter() - t1)
    if actions.shape != (cfg.policy.action_horizon, cfg.policy.action_dim):
        raise SystemExit(f"unexpected action shape {actions.shape}")
    result = {
        "run_dir": str(run_dir), "step": step, "checkpoint_dir": str(ckpt),
        "checkpoint_fingerprint_sha256": checkpoint_fingerprint(build_lock(ckpt)), "load_s": round(load_s, 1),
        "inference_s": [round(x, 4) for x in times], "action_shape": list(actions.shape),
        "first_action": np.round(actions[0], 4).tolist(), "demo_action": np.round(demo_action, 4).tolist(),
        "validation_frame": {"file": str(hdf5), "demo": val_key, "t": 0}, "norm_stats_identical": True, "status": "ok",
    }
    out_path.write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
