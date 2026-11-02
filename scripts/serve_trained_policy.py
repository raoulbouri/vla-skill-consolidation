"""Serve a trained clvla run (LoRA specialist / smoke) with OpenPI's official websocket policy server.

  scripts/serve_trained_policy.sh <run_dir> [port] [step]      # preferred: GPU guard + deterministic XLA flags
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from clvla.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--step", type=int, default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    from openpi.serving import websocket_policy_server

    from clvla.openpi_training import load_trained_policy

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, step, ckpt = load_trained_policy(Path(args.run_dir), cfg.checkpoint.local_dir, args.step)
    meta = json.loads((Path(args.run_dir) / "run_metadata.json").read_text())
    logging.info("serving %s step %s (%s) on port %s", args.run_dir, step, ckpt, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy, host="0.0.0.0", port=args.port,
        metadata={"clvla_run": str(args.run_dir), "step": step, "task": meta["task"]["libero_90_index"]},
    ).serve_forever()


if __name__ == "__main__":
    main()
