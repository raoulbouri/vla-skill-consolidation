"""Stage 2: serve the unadapted student through the Task 6 SDE sampler. No optimiser, no teacher in this process.
  CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy scripts/opd_stage2_serve.py --port P --record-dir D --base-seed S
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"

from clvla.config import load_experiment_config
from clvla.ogpo.pi05_sde_policy import SDERolloutPolicy, tree_sha
from clvla.policy import load_official_policy
from openpi.serving import websocket_policy_server

ROOT = Path(__file__).resolve().parents[1]


class Server(SDERolloutPolicy):
    def infer(self, request):
        if request.get("_opd_finish"):
            (self.record_dir / "student_sha_end.txt").write_text(tree_sha(self.state))
            return {"actions": 0}
        return super().infer(request)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--record-dir", required=True)
    ap.add_argument("--base-seed", type=int, required=True)
    a = ap.parse_args()
    assert not ({a.base_seed + i for i in range(100)} & (set(range(12000, 12008)) | set(range(13000, 13008)) | {15000}))
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, _ = load_official_policy(cfg.checkpoint.openpi_config, cfg.checkpoint.local_dir, expected_asset_id=cfg.checkpoint.norm_stats_asset_id,
                                     state_dim=cfg.policy.state_dim, action_dim=cfg.policy.action_dim, action_horizon=cfg.policy.action_horizon)
    srv = Server(policy, Path(a.record_dir), a.base_seed, metadata={"sampler": "task6_ogpo_discrete"})
    (srv.record_dir / "student_sha_start.txt").write_text(tree_sha(srv.state))
    (srv.record_dir / "server_flags.json").write_text(json.dumps({"xla_flags": os.environ["XLA_FLAGS"]}))
    websocket_policy_server.WebsocketPolicyServer(policy=srv, host="127.0.0.1", port=a.port, metadata=srv.metadata).serve_forever()


if __name__ == "__main__":
    main()
