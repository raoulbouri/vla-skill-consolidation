"""Combined Stage 1/2 evaluation server: one policy, ODE or SDE per episode, noise seeded per episode and replanning event so policies can be paired.
  CLVLA_USE_GPU=1 scripts/py-policy scripts/combined_serve.py --kind base|teacher|student --port P [--run-dir D] [--adapter F]
ODE goes through OpenPI's own Policy.infer with explicit noise. SDE uses the frozen Task 6 sampler; there is no fallback between them.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"

import jax
import numpy as np
from clvla.config import load_experiment_config
from clvla.ogpo import pi05_sde as sde
from clvla.ogpo.pi05_sde_policy import event_noise, make_sampler, model_inputs, model_of
from clvla.openpi_training import load_trained_policy
from clvla.policy import load_official_policy
from openpi.serving import websocket_policy_server

ROOT = Path(__file__).resolve().parents[1]
cfg = sde.SDEConfig()


def load_student_policy(conf, adapter):
    from opd_lora_student import build_student_policy

    return build_student_policy(conf, np.load(adapter))


class Matched:
    def __init__(self, policy):
        self.policy, self.metadata, self.mode, self.seed, self.event = policy, {}, None, None, 0
        self.run, _ = make_sampler(model_of(policy), cfg)

    def infer(self, request):
        if "_episode_mode" in request:
            self.mode, self.seed, self.event = str(request["_episode_mode"]), int(request["_episode_seed"]), 0
            return {"actions": 0}
        x0, eps = event_noise(self.seed + self.event, cfg)
        self.event += 1
        if self.mode == "ode":
            return {"actions": np.asarray(self.policy.infer(request, noise=x0)["actions"])}
        if self.mode != "sde":
            raise ValueError(f"unknown mode {self.mode!r}")
        obs, inputs = model_inputs(self.policy, request)
        out = jax.block_until_ready(self.run(obs, x0, eps))
        return {"actions": np.asarray(self.policy._output_transform({"state": np.asarray(inputs["state"]), "actions": np.asarray(out["chain"])[0, cfg.num_steps]})["actions"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--run-dir")
    ap.add_argument("--adapter")
    ap.add_argument("--teacher-delta")
    a = ap.parse_args()
    conf = load_experiment_config(ROOT / "configs/experiment.yaml")
    if a.kind == "base":
        policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    elif a.kind == "teacher_delta":
        from opd_lora_student import build_delta_teacher_policy

        policy = build_delta_teacher_policy(conf, Path(a.teacher_delta))
    elif a.kind == "teacher":
        policy, _, _ = load_trained_policy(Path(a.run_dir), conf.checkpoint.local_dir)
    else:
        policy = load_student_policy(conf, a.adapter)
    websocket_policy_server.WebsocketPolicyServer(policy=Matched(policy), host="127.0.0.1", port=a.port, metadata={"kind": a.kind}).serve_forever()


if __name__ == "__main__":
    main()
