"""Random starts use the environment's own reset from a seed, plus seeded uniform noise on the 7 arm joints (state indices 1-7) and the training demos' mean gripper opening (indices 8-9); the sampled state is checked against the held-out demo starts.
Simulator side of the LoRA OPD pilot: asks the trainer which training state to run, runs one episode through run_episode, reports it. Reward is reported for logging only.
  scripts/py-sim scripts/opd_lora_collect.py --port P --position 4"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import h5py
import numpy as np
import yaml

from clvla.config import load_experiment_config
from clvla.evaluation import make_libero_env, run_episode
from clvla.lerobot_convert import resolve_split
from clvla.policy import RemotePolicy
from clvla.tasks import load_manifest, resolve_tasks

ROOT = Path(__file__).resolve().parents[1]


class RewardEnv:
    def __init__(self, env, joint):
        self.env, self.reward, self.joint, self.door_min = env, 0.0, joint, 0.0

    def reset(self):
        self.reward, self.door_min = 0.0, 0.0
        return self.env.reset()

    def set_init_state(self, s):
        return self.env.set_init_state(s)

    def step(self, a):
        o, r, d, i = self.env.step(a)
        self.reward += float(r)
        self.door_min = min(self.door_min, float(self.env.env.sim.data.get_joint_qpos(self.joint)))
        return o, r, d, i


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--position", type=int, required=True)
    ap.add_argument("--profile", default="n25")
    ap.add_argument("--random-starts", action="store_true")
    ap.add_argument("--arm-noise", type=float, default=0.0)
    ap.add_argument("--door-joint", default="microwave_1_microjoint")
    a = ap.parse_args()
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    rt = resolve_tasks(load_manifest(cfg.resolve(cfg.task_manifest)))[a.position]
    dcfg = yaml.safe_load((ROOT / "configs/datasets/processed_v1.yaml").read_text())
    train, held = resolve_split(dcfg, cfg, a.profile)
    proot = Path(os.environ[dcfg["output"]["processed_root_env"]]) / dcfg["version"] / "hdf5"
    with h5py.File(str(proot / (rt.spec.name + "_demo.hdf5")), "r") as f:
        inits = {i: np.asarray(f["data"]["demo_%d" % i]["states"][0]) for i in train}
        held_states = np.stack([np.asarray(f["data"]["demo_%d" % i]["states"][0]) for i in held])
    train_grip = np.stack([inits[i][8:10] for i in train]).mean(axis=0)
    env = RewardEnv(make_libero_env(rt.bddl_path, cfg.simulator.camera_resolution, cfg.simulator.env_seed), a.door_joint)
    client = RemotePolicy("127.0.0.1", a.port, cfg.policy.action_dim, cfg.policy.replan_steps)
    while True:
        msg = client._client.infer({"_ctl": "next"})
        if msg.get("stop"):
            return
        if msg.get("wait"):
            time.sleep(1.0)
            continue
        demo = int(msg["demo"])
        if a.random_starts:
            np.random.seed(demo)
            env.env.seed(demo)
            env.env.reset()
            start = np.asarray(env.env.get_sim_state()).copy()
            offset = np.random.RandomState(demo + 2).uniform(-a.arm_noise, a.arm_noise, 7)
            start[1:8] = start[1:8] + offset
            start[8:10] = train_grip
            arm_offset_max = float(np.abs(offset).max())
            np.random.seed(demo + 1)
        else:
            assert demo in train and demo not in held, "training rollouts must use training states"
            start = inits[demo]
            arm_offset_max = 0.0
        min_l2 = float(np.min(np.linalg.norm(held_states - start, axis=1)))
        assert min_l2 > 1e-3, "training start coincides with a held-out start"
        t0 = time.time()
        r = run_episode(env, client, start, rt.spec.instruction, max_steps=cfg.simulator.max_steps, num_steps_wait=cfg.simulator.num_steps_wait, replan_steps=cfg.policy.replan_steps, resize_size=cfg.policy.resize_size, dummy_action=cfg.simulator.dummy_action)
        client._client.infer({"_ctl": "done", "success": bool(r.success), "steps": int(r.steps), "calls": int(r.policy_calls), "reward": float(env.reward), "error": r.error, "seconds": time.time() - t0, "door_min": env.door_min, "start_sha": hashlib.sha256(start.tobytes()).hexdigest()[:16], "start_min_l2_to_held_out": min_l2, "arm_offset_max": arm_offset_max})


if __name__ == "__main__":
    main()
