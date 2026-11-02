"""Stage 2: one student episode in LIBERO through the existing `run_episode` loop. The reset state must be a training demo of split profile n25; reward is recorded, never used.
  scripts/py-sim scripts/opd_stage2_rollout.py --port P --out D --libero-index 35 --demo 0
"""

from __future__ import annotations

import argparse
import hashlib
import json
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


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


class RecordingEnv:
    """Passes calls through and records what the environment actually received and returned."""

    def __init__(self, env):
        self.env, self.actions, self.rewards, self.dones, self.states, self.image_sha, self.last = env, [], [], [], [], [], None

    def _keep(self, obs):
        self.last = obs
        self.states.append(np.concatenate([obs["robot0_eef_pos"], obs["robot0_eef_quat"], obs["robot0_gripper_qpos"]]))
        self.image_sha.append(sha(obs["agentview_image"]) + sha(obs["robot0_eye_in_hand_image"]))
        return obs

    def reset(self):
        return self.env.reset()

    def set_init_state(self, state):
        return self._keep(self.env.set_init_state(state))

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.actions.append(np.asarray(action, np.float64))
        self.rewards.append(float(reward))
        self.dones.append(bool(done))
        self._keep(obs)
        return obs, reward, done, info


class RecordingPolicy:
    def __init__(self, client, env):
        self.client, self.env, self.chunks, self.raw, self.requests, self.step_at_call = client, env, [], [], [], []

    def __call__(self, request):
        last = self.env.last
        self.raw.append((np.array(last["agentview_image"]), np.array(last["robot0_eye_in_hand_image"])))
        self.step_at_call.append(len(self.env.actions))
        self.requests.append({"image_sha": sha(request["observation/image"]), "wrist_sha": sha(request["observation/wrist_image"]), "state": np.asarray(request["observation/state"]).tolist()})
        chunk = self.client(request)
        self.chunks.append(chunk)
        return chunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--libero-index", type=int, default=35)
    ap.add_argument("--demo", type=int, default=0)
    ap.add_argument("--profile", default="n25")
    a = ap.parse_args()
    out = Path(a.out)
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    manifest = load_manifest(cfg.resolve(cfg.task_manifest))
    rt = next(r for r in resolve_tasks(manifest) if r.spec.libero_90_index == a.libero_index)
    dcfg = yaml.safe_load((ROOT / "configs/datasets/processed_v1.yaml").read_text())
    train, held = resolve_split(dcfg, cfg, a.profile)
    assert a.demo in train and a.demo not in held, "reset state must come from the training partition"
    proot = Path(os.environ[dcfg["output"]["processed_root_env"]]) / dcfg["version"] / "hdf5"
    with h5py.File(str(proot / (rt.spec.name + "_demo.hdf5")), "r") as f:
        init = np.asarray(f["data"]["demo_%d" % a.demo]["states"][0])
    env_seed = cfg.simulator.env_seed
    env = RecordingEnv(make_libero_env(rt.bddl_path, cfg.simulator.camera_resolution, env_seed))
    client = RemotePolicy("127.0.0.1", a.port, cfg.policy.action_dim, cfg.policy.replan_steps)
    policy = RecordingPolicy(client, env)
    t0 = time.time()
    res = run_episode(env, policy, init, rt.spec.instruction, max_steps=cfg.simulator.max_steps, num_steps_wait=cfg.simulator.num_steps_wait, replan_steps=cfg.policy.replan_steps,
                      resize_size=cfg.policy.resize_size, dummy_action=cfg.simulator.dummy_action)
    wall = time.time() - t0
    client._client.infer({"_opd_finish": True})
    np.savez_compressed(out / "environment.npz", actions=np.stack(env.actions), rewards=np.array(env.rewards), dones=np.array(env.dones), states=np.stack(env.states), image_sha=np.array(env.image_sha),
                        chunks=np.stack(policy.chunks), step_at_call=np.array(policy.step_at_call), raw_agentview=np.stack([x[0] for x in policy.raw]), raw_wrist=np.stack([x[1] for x in policy.raw]),
                        final_agentview=env.last["agentview_image"], final_wrist=env.last["robot0_eye_in_hand_image"], init_state=init)
    (out / "environment.json").write_text(json.dumps({"libero_90_index": a.libero_index, "task": rt.spec.name, "instruction": rt.spec.instruction, "reset": {"profile": a.profile, "demo": a.demo, "init_state_sha": sha(init)}, "env_seed": env_seed,
                                                     "num_steps_wait": cfg.simulator.num_steps_wait, "max_steps": cfg.simulator.max_steps, "replan_steps": cfg.policy.replan_steps, "success": res.success, "error": res.error, "policy_steps": res.steps, "policy_calls": res.policy_calls,
                                                     "env_steps": len(env.actions), "reward_sum": sum(env.rewards), "wall_s": wall, "requests": policy.requests}, indent=1))
    print(json.dumps({"success": res.success, "error": res.error, "policy_steps": res.steps, "calls": res.policy_calls, "env_steps": len(env.actions), "reward_sum": sum(env.rewards), "wall_s": wall}), flush=True)


if __name__ == "__main__":
    main()
