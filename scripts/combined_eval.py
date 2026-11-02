"""Combined Stage 1/2 evaluation client: fixed held-out states demo_25..49 (profile n25), env seed 7, the existing run_episode loop. Appends one JSON line per episode; finished keys are skipped on restart.
  scripts/py-sim scripts/combined_eval.py --port P --policy-id ID --positions 0,1 --modes ode,sde --out F.jsonl [--states 25]
Noise seed = 200000 + 100000 * position + 1000 * state, plus the replanning event: the same for every policy and both modes.
"""

from __future__ import annotations

import argparse
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


class MetricEnv:
    def __init__(self, env, joint=None, geometry=False):
        self.env, self.joint = env, joint
        self.geo = None
        if geometry:
            from opd_diag_rollout import Geometry

            self.geo = Geometry(env)
        self.clear()

    def clear(self):
        self.rewards, self.actions, self.door_min = [], [], 0.0
        if self.geo is not None:
            self.geo.rows = []

    def reset(self):
        return self.env.reset()

    def set_init_state(self, s):
        return self.env.set_init_state(s)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.rewards.append(float(reward))
        self.actions.append(np.asarray(action, np.float64))
        if self.joint:
            self.door_min = min(self.door_min, float(self.env.env.sim.data.get_joint_qpos(self.joint)))
        if self.geo is not None:
            self.geo.rows.append(np.concatenate([self.geo.snapshot(), np.asarray(action, np.float64)]))
        return obs, reward, done, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--policy-id", required=True)
    ap.add_argument("--positions", required=True)
    ap.add_argument("--modes", default="ode,sde")
    ap.add_argument("--out", required=True)
    ap.add_argument("--states", type=int, default=50)
    ap.add_argument("--profile", default="n25")
    ap.add_argument("--door-joint")
    ap.add_argument("--geometry", action="store_true", help="also log handle distance, contact and door onset (opd_diag_rollout.Geometry); does not change the episode")
    a = ap.parse_args()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        done = {(r["policy_id"], r["position"], r["mode"], r["state"]) for r in map(json.loads, out.read_text().splitlines())}
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    tasks = resolve_tasks(load_manifest(cfg.resolve(cfg.task_manifest)))
    dcfg = yaml.safe_load((ROOT / "configs/datasets/processed_v1.yaml").read_text())
    _, held = resolve_split(dcfg, cfg, a.profile)
    assert list(held) == {"n25": list(range(25, 50)), "n10": list(range(10, 50))}[a.profile], "held-out split changed"
    proot = Path(os.environ[dcfg["output"]["processed_root_env"]]) / dcfg["version"] / "hdf5"
    client = RemotePolicy("127.0.0.1", a.port, cfg.policy.action_dim, cfg.policy.replan_steps)
    env_seed = cfg.simulator.env_seed
    wait = cfg.simulator.num_steps_wait
    for pos in map(int, a.positions.split(",")):
        rt = tasks[pos]
        with h5py.File(str(proot / (rt.spec.name + "_demo.hdf5")), "r") as f:
            inits = {i: np.asarray(f["data"]["demo_%d" % i]["states"][0]) for i in held}
        env = MetricEnv(make_libero_env(rt.bddl_path, cfg.simulator.camera_resolution, env_seed), a.door_joint, a.geometry)
        for mode in a.modes.split(","):
            for state in list(held)[: a.states]:
                if (a.policy_id, pos, mode, state) in done:
                    continue
                seed = 200000 + 100000 * pos + 1000 * state
                client._client.infer({"_episode_mode": mode, "_episode_seed": seed})
                env.clear()
                n_lat = len(client.latencies_s)
                t0 = time.time()
                r = run_episode(env, client, inits[state], rt.spec.instruction, max_steps=cfg.simulator.max_steps, num_steps_wait=wait, replan_steps=cfg.policy.replan_steps, resize_size=cfg.policy.resize_size, dummy_action=cfg.simulator.dummy_action)
                acts = np.stack(env.actions[wait:]) if len(env.actions) > wait else np.zeros((0, 7))
                lat = client.latencies_s[n_lat:]
                row = {"policy_id": a.policy_id, "position": pos, "libero_90_index": rt.spec.libero_90_index, "task": rt.spec.name, "mode": mode, "state": state, "env_seed": env_seed, "noise_seed": seed, "success": bool(r.success), "reward_sum": float(sum(env.rewards)), "steps": int(r.steps), "termination": "error" if r.error else ("success" if r.success else "timeout"), "error": r.error, "replans": int(r.policy_calls),
                       "nonfinite_actions": int(np.sum(~np.isfinite(acts))), "arm_abs_max": float(np.abs(acts[:, :6]).max()) if len(acts) else None, "arm_outside_input_range_fraction": float(np.mean(np.abs(acts[:, :6]) > 1.0)) if len(acts) else None, "gripper_min": float(acts[:, 6].min()) if len(acts) else None, "gripper_max": float(acts[:, 6].max()) if len(acts) else None, "latency_mean_s": float(np.mean(lat)) if lat else None, "latency_max_s": float(np.max(lat)) if lat else None, "runtime_s": time.time() - t0, "door_min": env.door_min if a.door_joint else None}
                if env.geo is not None:
                    from opd_diag_rollout import summary as geo_summary

                    row.update({"geo_" + k: v for k, v in geo_summary(env.geo.rows, wait).items()})
                with out.open("a") as f:
                    f.write(json.dumps(row) + "\n")
                print(row["policy_id"], pos, mode, state, row["success"], row["steps"], round(row["runtime_s"], 1), flush=True)


if __name__ == "__main__":
    main()
