"""t35 diagnostics, Experiment 1: rollouts from 40 diagnostic starts with per-step geometry; with --save-obs also every replanning event's observation, sim state and action chunk, for Experiments 2 and 3.
Starts use seeds 950000-950039, built exactly like the training random starts (environment reset, seeded arm noise 0.04 rad, training demos' mean gripper opening), and are checked against the held-out demo starts.
  scripts/py-sim scripts/opd_diag_rollout.py --port P --policy-id ID --out DIR [--save-obs] [--starts 40]
Noise seed 300000 (--noise-seed0) + 1000 * k for start k, plus the replanning event: the same for every policy."""

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
POSITION, ARM_NOISE, START_SEED0, NOISE_SEED0 = 4, 0.04, 950000, 300000
COLS = ["grip_x", "grip_y", "grip_z", "handle_x", "handle_y", "handle_z", "door_q", "grip_handle_dist", "contact", "gripper_q0", "gripper_q1", "a0", "a1", "a2", "a3", "a4", "a5", "a6"]


class Geometry:
    """LIBERO env wrapper: after every step records grip-site and handle (capsule microwave_1_g14) positions, door angle, any robot-door contact, gripper joints and the executed action."""

    def __init__(self, env):
        self.env = env
        m = env.env.sim.model
        door = m.body_name2id("microwave_1_microdoorroot")
        self.door_geoms = {g for g in range(m.ngeom) if m.geom_bodyid[g] == door}
        self.robot_geoms = {g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith(("robot0", "gripper0"))}
        self.handle, self.site = m.geom_name2id("microwave_1_g14"), m.site_name2id("gripper0_grip_site")
        self.qadr = m.jnt_qposadr[m.joint_name2id("microwave_1_microjoint")]
        self.fingers = [m.jnt_qposadr[m.joint_name2id(n)] for n in ("gripper0_finger_joint1", "gripper0_finger_joint2")]
        self.rows = []

    def snapshot(self):
        d = self.env.env.sim.data
        contact = False
        for i in range(d.ncon):
            c = d.contact[i]
            if (c.geom1 in self.door_geoms and c.geom2 in self.robot_geoms) or (c.geom2 in self.door_geoms and c.geom1 in self.robot_geoms):
                contact = True
                break
        grip, handle = d.site_xpos[self.site].copy(), d.geom_xpos[self.handle].copy()
        return np.concatenate([grip, handle, [float(d.qpos[self.qadr]), float(np.linalg.norm(grip - handle)), float(contact)], [float(d.qpos[j]) for j in self.fingers]])

    def sim_state(self):
        return np.asarray(self.env.get_sim_state()).copy()

    def reset(self):
        self.rows = []
        return self.env.reset()

    def set_init_state(self, s):
        return self.env.set_init_state(s)

    def step(self, action):
        out = self.env.step(action)
        self.rows.append(np.concatenate([self.snapshot(), np.asarray(action, np.float64)]))
        return out


class Recorder:
    """Policy callable for run_episode: forwards to the server and keeps each event's observation, sim state, geometry and returned chunk."""

    def __init__(self, client, geo, save_obs):
        self.client, self.geo, self.save_obs, self.events = client, geo, save_obs, []

    def __call__(self, request):
        sim, snap, t = self.geo.sim_state(), self.geo.snapshot(), len(self.geo.rows)
        chunk = self.client(request)
        ev = {"t_env": t, "snap": snap, "chunk": np.asarray(chunk, np.float64)}
        if self.save_obs:
            ev.update(sim=sim, image=np.asarray(request["observation/image"]), wrist=np.asarray(request["observation/wrist_image"]), state=np.asarray(request["observation/state"], np.float64))
        self.events.append(ev)
        return chunk


def save_episode(path, rows, events, save_obs):
    z = {"rows": np.stack(rows) if rows else np.zeros((0, len(COLS))), "cols": np.array(COLS), "ev_t_env": np.array([e["t_env"] for e in events]), "ev_snap": np.stack([e["snap"] for e in events]), "ev_chunk": np.stack([e["chunk"] for e in events])}
    if save_obs:
        z.update(ev_sim=np.stack([e["sim"] for e in events]), ev_image=np.stack([e["image"] for e in events]), ev_wrist=np.stack([e["wrist"] for e in events]), ev_state=np.stack([e["state"] for e in events]))
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **z)
    os.replace(tmp, path)


def summary(rows, wait):
    r = np.stack(rows[wait:]) if len(rows) > wait else np.zeros((0, len(COLS)))
    if not len(r):
        return {}
    dist, q, con = r[:, 7], r[:, 6], r[:, 8] > 0
    hit = lambda m: int(np.argmax(m)) if m.any() else None
    return {"min_dist": float(dist.min()), "t_min_dist": int(dist.argmin()), "first_near": hit(dist <= 0.12), "first_contact": hit(con), "contact_steps": int(con.sum()), "door_onset": hit(q <= -0.05), "door_min": float(q.min())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--policy-id", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--starts", type=int, default=40)
    ap.add_argument("--save-obs", action="store_true")
    ap.add_argument("--start-seed0", type=int, default=START_SEED0, help="diagnostics 950000; curriculum development set 960000")
    ap.add_argument("--noise-seed0", type=int, default=NOISE_SEED0)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "episodes.jsonl"
    done = {json.loads(l)["k"] for l in log.read_text().splitlines()} if log.exists() else set()
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    rt = resolve_tasks(load_manifest(cfg.resolve(cfg.task_manifest)))[POSITION]
    dcfg = yaml.safe_load((ROOT / "configs/datasets/processed_v1.yaml").read_text())
    train, held = resolve_split(dcfg, cfg, "n10")
    assert list(held) == list(range(10, 50)), "held-out split changed"
    proot = Path(os.environ[dcfg["output"]["processed_root_env"]]) / dcfg["version"] / "hdf5"
    with h5py.File(str(proot / (rt.spec.name + "_demo.hdf5")), "r") as f:
        train_grip = np.stack([np.asarray(f["data"]["demo_%d" % i]["states"][0])[8:10] for i in train]).mean(axis=0)
        held_states = np.stack([np.asarray(f["data"]["demo_%d" % i]["states"][0]) for i in held])
    geo = Geometry(make_libero_env(rt.bddl_path, cfg.simulator.camera_resolution, cfg.simulator.env_seed))
    client = RemotePolicy("127.0.0.1", a.port, cfg.policy.action_dim, cfg.policy.replan_steps)
    wait = cfg.simulator.num_steps_wait
    for k in range(a.starts):
        if k in done:
            continue
        demo = a.start_seed0 + k
        np.random.seed(demo)
        geo.env.seed(demo)
        geo.env.reset()
        start = geo.sim_state()
        offset = np.random.RandomState(demo + 2).uniform(-ARM_NOISE, ARM_NOISE, 7)
        start[1:8] = start[1:8] + offset
        start[8:10] = train_grip
        np.random.seed(demo + 1)
        min_l2 = float(np.min(np.linalg.norm(held_states - start, axis=1)))
        assert min_l2 > 1e-3, "diagnostic start coincides with a held-out start"
        seed = a.noise_seed0 + 1000 * k
        client._client.infer({"_episode_mode": "ode", "_episode_seed": seed})
        rec = Recorder(client, geo, a.save_obs)
        t0 = time.time()
        r = run_episode(geo, rec, start, rt.spec.instruction, max_steps=cfg.simulator.max_steps, num_steps_wait=wait, replan_steps=cfg.policy.replan_steps, resize_size=cfg.policy.resize_size, dummy_action=cfg.simulator.dummy_action)
        save_episode(out / ("ep_%02d.npz" % k), geo.rows, rec.events, a.save_obs)
        row = {"policy_id": a.policy_id, "k": k, "start_seed": demo, "noise_seed": seed, "wait": wait, "success": bool(r.success), "steps": int(r.steps), "events": len(rec.events), "error": r.error, "start_min_l2_to_held_out": min_l2, "arm_offset_max": float(np.abs(offset).max()), "start": start.tolist(), "runtime_s": time.time() - t0, **summary(geo.rows, wait)}
        with log.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(a.policy_id, k, row["success"], row["steps"], round(row.get("min_dist", -1), 3), round(row.get("door_min", 0), 3), round(row["runtime_s"], 1), flush=True)


if __name__ == "__main__":
    main()
