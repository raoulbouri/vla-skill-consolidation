"""t35 diagnostics, Experiment 2b: restore the teacher's saved sim state at 25%, 50% and 75% of each successful teacher episode and let the served policy finish within the teacher's remaining step budget.
Noise for restored event e is the episode's recorded noise seed + e, the noise the teacher used there, so a teacher resume should retrace the original and checks the restore.
  scripts/py-sim scripts/opd_diag_takeover.py --port P --policy-id ID --teacher-dir DIR --out DIR [--limit N] [--any]"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from clvla.config import load_experiment_config
from clvla.evaluation import make_libero_env, run_episode
from clvla.policy import RemotePolicy
from clvla.tasks import load_manifest, resolve_tasks
from opd_diag_rollout import POSITION, Geometry, Recorder, save_episode, summary

ROOT = Path(__file__).resolve().parents[1]
FRACTIONS = (0.25, 0.5, 0.75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--policy-id", required=True)
    ap.add_argument("--teacher-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--any", action="store_true", help="smoke test only: also use failed teacher episodes")
    a = ap.parse_args()
    out, tdir = Path(a.out), Path(a.teacher_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "takeovers.jsonl"
    done = {(r["k"], r["fraction"]) for r in map(json.loads, log.read_text().splitlines())} if log.exists() else set()
    eps = [json.loads(l) for l in (tdir / "episodes.jsonl").read_text().splitlines()]
    eps = [e for e in eps if e["success"] or a.any][: a.limit]
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    rt = resolve_tasks(load_manifest(cfg.resolve(cfg.task_manifest)))[POSITION]
    geo = Geometry(make_libero_env(rt.bddl_path, cfg.simulator.camera_resolution, cfg.simulator.env_seed))
    client = RemotePolicy("127.0.0.1", a.port, cfg.policy.action_dim, cfg.policy.replan_steps)
    wait = cfg.simulator.num_steps_wait
    for e in eps:
        z = np.load(tdir / ("ep_%02d.npz" % e["k"]))
        exec_t = z["ev_t_env"] - wait
        for fr in FRACTIONS:
            if (e["k"], fr) in done:
                continue
            idx = int(np.argmax(exec_t >= fr * e["steps"]))
            t_exec = int(exec_t[idx])
            budget = cfg.simulator.max_steps - t_exec
            seed = e["noise_seed"] + idx
            client._client.infer({"_episode_mode": "ode", "_episode_seed": seed})
            rec = Recorder(client, geo, False)
            t0 = time.time()
            r = run_episode(geo, rec, z["ev_sim"][idx], rt.spec.instruction, max_steps=budget, num_steps_wait=0, replan_steps=cfg.policy.replan_steps, resize_size=cfg.policy.resize_size, dummy_action=cfg.simulator.dummy_action)
            save_episode(out / ("ep_%02d_f%02d.npz" % (e["k"], int(fr * 100))), geo.rows, rec.events, False)
            orig = z["rows"][wait + t_exec : wait + t_exec + 25, :3]
            new = np.stack(geo.rows)[: len(orig), :3] if geo.rows else np.zeros((0, 3))
            n = min(len(orig), len(new))
            row = {"policy_id": a.policy_id, "k": e["k"], "fraction": fr, "event": idx, "t_exec": t_exec, "budget": budget, "noise_seed": seed, "teacher_success": e["success"], "teacher_steps": e["steps"], "start_snap": z["ev_snap"][idx].tolist(), "success": bool(r.success), "steps": int(r.steps), "error": r.error, "first_chunk_vs_teacher_max_abs": float(np.abs(rec.events[0]["chunk"] - z["ev_chunk"][idx]).max()) if rec.events else None, "grip_dev_vs_teacher_first25_max": float(np.linalg.norm(orig[:n] - new[:n], axis=1).max()) if n else None, "runtime_s": time.time() - t0, **summary(geo.rows, 0)}
            with log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            print(a.policy_id, e["k"], fr, row["success"], row["steps"], "/", budget, round(row["first_chunk_vs_teacher_max_abs"] or -1, 4), round(row["grip_dev_vs_teacher_first25_max"] or -1, 4), flush=True)


if __name__ == "__main__":
    main()
