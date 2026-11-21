"""Pre-flight for the 5-task continual OPD run, CPU part (read-only; writes only to --out).
  PYTHONPATH=scripts:tests/ogpo_port scripts/py-policy scripts/cl_opd_preflight_cpu.py --out outputs/cl_opd/analysis/preflight
B1 task anatomy from the training demos (demo_0..24 only), D handoff Phase 1 -> Phase 2 from the logged runs,
E3 driver dry-runs for tasks 0..4 with clipping on, E4 old-task buffer coverage, E5 GPU-hour and disk projection.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
PROC = Path("/mnt/data/users/bbouri/IRL-project-data/processed/libero90_aligned256_v1/hdf5")
RUNS = {"t035 (task 0)": "outputs/cl_opd/full_20261108T214604Z/k0_t035/student", "t063 replay": "outputs/cl_opd/full_20261108T214604Z/k1_t063/student",
        "t063 no-replay": "outputs/cl_opd/t063_noreplay_20261109T150326Z/k1_t063/student"}
TRAIN_DEMOS = range(25)  # profile n25: demo_0..24 train; demo_25..49 held out, never read here


def task_files(cfg):
    out = {}
    for t in cfg["tasks"]:
        name = Path(t["teacher"]).name.split("_", 1)[1]
        out[t["id"]] = PROC / f"{name}_demo.hdf5"
    return out


def anatomy(cfg):
    res = {}
    for tid, f in task_files(cfg).items():
        with h5py.File(f, "r") as h:
            L, tog, closes, near = [], [], [], []
            for i in TRAIN_DEMOS:
                a = h["data"][f"demo_{i}"]["actions"][()]
                g = np.sign(a[:, 6])
                ch = np.flatnonzero(np.diff(g) != 0)
                L.append(len(a))
                tog.append(len(ch))
                closes.append(int(np.sum((g[1:] > 0) & (g[:-1] <= 0))))
                m = np.zeros(len(a), bool)
                for c in ch:
                    m[max(0, c - 5):c + 6] = True
                near.append(float(m.mean()))
        res[tid] = {"len_median": float(np.median(L)), "len_p90": float(np.percentile(L, 90)), "len_max": int(max(L)), "gripper_toggles_median": float(np.median(tog)),
                    "gripper_closes_median": float(np.median(closes)), "frac_steps_within_5_of_toggle": float(np.mean(near)), "policy_queries_median": float(np.median(L)) / 5}
    return res


def rows(student):
    out = []
    for f in ("training_metrics.csv", "training_metrics_from_u100.csv"):
        p = ROOT / student / f
        if p.exists():
            out += list(csv.DictReader(open(p)))
    return {int(r["update"]): r for r in out}


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def handoff():
    res = {}
    for name, st in RUNS.items():
        h = rows(st)
        win = lambda a, b, k: float(np.nanmean([fnum(h[u][k]) for u in range(a, b + 1) if u in h]))  # noqa: E731
        r = {}
        for lab, (a, b) in {"p1_last10": (91, 100), "p2_first10": (101, 110), "p2_111_150": (111, 150), "p2_151_250": (151, 250), "p2_251_400": (251, 400)}.items():
            r[lab] = {k: win(a, b, k) for k in ("kl_mean", "act_grip_agree", "act_xyz", "rollout_success", "grad_norm")}
            r[lab]["rollouts"] = win(a, b, "rollouts")
            r[lab]["spikes_gt_1e4"] = int(sum(fnum(h[u]["grad_norm"]) > 1e4 for u in range(a, b + 1) if u in h))
        p1 = [h[u] for u in range(1, 101) if u in h]
        r["phase1_teacher_tries_mean"] = float(np.mean([fnum(x.get("teacher_tries")) for x in p1]))
        r["phase1_rollout_success_rate"] = float(np.mean([fnum(x["rollout_success"]) for x in p1]))
        p2 = [h[u] for u in range(101, 401) if u in h]
        sr = np.array([fnum(x["rollout_success"]) / max(fnum(x["rollouts"]), 1) for x in p2])
        r["phase2_student_success_by_50"] = [float(np.nanmean(sr[i:i + 50])) for i in range(0, len(sr), 50)]
        res[name] = r
    return res


def dry_runs(cfg_path, tmp):
    c = yaml.safe_load(Path(cfg_path).read_text())
    c["per_task"]["traj_clip_k"] = 3.0
    f = tmp / "cl_opd_clip3_dry.yaml"
    f.write_text(yaml.safe_dump(c))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "DRYRUN": "1"}
    out = {}
    for k in range(len(c["tasks"])):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/cl_opd_run.py"), "--config", str(f), "--run", "/nonexistent/cl_preflight_dry", "--task", str(k), "--dry-run"], capture_output=True, text=True, env=env, cwd=ROOT)
        cmds = [l for l in r.stdout.splitlines() if "opd_lora_train.py" in l]
        out[str(k)] = {"rc": r.returncode, "stderr_tail": r.stderr[-400:], "n_trainer_cmds": len(cmds), "clip_in_all": all("--traj-clip-k 3.0" in l for l in cmds) and bool(cmds),
                       "old_per_update": [l.split("--old-per-update ")[1].split()[0] if "--old-per-update" in l else None for l in cmds], "init_master": all(("--init-master" in l) == (k > 0) for l in cmds),
                       "eval_lines": [l.strip() for l in r.stdout.splitlines() if l.strip().startswith(f"stage{k}")]}
    f.unlink()
    return out


def coverage(cfg):
    o, p = cfg["old_tasks"], cfg["per_task"]
    n_upd = p["total_updates"]
    res = {}
    for k in range(1, len(cfg["tasks"])):
        share = o["buffer_total"] // k
        per_upd = 40 / k  # cl_opd.yaml per_update = 40, split over the k old tasks
        res[f"task{k}"] = {"old_tasks": k, "states_per_old_task": share, "old_obs_per_update_per_task": per_upd, "visits_per_state_during_task": n_upd * per_upd / share,
                           "old_traj_share_of_update_w0.5": 0.5}
    man = {}
    bdir = ROOT / "outputs/cl_opd/full_20261108T214604Z/buffer"
    for t in ("t035", "t063"):
        ph = {}
        for line in open(bdir / t / "manifest.jsonl"):
            e = json.loads(line)
            ph[str(e.get("phase"))] = ph.get(str(e.get("phase")), 0) + 1
        man[t] = ph
    return {"schedule": res, "existing_buffer_phase_counts": man}


def du(p):
    return int(subprocess.run(["du", "-sb", str(p)], capture_output=True, text=True).stdout.split()[0]) if Path(p).exists() else None


def budget(cfg, hand):
    st = ROOT / RUNS["t063 replay"]
    h = rows(RUNS["t063 replay"])
    s1 = np.median([fnum(h[u]["seconds"]) for u in range(2, 101) if u in h])
    s2 = np.median([fnum(h[u]["seconds"]) for u in range(101, 401) if u in h])
    per_task_h = (100 * s1 + 300 * s2) / 3600
    ev = ROOT / "outputs/cl_opd/full_20261108T214604Z/k0_t035/logs/eval.log"
    ev_h = (ev.stat().st_mtime - ev.stat().st_ctime) / 3600 if ev.exists() else None
    disk = {"batches_GB": (du(st / "batches") or 0) / 1e9, "checkpoints_GB": (du(st / "checkpoints") or 0) / 1e9, "state_GB": (du(st / "state") or 0) / 1e9, "teacher_episodes_GB": (du(st.parent / "teacher_episodes") or 0) / 1e9}
    free = subprocess.run(["df", "-B1", str(ROOT / "outputs")], capture_output=True, text=True).stdout.splitlines()[-1].split()
    return {"median_update_s_phase1": s1, "median_update_s_phase2": s2, "train_hours_per_task_no_clip": per_task_h, "train_hours_per_task_clip_+10pct": per_task_h * 1.1,
            "train_hours_tasks1_4_clip": 4 * per_task_h * 1.1, "eval_log_span_hours_task0_(rough)": ev_h, "disk_per_task_GB": disk, "disk_free_TB": int(free[3]) / 1e12}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((ROOT / "configs/cl_opd.yaml").read_text())
    R = {"B1_anatomy": anatomy(cfg), "D_handoff": handoff()}
    R["E3_dry_runs"] = dry_runs(ROOT / "configs/cl_opd.yaml", out)
    R["E4_coverage"] = coverage(cfg)
    R["E5_budget"] = budget(cfg, R["D_handoff"])
    (out / "cpu.json").write_text(json.dumps(R, indent=1))
    print(json.dumps(R, indent=1))


if __name__ == "__main__":
    main()
