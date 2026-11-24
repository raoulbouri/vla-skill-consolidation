"""Continual-learning OPD driver: one task of the curriculum per call, every choice from configs/cl_opd.yaml.
  scripts/py-policy scripts/cl_opd_run.py --config configs/cl_opd.yaml --run outputs/cl_opd/<run> --task K [--gpu G] [--stage all|train|buffer|eval] [--dry-run]

Task K (curriculum index), in order, each step skipped if already done, so rerunning after a crash or SIGTERM continues:
  train   Phase 1 (teacher drives, --stop-after warm_updates) then Phase 2 (student drives) with scripts/opd_lora_train.py.
          K > 0: the student starts from task K-1's final master, and every update adds old_tasks.per_update buffer
          observations of tasks 0..K-1, each labelled by its own teacher. The full training state is saved every
          per_task.state_every updates and on SIGTERM; an interrupted phase resumes from it.
  buffer  rank task K's recorded teacher episodes and admit them to the fixed-total buffer (every earlier task is cut to
          its new share).
  eval    the task-K student on tasks 0..K (+ K+1 if evaluation.next_task; base pi0.5 on every task at K = 0), held-out
          states, then <run>/matrix.json and matrix.png.
Task K needs task K-1 finished (DONE file). The run directory keeps a copy of the config; another config is refused.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from clvla.cl_opd import buffer as B  # noqa: E402
from clvla.cl_opd import config as C  # noqa: E402
from clvla.cl_opd import matrix as MX  # noqa: E402


def stage_dir(run: Path, k: int, task: dict) -> Path:
    return run / f"k{k}_{task['id']}"


def state_update(student: Path) -> int:
    f = student / "state" / "state.json"
    return json.loads(f.read_text())["update"] if f.exists() else 0


def final_master(student: Path) -> Path | None:
    m = student / "metrics.json"
    if not m.exists():
        return None
    u = json.loads(m.read_text()).get("completed_updates")
    f = student / "checkpoints" / f"adapter_{u:03d}.npz" if u else None
    return f if f is not None and f.exists() else None


def trainer_cmd(c: dict, run: Path, k: int, phase: int, resume: bool) -> list:
    t, p, o = c["tasks"][k], c["per_task"], c["old_tasks"]
    sd = stage_dir(run, k, t)
    cmd = ["scripts/opd_lora_train.py", "--teacher-run", t["teacher"], "--teacher-delta", t["teacher"], "--teacher-variant", t["variant"],
           "--name", "student", "--out", str(sd / "student"), "--position", str(t["position"]), "--updates", str(p["total_updates"]),
           "--random-starts", "--arm-noise", str(p["arm_noise"]), "--profile", c["profile"], "--teacher-control-updates", str(p["warm_updates"]),
           "--teacher-success-only", "--student-mode", c["student"], "--geometry", t["geometry"], "--state-every", str(p["state_every"]),
           "--ckpt-at", ",".join(map(str, p["checkpoint_updates"])), "--record-dir", str(sd / "teacher_episodes")]
    if k > 0:
        prev = final_master(stage_dir(run, k - 1, c["tasks"][k - 1]) / "student")
        assert prev is not None or "DRYRUN" in os.environ, f"task {k - 1} has no final master"
        cmd += ["--init-master", str(prev or "<task k-1 final master>")]
        if o["per_update"]:
            cmd += ["--old-buffer", str(run / "buffer"), "--old-teachers", str(sd / "old_teachers.json"), "--old-per-update", str(o["per_update"]), "--old-weight", str(o["weight"]), "--old-seed", str(o["seed"])]
    if p.get("traj_clip_k"):
        cmd += ["--traj-clip-k", str(p["traj_clip_k"])]  # per-trajectory gradient clipping (both phases); absent or 0: off
    if p.get("kl_gate", True) is False:
        cmd += ["--no-kl-gate"]  # the one-batch gate's fresh-noise KL check only logs; absent or true: it stops training
    if p.get("trust_region_k"):
        cmd += ["--trust-region-k", str(p["trust_region_k"])]  # action-space trust region by step backtracking (both phases); absent or 0: off
    if phase == 1:
        cmd += ["--stop-after", str(p["warm_updates"])]
    else:
        cmd += ["--rollouts-per-update", str(p["rollouts_per_update"]), "--chains-per-obs", str(p["chains_per_obs"]),
                "--replay-per-update", str(p["current_replay_per_update"]), "--replay-capacity", str(p["current_replay_capacity"])]
        if t["geometry"] == "microwave" and p["stop_contact_rate"]:
            cmd += ["--stop-contact-rate", str(p["stop_contact_rate"]), "--stop-window", "100", "--stop-min-episodes", "200"]
    if resume:
        cmd += ["--resume"]
    return cmd


def child_env(gpu: int, mem_fraction: str = ".75") -> dict:
    e = {k: v for k, v in os.environ.items() if k not in ("JAX_PLATFORMS", "CUDA_VISIBLE_DEVICES")}
    e.update(CLVLA_USE_GPU="1", CLVLA_GPU=str(gpu), MALLOC_ARENA_MAX="2", XLA_PYTHON_CLIENT_MEM_FRACTION=mem_fraction,
             PYTHONPATH=f"{ROOT / 'scripts'}:{ROOT / 'tests' / 'ogpo_port'}")
    return e


def run_cmd(cmd: list, log: Path, env: dict, dry: bool) -> int:
    print(("[dry-run] " if dry else "[run] ") + " ".join(shlex.quote(x) for x in cmd) + f"  > {log}", flush=True)
    if dry:
        return 0
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "a") as f:
        return subprocess.call(cmd, stdout=f, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=ROOT, env=env)


def train(c, run, k, gpu, dry) -> bool:
    t, p = c["tasks"][k], c["per_task"]
    sd = stage_dir(run, k, t)
    if (sd / "TRAINED").exists():
        print(f"[train] task {k} already trained")
        return True
    if not dry and (sd / "student").exists() and not (sd / "student" / "state" / "state.json").exists():
        # died before its first saved state: keep it for inspection, start the task again
        n = len(list(sd.glob("student.failed_*")))
        (sd / "student").rename(sd / f"student.failed_{n}")
        shutil.rmtree(sd / "teacher_episodes", ignore_errors=True)
        print(f"[train] task {k}: no saved state; previous attempt moved to student.failed_{n}")
    if k > 0 and c["old_tasks"]["per_update"] and not dry:
        sd.mkdir(parents=True, exist_ok=True)
        (sd / "old_teachers.json").write_text(json.dumps({x["id"]: {"delta": x["teacher"], "variant": x["variant"]} for x in c["tasks"][:k]}, indent=1))
    env = child_env(gpu, str(p.get("gpu_mem_fraction", 0.75)))
    py = [str(ROOT / "scripts" / "py-policy"), "-W", "ignore"]
    nice = ["nice", "-n", "5"]
    u = state_update(sd / "student")
    if u < p["warm_updates"]:
        rc = run_cmd(nice + py + trainer_cmd(c, run, k, 1, resume=u > 0), sd / "logs" / "train_phase1.log", env, dry)
        if dry:
            u = p["warm_updates"]
        elif rc != 0 or state_update(sd / "student") < p["warm_updates"]:
            print(f"[train] task {k} Phase 1 stopped at update {state_update(sd / 'student')} (rc {rc}); rerun to continue")
            return False
    rc = run_cmd(nice + py + trainer_cmd(c, run, k, 2, resume=True), sd / "logs" / "train_phase2.log", env, dry)
    if dry:
        return True
    m = json.loads((sd / "student" / "metrics.json").read_text())
    if rc != 0 or str(m.get("stop_reason", "")).startswith(("SIGTERM", "rollout error", "non-finite", "one-batch", "diagnostic", "no successful")):
        print(f"[train] task {k} Phase 2 ended: {m.get('stop_reason')} (rc {rc}); rerun to resume from the saved state")
        return False
    (sd / "TRAINED").write_text(json.dumps({"completed_updates": m["completed_updates"], "stop_reason": m.get("stop_reason"), "final_master": str(final_master(sd / "student"))}))
    shutil.rmtree(sd / "student" / "state", ignore_errors=True)  # 5 GB; the run is finished and its final master is kept
    return True


def admit(c, run, k, dry) -> bool:
    t, o = c["tasks"][k], c["old_tasks"]
    buf = B.FixedTotalBuffer(run / "buffer", o["buffer_total"]) if not dry else None
    if buf is not None and t["id"] in buf.state["tasks"]:
        print(f"[buffer] {t['id']} already admitted")
        return True
    rec = stage_dir(run, k, t) / "teacher_episodes"
    print(f"[buffer] admit {t['id']} from {rec} (total {o['buffer_total']}, seed {o['seed'] + k}, min gap {o['min_gap']})")
    if dry:
        return True
    rows, arrays, meta = B.ranked_from_records(rec, o["seed"] + k, o["min_gap"])
    shares = buf.admit(t["id"], rows, arrays, {**meta, "teacher": t["teacher"], "variant": t["variant"]})
    assert buf.verify() and buf.stored() <= o["buffer_total"]
    print(f"[buffer] shares {shares}; stored {buf.stored()}; {buf.bytes_on_disk() / 2**30:.2f} GiB")
    return True


def evaluate(c, run, k, gpu, dry) -> bool:
    tasks, e = c["tasks"], c["evaluation"]
    sd = stage_dir(run, k, tasks[k])
    pos = [t["position"] for t in tasks[: k + 1 + (1 if e["next_task"] and k + 1 < len(tasks) else 0)]]
    master = final_master(sd / "student") if not dry else "<final_master>"
    jobs = [f"stage{k} student {master} {','.join(map(str, pos))} {e['modes']}"]
    if k == 0 and e["include_base"]:
        jobs.append(f"base base - {','.join(str(t['position']) for t in tasks)} {e['modes']}")
    jf = sd / "eval_jobs"
    print("[eval] jobs:\n  " + "\n  ".join(jobs))
    if not dry:
        jf.write_text("\n".join(jobs) + "\n")
        env = {**child_env(gpu, ".3"), "EVAL_SUBDIR": "evaluation", "SERVER_MEM_FRACTION": ".45"}
        rc = run_cmd(["bash", "scripts/combined_worker.sh", str(run), str(gpu), str(jf), "--profile", c["profile"], "--states", str(e["states"])], sd / "logs" / "eval.log", env, False)
        if rc != 0:
            return False
    write_matrix(c, run, dry)
    return True


def write_matrix(c, run, dry=False):
    ids, pos = [t["id"] for t in c["tasks"]], [t["position"] for t in c["tasks"]]
    if dry:
        return None
    m = MX.from_files(run / "evaluation" / "workers", ids, pos, c["evaluation"]["modes"].split(",")[0])
    (run / "matrix.json").write_text(json.dumps(m, indent=1))
    MX.heatmap(m, run / "matrix.png")
    print("[matrix]", json.dumps({k: m.get(k) for k in ("final_avg", "nbt_avg", "nbt", "fwt")}))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs" / "cl_opd.yaml"))
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", type=int, required=True, help="curriculum index K")
    ap.add_argument("--gpu", type=int, default=int(os.environ.get("GPUQ_GPU", 1)))
    ap.add_argument("--stage", default="all", choices=("all", "train", "buffer", "eval"))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if a.dry_run:
        os.environ["DRYRUN"] = "1"
    c = C.load(a.config)
    run = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
    k = a.task
    assert 0 <= k < len(c["tasks"]), f"task index {k} outside 0..{len(c['tasks']) - 1}"
    if not a.dry_run:
        run.mkdir(parents=True, exist_ok=True)
        saved = run / "config.json"
        if saved.exists():
            assert json.loads(saved.read_text())["sha"] == C.sha(c), "this run was started with another config; use a new run directory"
        else:
            saved.write_text(json.dumps({"sha": C.sha(c), "config": c}, indent=1))
        for t in c["tasks"][: k + 1]:
            assert (ROOT / t["teacher"] / "leaf_sha.json").exists(), f"teacher for {t['id']} missing: {t['teacher']}"
        if k > 0:
            prev = stage_dir(run, k - 1, c["tasks"][k - 1])
            assert (prev / "DONE").exists(), f"task {k - 1} is not finished"
            if c["old_tasks"]["per_update"]:  # per_update 0: no old-task replay (single-task ablation), so no buffer is needed
                assert B.FixedTotalBuffer(run / "buffer", c["old_tasks"]["buffer_total"]).state["tasks"] == [t["id"] for t in c["tasks"][:k]], "buffer does not hold exactly tasks 0..K-1"
        if a.stage in ("all", "train", "eval") and subprocess.call(["scripts/gpu_guard.sh", str(a.gpu)], cwd=ROOT) != 0:
            sys.exit(f"GPU {a.gpu} has other users")
    ok = True
    if a.stage in ("all", "train"):
        ok = train(c, run, k, a.gpu, a.dry_run)
    if ok and a.stage in ("all", "buffer"):
        ok = admit(c, run, k, a.dry_run)
    if ok and a.stage in ("all", "eval"):
        ok = evaluate(c, run, k, a.gpu, a.dry_run)
    if ok and a.stage == "all" and not a.dry_run:
        (stage_dir(run, k, c["tasks"][k]) / "DONE").write_text("ok\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
