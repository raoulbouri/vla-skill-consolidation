"""Writes config.json, evaluation_manifest.json and four job lists for the 1,500-episode Stage 1 matrix, from outputs/checkpoints."""
import json, sys, os, hashlib, subprocess
from pathlib import Path
run = Path(sys.argv[1]); run.mkdir(parents=True, exist_ok=True); (run/"jobs").mkdir(exist_ok=True)
idx = [78, 84, 16, 56, 35, 42, 28, 3, 89, 63]
rows, jobs = [], []
for pos, t in enumerate(idx):
    for var, did in (("expert300m-10demos-s250", "teacher10d"), ("expert300m-25demos-s250", "teacher25d")):
        link = next(Path("outputs/checkpoints", var).glob(f"t{t:03d}_*"))
        ck = link.resolve(); r = Path(str(ck).split("/checkpoints/")[0])
        s = json.loads((r/"summary.json").read_text())
        assert s["status"] == "completed"
        m = json.loads((r/"run_metadata.json").read_text())
        pid = f"{did}_t{t:03d}"
        rows.append({"policy_id": pid, "position": pos, "libero_90_index": t, "variant": var, "checkpoint": str(ck), "run_dir": str(r), "fingerprint": s.get("trained_checkpoint_fingerprint"), "trainable_params": s["init"].get("trainable_params"), "total_params": s["init"].get("total_params"), "final_val_loss": s.get("final_val_loss"), "mode": (s["init"].get("mode"))})
        jobs.append(f"{pid} teacher {r} {pos} ode,sde")
assert len(jobs) == 20
cfg = {"created_by": "scripts/combined_prep.py", "teacher_collection": "outputs/checkpoints/expert300m-{10,25}demos-s250 (20 checkpoints; the only full expert-and-head set in the index; no other manifest exists on either machine)", "base": "pi05_libero", "held_out": "demo_25..49 (profile n25)", "env_seed": 7, "noise_seed": "200000 + 100000*position + 1000*state + event", "sde": {"K": 10, "sigma": 0.05, "noise_mode": "ogpo_discrete"}, "xla_flags": "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0", "compute": "aims, 2 RTX 4090, user-approved 2026-11-02"}
(run/"config.json").write_text(json.dumps(cfg, indent=1))
(run/"evaluation_manifest.json").write_text(json.dumps({"tasks": idx, "policies": rows, "modes": ["ode", "sde"], "episodes_per_cell": 25, "total_episodes": 1500}, indent=1))
all_jobs = [f"base_t{t:03d} base - {p} ode,sde" for p, t in enumerate(idx)] + jobs
for w in range(4):
    (run/"jobs"/f"w{w}").write_text("\n".join(all_jobs[w::4]) + "\n")
print(len(all_jobs))
