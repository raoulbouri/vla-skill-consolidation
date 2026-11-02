"""Side-by-side t35 comparison on the 40 states demo_10..49, never used by the student or Teacher-10D. Rule fixed before evaluation: on ODE, the named student is called similar to the teacher only if the upper 95% limit of (teacher - student) success is at most 0.15. Door opening is the most negative hinge angle in the episode; the task's open range starts at -1.3 rad.
  python scripts/opd_t35_compare.py <run_dir> <policy_ids comma-separated, teacher10d and base included> <student_id>"""
import csv, json, sys
from math import comb
from pathlib import Path
import numpy as np
run = Path(sys.argv[1]); E = run / "evaluation"
rows = [json.loads(l) for f in sorted((E / "workers").glob("*.jsonl")) for l in f.read_text().splitlines()]
idx = {(r["policy_id"], r["mode"], r["state"]): r for r in rows}
states = sorted({r["state"] for r in rows}); assert states == list(range(10, 50))
P = tuple(sys.argv[2].split(",")); SID = sys.argv[3]
assert len(idx) == len(rows) == len(P) * 2 * 40
rng = np.random.default_rng(20261102)
v = lambda p, m: np.array([idx[(p, m, s)]["success"] for s in states], float)
def cmp(a, b):
    d = a - b; q = np.percentile(d[rng.integers(0, 40, size=(20000, 40))].mean(axis=1), [2.5, 97.5])
    x, y = int((d > 0).sum()), int((d < 0).sum()); n = x + y
    return {"a": int(a.sum()), "b": int(b.sum()), "difference": float(d.mean()), "ci95": q.tolist(), "a_only": x, "b_only": y, "mcnemar_p": 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2**n)}
summary = {"cells": {}, "comparisons": {}}
for p in P:
    for m in ("ode", "sde"):
        rs = [idx[(p, m, s)] for s in states]
        summary["cells"][f"{p}_{m}"] = {"successes": sum(r["success"] for r in rs), "episodes": 40, "reward_mean": float(np.mean([r["reward_sum"] for r in rs])), "steps_mean": float(np.mean([r["steps"] for r in rs])), "steps_mean_success": float(np.mean([r["steps"] for r in rs if r["success"]])) if any(r["success"] for r in rs) else None, "timeouts": sum(r["termination"] == "timeout" for r in rs), "errors": sum(r["termination"] == "error" for r in rs), "nonfinite": sum(r["nonfinite_actions"] for r in rs), "latency_mean_s": float(np.mean([r["latency_mean_s"] for r in rs])), "door_min_mean": float(np.mean([r["door_min"] for r in rs])), "door_min_best": float(np.min([r["door_min"] for r in rs])), "door_moved_over_0p1rad": sum(r["door_min"] < -0.1 for r in rs)}
for m in ("ode", "sde"):
    summary["comparisons"][f"teacher_minus_student_{m}"] = cmp(v("teacher10d", m), v(SID, m))
    summary["comparisons"][f"student_minus_base_{m}"] = cmp(v(SID, m), v("base", m))
    summary["comparisons"][f"teacher_minus_base_{m}"] = cmp(v("teacher10d", m), v("base", m))
for p in P:
    summary["comparisons"][f"{p}_sde_minus_ode"] = cmp(v(p, "sde"), v(p, "ode"))
up = summary["comparisons"]["teacher_minus_student_ode"]["ci95"][1]
summary["similar_rule"] = {"upper_limit": up, "margin": 0.15, "similar": up <= 0.15}
(run / "comparison.json").write_text(json.dumps(summary, indent=1))
with (run / "side_by_side.csv").open("w", newline="") as f:
    c = csv.writer(f); c.writerow(["state"] + [f"{p}_{m}" for p in P for m in ("ode", "sde")])
    for s in states: c.writerow([s] + [int(idx[(p, m, s)]["success"]) for p in P for m in ("ode", "sde")])
print(json.dumps(summary, indent=1))
