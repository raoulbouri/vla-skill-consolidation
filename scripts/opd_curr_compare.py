"""Final held-out comparison for the two-phase curriculum on demo_10..49, the same states, seeds and protocol as the earlier t35 benchmark (scripts/opd_t35_compare.py), plus handle reach, contact and door-onset rates from `--geometry`. Not used for checkpoint selection. Similar rule as before: on ODE, upper 95% limit of (teacher - student) success <= 0.15.
  python3 scripts/opd_curr_compare.py <run_dir> <final_student_id> <policy_ids comma-separated>"""
import csv, json, sys
from math import comb
from pathlib import Path
import numpy as np

run, SID, P = Path(sys.argv[1]), sys.argv[2], tuple(sys.argv[3].split(","))
rows = [json.loads(l) for f in sorted((run / "evaluation" / "workers").glob("*.jsonl")) for l in f.read_text().splitlines()]
idx = {(r["policy_id"], r["mode"], r["state"]): r for r in rows}
states = list(range(10, 50))
assert len(idx) == len(rows) == len(P) * 2 * 40, (len(idx), len(rows))
rng = np.random.default_rng(20261102)
v = lambda p, m, k="success": np.array([bool(idx[(p, m, s)][k]) if k == "success" else idx[(p, m, s)][k] is not None for s in states], float)  # noqa: E731


def wilson(x, n=40, z=1.96):
    p = x / n
    c, h = (p + z * z / (2 * n)) / (1 + z * z / n), z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return [c - h, c + h]


def cmp(a, b):
    d = a - b
    q = np.percentile(d[rng.integers(0, 40, size=(20000, 40))].mean(axis=1), [2.5, 97.5])
    x, y = int((d > 0).sum()), int((d < 0).sum())
    n = x + y
    return {"a": int(a.sum()), "b": int(b.sum()), "difference": float(d.mean()), "ci95": q.tolist(), "a_only": x, "b_only": y, "mcnemar_p": 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2**n)}


S = {"cells": {}, "comparisons": {}}
for p in P:
    for m in ("ode", "sde"):
        rs = [idx[(p, m, s)] for s in states]
        x = sum(r["success"] for r in rs)
        S["cells"][f"{p}_{m}"] = {"successes": x, "episodes": 40, "wilson95": wilson(x), "errors": sum(r["termination"] == "error" for r in rs), "nonfinite": sum(r["nonfinite_actions"] for r in rs), "steps_mean_success": float(np.mean([r["steps"] for r in rs if r["success"]])) if x else None, "near_handle": sum(r["geo_first_near"] is not None for r in rs), "contact": sum(r["geo_first_contact"] is not None for r in rs), "door_moved": sum(r["geo_door_onset"] is not None for r in rs), "median_min_dist": float(np.median([r["geo_min_dist"] for r in rs])), "door_min_mean": float(np.mean([r["door_min"] for r in rs]))}
pairs = [("teacher10d", SID), (SID, "base"), (SID, "student500_old")] + [(p, "base") for p in P if p not in ("base", SID, "teacher10d")] + [(SID, p) for p in P if p.startswith("curr_") and p != SID]
for m in ("ode", "sde"):
    for a_, b_ in pairs:
        S["comparisons"][f"{a_}_minus_{b_}_{m}"] = cmp(v(a_, m), v(b_, m))
    for k in ("geo_first_near", "geo_door_onset"):
        S["comparisons"][f"{SID}_minus_base_{k}_{m}"] = cmp(v(SID, m, k), v("base", m, k))
up = S["comparisons"][f"teacher10d_minus_{SID}_ode"]["ci95"][1]
S["similar_rule"] = {"upper_limit": up, "margin": 0.15, "similar": up <= 0.15}
(run / "comparison.json").write_text(json.dumps(S, indent=1))
with (run / "side_by_side.csv").open("w", newline="") as f:
    c = csv.writer(f)
    c.writerow(["state"] + [f"{p}_{m}" for p in P for m in ("ode", "sde")])
    for s in states:
        c.writerow([s] + [int(idx[(p, m, s)]["success"]) for p in P for m in ("ode", "sde")])
print(json.dumps(S, indent=1))
