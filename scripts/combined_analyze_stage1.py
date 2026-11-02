"""Stage 1 tables from the per-episode records; the run was stopped early at the user's request, so cells may be partial. Comparisons use only states present in both cells: risk difference, 95% percentile bootstrap over states, exact McNemar p. Macro rows average the tasks compared and name them.
  python scripts/combined_analyze_stage1.py <run_dir>"""
import csv, json, sys
from math import comb
from pathlib import Path
import numpy as np

run = Path(sys.argv[1]); S = run / "stage1"
rows = [json.loads(l) for f in sorted((S / "workers").glob("w*.jsonl")) for l in f.read_text().splitlines()]
for r in rows:
    r["policy"] = r["policy_id"].split("_t")[0]
assert len({(r["policy"], r["position"], r["mode"], r["state"]) for r in rows}) == len(rows), "duplicate episodes"
(S / "episode_results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
tasks = sorted({(r["position"], r["libero_90_index"], r["task"]) for r in rows})
POL, MODES = ("base", "teacher10d", "teacher25d"), ("ode", "sde")
idx = {(r["policy"], r["position"], r["mode"], r["state"]): r for r in rows}
rng = np.random.default_rng(20261102)
cell = lambda p, t, m: sorted(s for (a, b, c, s) in idx if (a, b, c) == (p, t, m))
def mcnemar(a, b):
    x, y = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 0) & (b == 1))); n = x + y
    return 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, i) for i in range(min(x, y) + 1)) / 2**n)
tm = []
for pos, li, name in tasks:
    for p in POL:
        for m in MODES:
            rs = [idx[(p, pos, m, s)] for s in cell(p, pos, m)]
            if rs:
                k = sum(r["success"] for r in rs)
                tm.append({"position": pos, "libero_90_index": li, "task": name, "policy": p, "mode": m, "successes": k, "episodes": len(rs), "complete": len(rs) == 25, "rate": k / len(rs), "reward_mean": float(np.mean([r["reward_sum"] for r in rs])), "steps_mean": float(np.mean([r["steps"] for r in rs])), "timeouts": sum(r["termination"] == "timeout" for r in rs), "errors": sum(r["termination"] == "error" for r in rs), "nonfinite": sum(r["nonfinite_actions"] for r in rs), "arm_outside_range_mean": float(np.mean([r["arm_outside_input_range_fraction"] for r in rs])), "latency_mean_s": float(np.mean([r["latency_mean_s"] for r in rs])), "runtime_mean_s": float(np.mean([r["runtime_s"] for r in rs]))})
def w(name, data):
    with (S / name).open("w", newline="") as f:
        c = csv.DictWriter(f, fieldnames=list(data[0])); c.writeheader(); c.writerows(data)
w("task_metrics.csv", tm)
agg = []
for p in POL:
    for m in MODES:
        x = [t for t in tm if t["policy"] == p and t["mode"] == m]
        c = [t for t in x if t["complete"]]
        agg.append({"policy": p, "mode": m, "cells_started": len(x), "cells_complete": len(c), "successes_all": sum(t["successes"] for t in x), "episodes_all": sum(t["episodes"] for t in x), "macro_rate_complete_cells": float(np.mean([t["rate"] for t in c])) if c else None, "errors": sum(t["errors"] for t in x), "nonfinite": sum(t["nonfinite"] for t in x), "timeouts": sum(t["timeouts"] for t in x), "latency_mean_s": float(np.mean([t["latency_mean_s"] for t in x])) if x else None})
w("aggregate_metrics.csv", agg)
def compare(pairs):
    out = []
    for lab, (pa, ma), (pb, mb) in pairs:
        ds, names = [], []
        for pos, li, name in tasks:
            st = sorted(set(cell(pa, pos, ma)) & set(cell(pb, pos, mb)))
            if len(st) < 5:
                continue
            a = np.array([idx[(pa, pos, ma, s)]["success"] for s in st], float); b = np.array([idx[(pb, pos, mb, s)]["success"] for s in st], float)
            d = a - b; ds.append(d); names.append(li)
            v = d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1); q = np.percentile(v, [2.5, 97.5])
            out.append({"comparison": lab, "task": f"t{li:03d}", "a": int(a.sum()), "b": int(b.sum()), "n": len(st), "difference": float(d.mean()), "ci_low": float(q[0]), "ci_high": float(q[1]), "a_only": int(np.sum(d > 0)), "b_only": int(np.sum(d < 0)), "mcnemar_p": mcnemar(a, b), "tasks": ""})
        if ds:
            bs = np.stack([d[rng.integers(0, len(d), size=(10000, len(d)))].mean(axis=1) for d in ds], 1).mean(axis=1); q = np.percentile(bs, [2.5, 97.5])
            out.append({"comparison": lab, "task": "macro_compared_tasks", "a": "", "b": "", "n": sum(len(d) for d in ds), "difference": float(np.mean([d.mean() for d in ds])), "ci_low": float(q[0]), "ci_high": float(q[1]), "a_only": "", "b_only": "", "mcnemar_p": "", "tasks": " ".join(map(str, names))})
    return out
base = compare([(f"{t}_{m}_vs_base", (t, m), ("base", m)) for t in POL[1:] for m in MODES]); w("base_vs_teachers.csv", base)
t25 = compare([(f"teacher25d_vs_teacher10d_{m}", ("teacher25d", m), ("teacher10d", m)) for m in MODES]); w("teacher10_vs_teacher25.csv", t25)
ods = compare([(f"{p}_sde_vs_ode", (p, "sde"), (p, "ode")) for p in POL]); w("ode_sde_comparison.csv", ods)
(S / "aggregate_metrics.json").write_text(json.dumps({"stopped_early_at_user_request": True, "episodes": len(rows), "planned": 1500, "aggregate": agg}, indent=1))
g = {(t["position"], t["policy"], t["mode"]): f"{t['successes']}/{t['episodes']}" for t in tm}
L = ["# Stage 1: base and teacher evaluation (stopped early at the user's request)", "", f"{len(rows)} of 1500 planned episodes; held-out states `demo_25..49`; env seed 7; noise matched by state. Blank cells were not started; counts below /25 are partial.", "", "| Task | Base ODE | Base SDE | 10D ODE | 10D SDE | 25D ODE | 25D SDE |", "|---|---|---|---|---|---|---|"]
for pos, li, name in tasks:
    L.append(f"| t{li:03d} {name} | " + " | ".join(g.get((pos, p, m), "") for p in POL for m in MODES) + " |")
(S / "stage1_report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
for name, data in (("base", base), ("t25", t25), ("ode_sde", ods)):
    print(name)
    for r in data:
        if r["task"] in ("macro_compared_tasks", "t035"):
            print(" ", r["comparison"], r["task"], r["a"], r["b"], r["n"], round(r["difference"], 3), [round(r["ci_low"], 3), round(r["ci_high"], 3)], r["mcnemar_p"], r["tasks"])
for a_ in agg: print(a_)
