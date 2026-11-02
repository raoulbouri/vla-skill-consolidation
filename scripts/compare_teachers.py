"""Paired held-out comparison: original pi05_libero vs the 25-demo teacher of each task (CPU, read-only inputs).

Both arms ran on the same 25 initial states per task (held-out demos demo_25..49), so per-task differences are paired:
reports success counts, Wilson 95% CIs, discordant pairs and an exact two-sided McNemar p-value.
  scripts/py-policy scripts/compare_teachers.py   -> outputs/teacher_eval/comparison_n25.{csv,json,md}
"""

from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys
from pathlib import Path

from clvla.evaluation import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
TEACHER = os.environ.get("CLVLA_TEACHER_NAME", "teacher-n25-s500")


def episodes(run_dir: Path):
    return [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines() if line.strip()]


def mcnemar_exact(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2**n)


def latest(pattern: str):
    runs = [Path(p).parent for p in sorted(glob.glob(pattern))]
    return runs[-1] if runs else None


def main() -> int:
    out_dir = ROOT / "outputs/teacher_eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dir = latest(str(ROOT / "outputs/baseline/*_pi05_libero_heldout_n25/summary.json"))
    if base_dir is None:
        print("no held-out baseline run yet (scripts/eval_teachers_heldout.sh baseline)")
        return 1
    base = {}
    for e in episodes(base_dir):
        base.setdefault(e["libero_90_index"], {})[e["init_state_idx"]] = e
    rows, tot = [], {"b": 0, "t": 0, "n": 0}
    for idx in base:
        tdir = latest(str(ROOT / f"outputs/teacher_eval/*_{TEACHER}-t{idx:03d}_heldout/summary.json"))
        b_eps = base[idx]
        row = {"libero_90_index": idx, "task_name": next(iter(b_eps.values()))["task_name"], "n": len(b_eps),
               "original_successes": sum(e["success"] for e in b_eps.values())}
        row["original_ci95"] = [round(x, 3) for x in wilson_interval(row["original_successes"], row["n"])]
        if tdir is None:
            row.update(teacher_successes=None, note="teacher not evaluated yet")
        else:
            t_eps = {e["init_state_idx"]: e for e in episodes(tdir)}
            if set(t_eps) != set(b_eps):
                print(f"t{idx}: init states differ between arms; refusing to pair")
                return 1
            ts = sum(e["success"] for e in t_eps.values())
            only_t = sum(t_eps[i]["success"] and not b_eps[i]["success"] for i in b_eps)
            only_b = sum(b_eps[i]["success"] and not t_eps[i]["success"] for i in b_eps)
            row.update(teacher_successes=ts, teacher_ci95=[round(x, 3) for x in wilson_interval(ts, row["n"])],
                       teacher_only=only_t, original_only=only_b, mcnemar_p=round(mcnemar_exact(only_t, only_b), 4),
                       infra_errors=sum(e["infra_error"] for e in t_eps.values()) + sum(e["infra_error"] for e in b_eps.values()),
                       teacher_run=str(tdir))
            tot["b"] += row["original_successes"]; tot["t"] += ts; tot["n"] += row["n"]
        rows.append(row)
    summary = {"baseline_run": str(base_dir), "init_states": "held-out demos demo_25..49 (split profile n25)",
               "tasks": rows, "paired_total": tot}
    (out_dir / f"comparison_{TEACHER}.json").write_text(json.dumps(summary, indent=2))
    with (out_dir / f"comparison_{TEACHER}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), extrasaction="ignore")
        w.writeheader()
        [w.writerow({k: r.get(k) for k in rows[0]}) for r in rows]
    lines = ["| task | original /25 | teacher /25 | teacher-only | original-only | McNemar p |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| t{r['libero_90_index']:02d} {r['task_name'][:48]} | {r['original_successes']} | "
                     f"{r.get('teacher_successes', '-')} | {r.get('teacher_only', '-')} | {r.get('original_only', '-')} | "
                     f"{r.get('mcnemar_p', '-')} |")
    if tot["n"]:
        lines.append(f"| **paired total** | {tot['b']}/{tot['n']} | {tot['t']}/{tot['n']} | | | |")
    (out_dir / f"comparison_{TEACHER}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
