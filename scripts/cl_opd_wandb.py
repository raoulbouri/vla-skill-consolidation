"""Log a continual-learning OPD run to W&B after the fact: one W&B run per finished task (every training-metrics row,
step = global update across the curriculum) and one for the success matrix. The key is read from stdin and only kept in
this process's environment (never on disk or the command line), so training jobs never need it; already-logged tasks are
skipped (<stage>/WANDB.json).
  scripts/py-policy scripts/cl_opd_wandb.py --run outputs/cl_opd/<run> < key_on_stdin     (--dry-run: print, no network)
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def rows_of(student: Path) -> list:
    out = []
    for f in sorted(student.glob("training_metrics*.csv")):
        with f.open() as fh:
            for r in csv.DictReader(fh):
                out.append({k: (float(v) if v not in ("", "None", "True", "False") and _num(v) else v) for k, v in r.items()})
    seen, uniq = set(), []
    for r in sorted(out, key=lambda r: float(r["update"])):
        if r["update"] not in seen:  # a resumed run can repeat an update in a second csv
            seen.add(r["update"])
            uniq.append(r)
    return uniq


def _num(v):
    try:
        float(v)
        return True
    except ValueError:
        return False


def run_name(c: dict, k: int) -> str:
    t = c["tasks"][k]
    return c.get("wandb", {}).get("name_pattern", "{method}-{demos}demos-s{steps}-t{idx}").format(method=c["name"], demos=c["profile"].lstrip("n"), steps=c["per_task"]["total_updates"], idx=t["id"].lstrip("t"))


def plan(run: Path) -> list:
    c = json.loads((run / "config.json").read_text())["config"]
    jobs = []
    for k, t in enumerate(c["tasks"]):
        sd = run / f"k{k}_{t['id']}"
        if (sd / "DONE").exists() and not (sd / "WANDB.json").exists():
            jobs.append({"k": k, "name": run_name(c, k), "rows": rows_of(sd / "student"), "dir": sd, "task": t})
    return c, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run = Path(a.run) if Path(a.run).is_absolute() else ROOT / a.run
    c, jobs = plan(run)
    total = c["per_task"]["total_updates"]
    for j in jobs:
        print(f"[wandb] {j['name']}: {len(j['rows'])} rows, steps {j['k'] * total + 1}..{j['k'] * total + len(j['rows'])}")
    if a.dry_run:
        return
    key = sys.stdin.readline().strip()
    assert key, "W&B key expected on stdin"
    os.environ["WANDB_API_KEY"] = key
    import wandb

    entity, project = c["wandb"]["project"].split("/")
    for j in jobs:
        r = wandb.init(entity=entity, project=project, name=j["name"], group=c["name"], job_type="train", config={**c, "task_index": j["k"], "task": j["task"]}, reinit=True, dir=str(run))
        for row in j["rows"]:
            wandb.log({k: v for k, v in row.items() if isinstance(v, float)} | {"task_index": j["k"]}, step=j["k"] * total + int(row["update"]))
        trained = json.loads((j["dir"] / "TRAINED").read_text())
        r.summary.update(trained)
        r.finish()
        (j["dir"] / "WANDB.json").write_text(json.dumps({"url": r.url, "id": r.id}))
    if (run / "matrix.json").exists():
        m = json.loads((run / "matrix.json").read_text())
        last = m.get("last_stage")
        r = wandb.init(entity=entity, project=project, name=f"{c['name']}-matrix-after-k{last}", group=c["name"], job_type="matrix", config=c, reinit=True, dir=str(run))
        tbl = wandb.Table(columns=["after"] + m["tasks"], data=[[f"after {t}"] + [v for v in row] for t, row in zip(m["tasks"], m["S"])])
        r.log({"success_matrix": tbl, "matrix_png": wandb.Image(str(run / "matrix.png"))})
        r.summary.update({k: m.get(k) for k in ("final_avg", "nbt_avg", "nbt", "nbt_norm", "fwt", "base")})
        r.finish()
    os.environ.pop("WANDB_API_KEY", None)


if __name__ == "__main__":
    main()
