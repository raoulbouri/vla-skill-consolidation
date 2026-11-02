"""Cross-task evaluation of finished specialists (optional, later). DEFAULT IS A DRY RUN.

Each specialist is served alone (fresh server, GPU guard) and evaluated with the baseline loop (scripts/run_eval.py,
kind specialist_eval): on its own task, or with --matrix on every manifest task -> expert-by-task success matrix.
Evaluations run one after another. Selection init states (0-9) by default; the test split needs --confirmatory.

  scripts/py-sim scripts/eval_specialists.py --group spec-v1                        # plan
  scripts/py-sim scripts/eval_specialists.py --group spec-v1 --execute --episodes-per-task 5
  scripts/py-sim scripts/eval_specialists.py --group spec-v1 --execute --matrix --episodes-per-task 3
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from clvla.config import load_experiment_config
from clvla.experiment import utc_stamp
from clvla.specialists import assemble_matrix
from clvla.tasks import load_manifest

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--group", required=True, help="specialist spec name (run dirs *_<group>-t<idx>)")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--matrix", action="store_true", help="evaluate every specialist on every manifest task")
    ap.add_argument("--episodes-per-task", type=int, default=3)
    ap.add_argument("--split", choices=["selection", "test"], default="selection")
    ap.add_argument("--confirmatory", action="store_true")
    ap.add_argument("--save-videos", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    manifest = load_manifest(cfg.resolve(cfg.task_manifest))
    out_root = Path(cfg.output_root)
    runs = []
    for d in sorted((out_root / "specialists").glob(f"*_{args.group}-t*")):
        s = d / "summary.json"
        if s.is_file() and json.loads(s.read_text()).get("status") == "completed":
            runs.append((json.loads((d / "run_metadata.json").read_text())["task"]["libero_90_index"], d))
    if not runs:
        raise SystemExit(f"no completed specialists for group {args.group!r} under {out_root / 'specialists'}")
    experts = [i for i, _ in runs]
    if len(set(experts)) != len(experts):
        raise SystemExit(f"several completed runs for the same task in group {args.group}: {experts}; pick one")
    tasks = [t.libero_90_index for t in manifest.tasks] if args.matrix else None
    print(f"[eval_specialists] {len(runs)} specialist(s): {experts}; "
          f"{'all ' + str(len(tasks)) + ' tasks' if tasks else 'own task only'}; split {args.split}, "
          f"{args.episodes_per_task} episodes/task")
    if not args.execute:
        for i, d in runs:
            print(f"  t{i:03d}: {d}")
        print("[eval_specialists] dry run only.")
        return 0

    records = []
    for idx, run in runs:
        positions = ",".join(str(manifest.by_index(t).position) for t in (tasks or [idx]))
        name = f"{args.group}-t{idx:03d}-on-{'all' if tasks else 'own'}"
        cmd = ["bash", str(ROOT / "scripts/eval_checkpoint.sh"), str(run), "--kind", "specialist_eval",
               "--split", args.split, "--episodes-per-task", str(args.episodes_per_task), "--positions", positions,
               "--name", name] + (["--confirmatory"] if args.confirmatory else []) + (["--save-videos"] if args.save_videos else [])
        print("$ " + " ".join(cmd), flush=True)
        rc = subprocess.run(cmd).returncode
        ev_dir = sorted((out_root / "specialist_eval").glob(f"*_{name}"))[-1]
        summary = json.loads((ev_dir / "summary.json").read_text())
        for t in summary["per_task"]:
            records.append({"expert": idx, "task": t["libero_90_index"], "successes": t["successes"],
                            "episodes": t["episodes"], "infra_errors": t["infra_errors"], "eval_dir": str(ev_dir)})
        if rc not in (0, 2):
            raise SystemExit(f"evaluation of t{idx} failed (rc {rc}); stopping")
    mat = assemble_matrix(records, experts, tasks or experts)
    mat.update({"group": args.group, "split": args.split, "episodes_per_task": args.episodes_per_task,
                "created_utc": utc_stamp(), "records": records})
    out = out_root / "specialist_eval" / f"{utc_stamp()}_{args.group}_{'matrix' if tasks else 'own'}.json"
    out.write_text(json.dumps(mat, indent=1))
    print(f"[eval_specialists] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
