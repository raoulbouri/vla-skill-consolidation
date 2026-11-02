"""Independent task specialists: one LoRA fine-tune per selected task, each from the ORIGINAL pi05_libero checkpoint,
strictly one at a time.

DEFAULT IS A DRY RUN. Training requires --execute AND an explicit --positions/--indices selection AND
budget_approved: true in the spec. There is no "all tasks" default, no concurrency, no background workers.

  scripts/py-policy scripts/train_specialists.py                                      # plan for every manifest task
  CLVLA_USE_GPU=1 scripts/py-policy scripts/train_specialists.py --indices 78 --execute
  scripts/py-policy scripts/train_specialists.py --summarize                          # CSV/JSON over finished runs
  CLVLA_USE_GPU=1 scripts/py-policy scripts/train_specialists.py --indices 78 --execute --resume-dir outputs/specialists/<run>
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from clvla.config import load_experiment_config
from clvla.specialists import (
    RunnerRefused,
    default_gpu_guard,
    parse_selection,
    run_serial,
    summarize_runs,
    task_seeds,
    write_summary_files,
)
from clvla.tasks import load_manifest
from clvla.train_task import describe, execute_task_run, prepare_task_run
from clvla.training_core import TrainSpec

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(ROOT / "configs/training/specialists.yaml"))
    ap.add_argument("--manifest", default=None, help="explicit manifest path (default: the one named in the spec)")
    ap.add_argument("--positions")
    ap.add_argument("--indices")
    ap.add_argument("--execute", action="store_true")
    ap.add_argument("--resume-dir", default=None, help="resume ONE run; requires exactly one selected task")
    ap.add_argument("--num-steps", type=int, help="override (changes the signature; must stay equal across tasks)")
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--save-interval", type=int)
    ap.add_argument("--val-interval", type=int)
    ap.add_argument("--summarize", action="store_true", help="write summary CSV/JSON over finished runs and exit")
    args = ap.parse_args(argv)

    raw = yaml.safe_load(Path(args.spec).read_text())
    spec = TrainSpec.from_yaml(args.spec).with_overrides(num_steps=args.num_steps, batch_size=args.batch_size,
                                                         save_interval=args.save_interval, val_interval=args.val_interval)
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    output_root = Path(cfg.output_root)
    manifest_path = Path(args.manifest or ROOT / raw["manifest"])
    manifest = load_manifest(manifest_path)
    controls = list(raw.get("control_tasks", []))

    if args.summarize:
        runs = sorted((output_root / spec.kind).glob(f"*_{spec.name}-t*"))
        rows = summarize_runs(runs, control_indices=controls)
        write_summary_files(rows, output_root / spec.kind, stem=f"{spec.name}_summary")
        print(f"{len(rows)} run(s) summarised -> {output_root / spec.kind}/{spec.name}_summary.{{csv,json}}")
        return 0

    try:
        tasks = parse_selection(manifest, args.positions, args.indices, args.execute)
    except RunnerRefused as e:
        raise SystemExit(str(e))
    seeds = task_seeds(cfg.global_seed, tasks)
    print(f"[specialists] manifest {manifest_path} v{manifest.version} ({manifest.content_sha256[:12]}); "
          f"{len(tasks)} task(s); steps {spec.num_steps}; budget_approved={raw.get('budget_approved')}")
    plans = []
    for t in tasks:
        label = " [known-skill control]" if t.libero_90_index in controls else ""
        try:
            plan = prepare_task_run(spec=spec, cfg=cfg, manifest=manifest, libero_90_index=t.libero_90_index,
                                    data_cfg_path=ROOT / raw["data_config"], repo_root=ROOT)
            assert plan["seed"] == seeds[t.libero_90_index]
            plans.append(plan)
            print(f"- t{t.libero_90_index:03d}{label}\n" + describe(plan))
        except Exception as e:  # noqa: BLE001 - plan shows every blocked task instead of stopping at the first
            print(f"- t{t.libero_90_index:03d}{label}: NOT READY: {type(e).__name__}: {e}")
            if args.execute:
                raise SystemExit(f"t{t.libero_90_index}: preflight failed; nothing was started")
    if not args.execute:
        print("[specialists] dry run only.")
        return 0
    if raw.get("budget_approved") is not True:
        raise SystemExit("budget_approved is not true in the spec: approve one fixed step budget for all tasks first")
    if args.resume_dir and len(plans) != 1:
        raise SystemExit("--resume-dir needs exactly one selected task")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") == "":
        raise SystemExit("run with CLVLA_USE_GPU=1")
    from clvla.tracking import require_api_key

    try:
        require_api_key(spec)
    except RuntimeError as e:
        raise SystemExit(str(e))
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(spec.xla_mem_fraction))
    by_index = {p["task"]["libero_90_index"]: p for p in plans}

    def run_one(t):
        out = execute_task_run(by_index[t.libero_90_index], cfg=cfg, manifest=manifest, repo_root=ROOT,
                               output_root=output_root, spec_path=Path(args.spec),
                               resume_dir=Path(args.resume_dir) if args.resume_dir else None)
        return {"run_dir": out["run_dir"], "train_status": out["summary"]["status"]}

    results = run_serial(tasks, run_one, guard=default_gpu_guard(ROOT), lock_root=output_root)
    for r in results:
        print(r)
    rows = summarize_runs(sorted((output_root / spec.kind).glob(f"*_{spec.name}-t*")), control_indices=controls)
    write_summary_files(rows, output_root / spec.kind, stem=f"{spec.name}_summary")
    return 0 if all(r["status"] == "completed" for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
