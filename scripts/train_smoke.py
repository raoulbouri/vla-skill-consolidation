"""Bounded single-task training smoke: LoRA fine-tune pi05_libero on task 78 (manifest v1.0 position 0) for 100 steps.

DEFAULT IS A DRY RUN (verifies inputs that need no GPU and prints the plan). Training requires --execute; the GPU guard
runs first and refuses if another user has compute processes on CLVLA_GPU (nothing is ever killed).

  scripts/py-policy scripts/train_smoke.py                                   # plan
  CLVLA_USE_GPU=1 scripts/py-policy scripts/train_smoke.py --execute         # train (one process, no background)
  CLVLA_USE_GPU=1 scripts/py-policy scripts/train_smoke.py --execute --resume-dir outputs/finetune/<run>

Afterwards (separate processes): scripts/check_trained_policy.py, scripts/smoke_compare.sh. See docs/NEXT_EXPERIMENTS.md.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import yaml

from clvla.config import load_experiment_config
from clvla.specialists import default_gpu_guard, exclusive_training_lock
from clvla.tasks import load_manifest
from clvla.train_task import describe, execute_task_run, prepare_task_run
from clvla.training_core import TrainSpec

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--spec", default=str(ROOT / "configs/training/smoke_task78.yaml"))
    ap.add_argument("--execute", action="store_true", help="train (default: plan only)")
    ap.add_argument("--resume-dir", default=None, help="resume this run (signature must match exactly)")
    ap.add_argument("--deep-checkpoint-check", action="store_true", help="sha256 the base checkpoint (12 GB)")
    args = ap.parse_args(argv)

    spec = TrainSpec.from_yaml(args.spec)
    raw = yaml.safe_load(Path(args.spec).read_text())
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    manifest = load_manifest(cfg.resolve(cfg.task_manifest))
    idx, pos = raw["task"]["libero_90_index"], raw["task"]["manifest_position"]
    if manifest.by_position(pos).libero_90_index != idx:
        raise SystemExit(f"spec task {idx} is not at manifest position {pos}")
    plan = prepare_task_run(spec=spec, cfg=cfg, manifest=manifest, libero_90_index=idx,
                            data_cfg_path=ROOT / raw["data_config"], repo_root=ROOT,
                            deep_checkpoint=args.deep_checkpoint_check)
    print("[smoke plan]\n" + describe(plan))
    if not args.execute:
        print("[smoke plan] dry run only; add --execute to train (CLVLA_USE_GPU=1).")
        return 0
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") in ("", None):
        raise SystemExit("run with CLVLA_USE_GPU=1 (scripts/py-policy sets CUDA_VISIBLE_DEVICES to CLVLA_GPU)")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(spec.xla_mem_fraction))
    output_root = Path(cfg.output_root)
    default_gpu_guard(ROOT)()
    with exclusive_training_lock(output_root):
        out = execute_task_run(plan, cfg=cfg, manifest=manifest, repo_root=ROOT, output_root=output_root,
                               spec_path=Path(args.spec), resume_dir=Path(args.resume_dir) if args.resume_dir else None)
    s = out["summary"]
    print(f"[smoke] {s['status']}: train loss {s.get('final_train_loss')}, val loss {s.get('final_val_loss')}, "
          f"trainable params {s['init']['trainable_params']}, peak GPU {s.get('memory', {}).get('peak_bytes_in_use')}"
          f"\n[smoke] run dir: {out['run_dir']}")
    return 0 if s["status"] == "completed" else 1


if __name__ == "__main__":
    sys.exit(main())
