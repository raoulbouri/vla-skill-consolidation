"""Single-task training run shared by scripts/train_smoke.py and scripts/train_specialists.py.

prepare_task_run(): all checks that need no GPU (base checkpoint identity, manifest/task, datasets and split isolation,
seed, signature, output path) -> a plan dict that is printed in dry runs.
execute_task_run(): creates (or resumes) the run directory, writes metadata, runs clvla.training_core.run_loop with
the OpenPI backend, fingerprints the final trained checkpoint. Imports JAX only here.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from clvla.assets import build_lock, checkpoint_fingerprint
from clvla.experiment import build_run_metadata, git_revision, make_run_dir
from clvla.preflight import verify_base_checkpoint, verify_datasets, verify_task
from clvla.training_core import RunPaths, TrainSpec, open_run, run_loop, run_signature, task_seed


def lerobot_manifest_path(data_cfg: Dict[str, Any], task_name: str, profile: Optional[str] = None) -> Path:
    root = os.environ.get(data_cfg["output"]["processed_root_env"])
    if not root:
        raise RuntimeError(f"{data_cfg['output']['processed_root_env']} not set (source env.sh)")
    from clvla.lerobot_convert import lerobot_manifest_name

    return Path(root) / data_cfg["version"] / "lerobot" / lerobot_manifest_name(task_name, profile)


def prepare_task_run(*, spec: TrainSpec, cfg: Any, manifest: Any, libero_90_index: int, data_cfg_path: Path,
                     repo_root: Path, deep_checkpoint: bool = False, fingerprint_fn: Any = None) -> Dict[str, Any]:
    data_cfg = yaml.safe_load(Path(data_cfg_path).read_text())
    task = verify_task(manifest, libero_90_index, repo_root / "configs/libero90_tasks.json")
    base = verify_base_checkpoint(cfg, repo_root, deep=deep_checkpoint)
    from clvla.lerobot_convert import resolve_split

    profile = spec.raw.get("data_split_profile")  # None = frozen protocol split (0-44 / 45-49)
    train_idx, val_idx = resolve_split(data_cfg, cfg, profile)
    datasets = verify_datasets(lerobot_manifest_path(data_cfg, task["name"], profile), task, train_idx, val_idx,
                               fingerprint_fn=fingerprint_fn)
    datasets["split_profile"] = profile
    seed = task_seed(cfg.global_seed, libero_90_index)
    code = git_revision(repo_root)
    identity = {
        "base_checkpoint_fingerprint": base["fingerprint_sha256"], "norm_stats_sha256": base["norm_stats_sha256"],
        "manifest_sha256": task["manifest_sha256"], "task": task["libero_90_index"], "instruction": task["instruction"],
        "train_dataset": datasets["train"]["fingerprint_sha256"], "val_dataset": datasets["val"]["fingerprint_sha256"],
        "processed_sha256": datasets["processed_sha256"], "seed": seed, "project_commit": code.get("commit"),
    }
    if profile:  # only when set, so protocol-split runs keep their existing signatures
        identity["split_profile"] = profile
    return {"spec": spec, "task": task, "base": base, "datasets": datasets, "seed": seed, "identity": identity,
            "signature": run_signature(spec, identity), "data_cfg": data_cfg, "code": code,
            "run_name": f"{spec.name}-t{libero_90_index:03d}"}


def describe(plan: Dict[str, Any]) -> str:
    s, t, d = plan["spec"], plan["task"], plan["datasets"]
    return (f"  task t{t['libero_90_index']:03d} (pos {t['position']}): {t['instruction']}\n"
            f"  base checkpoint {plan['base']['fingerprint_sha256'][:16]}... ({plan['base']['verification']})\n"
            f"  train {d['train']['repo_id']} ({d['train']['num_episodes']} eps, {d['train']['num_frames']} frames, "
            f"{d['train']['fingerprint_sha256'][:12]})\n"
            f"  val   {d['val']['repo_id']} ({d['val']['num_episodes']} eps, {d['val']['num_frames']} frames)\n"
            f"  {s.mode} {s.paligemma_variant}+{s.action_expert_variant}, lora_b_init={s.lora_b_init}, "
            f"freeze_vision={s.freeze_vision_encoder}; steps {s.num_steps}, batch {s.batch_size}, lr {s.peak_lr} "
            f"(warmup {s.warmup_steps}); val every {s.val_interval}, save every {s.save_interval}\n"
            f"  seed {plan['seed']}; signature {plan['signature'][:16]}")


def execute_task_run(plan: Dict[str, Any], *, cfg: Any, manifest: Any, repo_root: Path, output_root: Path,
                     spec_path: Path, resume_dir: Optional[Path] = None) -> Dict[str, Any]:
    from clvla.openpi_training import OpenPIBackend

    spec: TrainSpec = plan["spec"]
    paths: RunPaths = open_run(lambda: make_run_dir(output_root, spec.kind, plan["run_name"]), plan["signature"],
                               resume_dir=resume_dir)
    if resume_dir is None:
        shutil.copyfile(spec_path, paths.spec)
        meta = build_run_metadata(
            run_id=paths.root.name, kind=spec.kind, config=cfg, manifest=manifest, checkpoint=plan["base"],
            repo_root=repo_root,
            extra={"task": plan["task"], "seed": plan["seed"], "datasets": plan["datasets"],
                   "train_spec": spec.raw, "signature": plan["signature"], "identity": plan["identity"],
                   "processed_data_version": plan["data_cfg"]["version"],
                   "initialisation": "official pi05_libero params (local) via StrictCheckpointWeightLoader; "
                                     f"mode={spec.mode}, lora_b_init={spec.lora_b_init}"})
        paths.metadata.write_text(json.dumps(meta, indent=2, default=str))
    backend = OpenPIBackend(spec, run_dir=paths.root, openpi_dir=repo_root / "third_party/openpi",
                            base_ckpt_dir=plan["base"]["local_dir"], train_repo=plan["datasets"]["train"]["repo_id"],
                            val_repo=plan["datasets"]["val"]["repo_id"], seed=plan["seed"], resume=resume_dir is not None)
    from clvla.tracking import Tracker

    tracker = Tracker(plan, str(paths.root))
    try:
        summary = run_loop(backend, spec, paths, on_record=tracker.on_record)
    except Exception:
        tracker.finish({"status": "failed"})
        raise
    final = Path(summary["checkpoints"][-1]["dir"])
    lock = build_lock(final)
    summary["trained_checkpoint_fingerprint"] = checkpoint_fingerprint(lock)
    summary["trained_checkpoint_files"] = len(lock)
    paths.summary.write_text(json.dumps(summary, indent=1, default=str))
    tracker.finish(summary)
    return {"run_dir": str(paths.root), "summary": summary}
