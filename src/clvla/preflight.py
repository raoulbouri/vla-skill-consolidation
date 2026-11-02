"""Pre-run verification shared by the training smoke and the specialist runner (pure Python; no JAX).

Every check raises PreflightError with an actionable message. `verify_training_inputs` returns the identity dict that
goes into the run signature and metadata.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from clvla.assets import checkpoint_fingerprint, load_lock, sha256_file, verify_checkpoint

REQUIRED_CHECKPOINT_FINGERPRINT = "cd22f6becde68e647885fb9beb54a6a8fc296b117d87b6de03dea9d53dd07fee"
NORM_STATS_REL = "assets/physical-intelligence/libero/norm_stats.json"


class PreflightError(RuntimeError):
    pass


def verify_base_checkpoint(cfg: Any, repo_root: Path, deep: bool = False) -> Dict[str, Any]:
    """The source checkpoint must be the exact official pi05_libero (lock fingerprint) and intact on disk."""
    lock = load_lock(repo_root / cfg.checkpoint.lock_file)
    fp = checkpoint_fingerprint(lock["files"])
    if fp != lock["fingerprint_sha256"] or fp != REQUIRED_CHECKPOINT_FINGERPRINT:
        raise PreflightError(f"checkpoint lock fingerprint {fp} != required {REQUIRED_CHECKPOINT_FINGERPRINT}; "
                             "never train from a different checkpoint")
    rep = verify_checkpoint(cfg.checkpoint.local_dir, lock["files"], cfg.checkpoint.norm_stats_asset_id,
                            cfg.policy.state_dim, cfg.policy.action_dim, deep=deep)
    if not rep.ok:
        raise PreflightError("base checkpoint invalid:\n  - " + "\n  - ".join(rep.errors))
    ns = Path(cfg.checkpoint.local_dir) / NORM_STATS_REL
    ns_sha = sha256_file(ns)
    if ns_sha != lock["files"][NORM_STATS_REL]["sha256"]:
        raise PreflightError(f"bundled norm stats sha256 {ns_sha} != lock; refusing")
    return {"fingerprint_sha256": fp, "local_dir": cfg.checkpoint.local_dir, "params_dir": str(Path(cfg.checkpoint.local_dir) / "params"),
            "norm_stats_sha256": ns_sha, "verification": "sha256" if deep else "sizes+norm_stats_sha256"}


def verify_task(manifest: Any, libero_90_index: int, snapshot_path: Path) -> Dict[str, Any]:
    t = manifest.by_index(libero_90_index)
    snap = {r["idx"]: r for r in json.loads(Path(snapshot_path).read_text())["libero_90"]}
    if snap[libero_90_index]["name"] != t.name or snap[libero_90_index]["language"] != t.instruction:
        raise PreflightError(f"task {libero_90_index}: manifest entry disagrees with the LIBERO-90 snapshot")
    return {"libero_90_index": t.libero_90_index, "position": t.position, "name": t.name, "instruction": t.instruction,
            "manifest_path": manifest.path, "manifest_version": manifest.version,
            "manifest_sha256": manifest.content_sha256}


def verify_datasets(lerobot_manifest_path: Path, task: Dict[str, Any], train_idx: Any, val_idx: Any,
                    fingerprint_fn: Optional[Any] = None) -> Dict[str, Any]:
    """LeRobot train/val datasets: present, matching the task, split-isolated, fingerprints unchanged."""
    from clvla.lerobot_convert import dataset_fingerprint
    from clvla.rerender import check_split_isolation

    fingerprint_fn = fingerprint_fn or dataset_fingerprint
    if not Path(lerobot_manifest_path).is_file():
        raise PreflightError(f"no LeRobot manifest at {lerobot_manifest_path}: run rerender_demos.py + convert_to_lerobot.py")
    m = json.loads(Path(lerobot_manifest_path).read_text())
    if m["libero_90_index"] != task["libero_90_index"] or m["instruction"] != task["instruction"]:
        raise PreflightError(f"{lerobot_manifest_path}: task/instruction mismatch with the manifest")
    tr, va = m["splits"]["train"], m["splits"]["val"]
    if tr["repo_id"] == va["repo_id"] or tr["root"] == va["root"]:
        raise PreflightError("train and validation datasets must be distinct repos")
    if list(tr["protocol_indices"]) != list(train_idx) or list(va["protocol_indices"]) != list(val_idx):
        raise PreflightError("dataset splits differ from the protocol's train/val demo indices")
    check_split_isolation(tr["demo_keys"], va["demo_keys"], tr["demo_keys"] + va["demo_keys"])
    out = {"lerobot_manifest": str(lerobot_manifest_path), "processed_sha256": m["source"]["processed_sha256"],
           "raw_sha256": m["source"]["raw_sha256"], "version": m["version"]}
    for part, s in (("train", tr), ("val", va)):
        home = os.environ.get("HF_LEROBOT_HOME")
        if not home or Path(s["root"]) != Path(home) / s["repo_id"]:
            raise PreflightError(f"{part} dataset root {s['root']} is not $HF_LEROBOT_HOME/{s['repo_id']} "
                                 "(OpenPI's loader would read a different location)")
        fp = fingerprint_fn(s["root"])
        if fp["sha256"] != s["fingerprint"]["sha256"]:
            raise PreflightError(f"{part} dataset {s['repo_id']} changed since conversion (fingerprint mismatch)")
        out[part] = {"repo_id": s["repo_id"], "fingerprint_sha256": fp["sha256"], "num_episodes": s["num_episodes"],
                     "num_frames": s["num_frames"], "demo_keys": s["demo_keys"]}
    return out


def verify_output_path(path: Path) -> None:
    if path.exists():
        raise PreflightError(f"output path already exists, refusing to overwrite: {path}")
