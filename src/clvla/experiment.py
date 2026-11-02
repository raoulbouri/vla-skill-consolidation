"""Run metadata, checkpoint references and reproducible output directories.

Layout: <output_root>/<kind>/<UTC timestamp>_<name>/   with kind in RUN_KINDS. Checkpoint-baseline evaluations,
fine-tuning runs and continual-learning runs therefore never share a directory. Every run directory gets:
  run_metadata.json   config (resolved), protocol + manifest hashes, seeds, checkpoint identity, code revisions,
                      environment versions, host/GPU, command line
  config.snapshot.yaml / task_manifest.snapshot.yaml   byte copies of the inputs, so the run can be re-executed
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import platform
import re
import shutil
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

RUN_KINDS = ("baseline", "finetune", "continual", "smoke", "specialists", "specialist_eval", "screening", "teacher_eval")
REQUIRED_METADATA = (
    "run_id",
    "kind",
    "created_utc",
    "command",
    "config",
    "protocol_sha256",
    "task_manifest",
    "seeds",
    "checkpoint",
    "code",
    "environment",
)
DEFAULT_PACKAGES = (
    "numpy",
    "torch",
    "jax",
    "jaxlib",
    "flax",
    "orbax-checkpoint",
    "mujoco",
    "robosuite",
    "libero",
    "openpi",
    "openpi-client",
    "h5py",
    "pyyaml",
    "opencv-python",
    "pillow",
    "transformers",
    "bddl",
    "robomimic",
    "gym",
)


def utc_stamp() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def make_run_dir(output_root: os.PathLike | str, kind: str, name: str, stamp: Optional[str] = None) -> Path:
    if kind not in RUN_KINDS:
        raise ValueError(f"run kind {kind!r} not in {RUN_KINDS}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"run name {name!r} must match [A-Za-z0-9_.-]+")
    d = Path(output_root) / kind / f"{stamp or utc_stamp()}_{name}"
    if d.exists():
        raise FileExistsError(f"run directory already exists, refusing to overwrite: {d}")
    d.mkdir(parents=True)
    return d


def git_revision(path: os.PathLike | str) -> Dict[str, Any]:
    def run(*args):
        return subprocess.run(
            ["git", "-C", str(path), *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        sha = run("rev-parse", "HEAD")
        dirty = bool(run("status", "--porcelain", "--untracked-files=no"))
        return {"path": str(path), "commit": sha, "dirty": dirty}
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        return {"path": str(path), "commit": None, "dirty": None, "error": str(e).strip()[:200]}


def package_versions(names: Iterable[str] = DEFAULT_PACKAGES) -> Dict[str, Optional[str]]:
    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover
        import importlib_metadata as metadata  # type: ignore
    out = {}
    for n in names:
        try:
            out[n] = metadata.version(n)
        except metadata.PackageNotFoundError:
            out[n] = None
    return out


def gpu_info() -> Dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout
        gpus = [
            dict(zip(("index", "name", "memory_total", "driver"), [x.strip() for x in line.split(",")]))
            for line in q.strip().splitlines()
        ]
    except (OSError, subprocess.SubprocessError):
        gpus = []
    return {"cuda_visible_devices": visible, "gpus": gpus}


def environment_info(packages: Iterable[str] = DEFAULT_PACKAGES) -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "packages": package_versions(packages),
        "gpu": gpu_info(),
        "env": {
            k: os.environ.get(k)
            for k in (
                "MUJOCO_GL",
                "MUJOCO_EGL_DEVICE_ID",
                "CUDA_VISIBLE_DEVICES",
                "OPENPI_DATA_HOME",
                "LIBERO_DATA_ROOT",
                "LIBERO_CONFIG_PATH",
            )
        },
    }


def build_run_metadata(
    *,
    run_id: str,
    kind: str,
    config: Any,
    manifest: Any,
    checkpoint: Dict[str, Any],
    repo_root: os.PathLike | str,
    extra: Optional[Dict[str, Any]] = None,
    env_info: Optional[Dict[str, Any]] = None,
    code: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    repo_root = Path(repo_root)
    if code is None:
        code = {
            "project": git_revision(repo_root),
            "openpi": git_revision(repo_root / "third_party/openpi"),
            "libero": git_revision(repo_root / "third_party/openpi/third_party/libero"),
        }
    meta = {
        "run_id": run_id,
        "kind": kind,
        "created_utc": utc_stamp(),
        "command": list(sys.argv),
        "config": {"path": config.path, "experiment_name": config.experiment_name, "resolved": config.raw},
        "protocol_sha256": config.protocol_sha256,
        "protocol_version": config.protocol_version,
        "task_manifest": {
            "path": manifest.path,
            "version": manifest.version,
            "content_sha256": manifest.content_sha256,
            "task_ids": [t.task_id for t in manifest.tasks],
        },
        "seeds": {"global": config.global_seed, "eval": config.eval_seeds, "env_seed": config.simulator.env_seed},
        "checkpoint": checkpoint,
        "code": code,
        "environment": env_info if env_info is not None else environment_info(),
    }
    if extra:
        meta.update(extra)
    missing = [k for k in REQUIRED_METADATA if meta.get(k) in (None, "", {})]
    if missing:
        raise ValueError(f"run metadata incomplete, missing: {missing}")
    return meta


def write_run_metadata(run_dir: Path, meta: Dict[str, Any], config_path: str, manifest_path: str) -> None:
    (run_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=str))
    shutil.copyfile(config_path, run_dir / "config.snapshot.yaml")
    shutil.copyfile(manifest_path, run_dir / "task_manifest.snapshot.yaml")


def checkpoint_identity(config: Any, lock: Dict[str, Any], fingerprint: str, verified: str) -> Dict[str, Any]:
    return {
        "openpi_config": config.checkpoint.openpi_config,
        "uri": config.checkpoint.uri,
        "local_dir": config.checkpoint.local_dir,
        "norm_stats_asset_id": config.checkpoint.norm_stats_asset_id,
        "fingerprint_sha256": fingerprint,
        "num_files": len(lock),
        "total_bytes": sum(v["size"] for v in lock.values()),
        "verification": verified,
    }
