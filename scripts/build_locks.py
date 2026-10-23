"""Write the asset lock files (run once after downloading; safe to rerun; refuses to change an existing lock).

  configs/checkpoint.lock.json  size + sha256 of every file in the local pi05_libero checkpoint
  configs/dataset.lock.json     size + sha256 of every LIBERO-90 HDF5 as published by the Hugging Face LFS API at the
                                pinned dataset revision (independent of the local copy, which validate_assets.py checks)

Usage (policy env): scripts/py-policy scripts/build_locks.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from clvla.assets import build_lock, checkpoint_fingerprint, find_partial_files
from clvla.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]
DATASET_REPO = "yifengzhu-hf/LIBERO-datasets"
DATASET_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


def write_once(path: Path, payload: dict, force: bool) -> None:
    if path.exists() and not force:
        old = json.loads(path.read_text())
        if old != payload:
            sys.exit(
                f"{path} exists and differs from the freshly computed lock; refusing to overwrite (use --force "
                "only if the asset was deliberately replaced, and document it in docs/ASSETS.md)"
            )
        print(f"{path}: unchanged")
        return
    path.write_text(json.dumps(payload, indent=1, sort_keys=True) + "\n")
    print(f"{path}: written")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-checkpoint", action="store_true")
    args = ap.parse_args()
    cfg = (
        load_experiment_config(ROOT / "configs/experiment.yaml", overrides=None)
        if (ROOT / "configs/experiment.yaml").exists()
        else None
    )
    ckpt_dir = Path(cfg.checkpoint.local_dir) if cfg else None

    if not args.skip_checkpoint:
        assert ckpt_dir is not None and ckpt_dir.is_dir(), f"checkpoint dir missing: {ckpt_dir}"
        partial = find_partial_files(ckpt_dir)
        assert not partial, f"checkpoint has partial files, re-run the download first: {partial}"
        files = build_lock(ckpt_dir)
        write_once(
            ROOT / "configs/checkpoint.lock.json",
            {
                "uri": cfg.checkpoint.uri,
                "openpi_config": cfg.checkpoint.openpi_config,
                "fingerprint_sha256": checkpoint_fingerprint(files),
                "files": files,
            },
            args.force,
        )

    import huggingface_hub

    info = huggingface_hub.HfApi().dataset_info(DATASET_REPO, revision=DATASET_REVISION, files_metadata=True)
    files = {
        s.rfilename: {"size": s.size, "sha256": s.lfs.sha256}
        for s in info.siblings
        if s.rfilename.startswith("libero_90/") and s.lfs is not None
    }
    assert len(files) == 90, f"expected 90 LIBERO-90 files at {DATASET_REVISION}, found {len(files)}"
    write_once(
        ROOT / "configs/dataset.lock.json",
        {
            "repo": f"https://huggingface.co/datasets/{DATASET_REPO}",
            "revision": info.sha,
            "suite": "libero_90",
            "license": "CC BY 4.0 (LIBERO datasets; HF card tag: apache-2.0)",
            "files": files,
        },
        args.force,
    )


if __name__ == "__main__":
    main()
