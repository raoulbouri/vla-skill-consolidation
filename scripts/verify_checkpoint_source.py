"""Provenance check: compare every local pi05_libero file with the object metadata published in GCS (size + MD5).

Independent of configs/checkpoint.lock.json (which only proves the local copy has not changed since download).
Usage (policy env, network): scripts/py-policy scripts/verify_checkpoint_source.py
"""

from __future__ import annotations

import base64
import hashlib
import sys
from pathlib import Path

import gcsfs

from clvla.config import load_experiment_config

ROOT = Path(__file__).resolve().parents[1]


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return base64.b64encode(h.digest()).decode()


def main() -> int:
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    local = Path(cfg.checkpoint.local_dir)
    remote = cfg.checkpoint.uri.replace("gs://", "").rstrip("/")
    fs = gcsfs.GCSFileSystem(token="anon")
    objects = {o["name"][len(remote) + 1 :]: o for o in fs.find(remote, detail=True).values()}
    bad = []
    for rel, o in sorted(objects.items()):
        p = local / rel
        if not p.is_file():
            bad.append(f"{rel}: missing locally")
        elif p.stat().st_size != int(o["size"]):
            bad.append(f"{rel}: size {p.stat().st_size} != GCS {o['size']}")
        elif o.get("md5Hash") and md5_b64(p) != o["md5Hash"]:
            bad.append(f"{rel}: MD5 differs from GCS")
    extra = sorted(str(p.relative_to(local)) for p in local.rglob("*") if p.is_file())
    extra = [e for e in extra if e not in objects]
    bad += [f"{e}: local file not in GCS" for e in extra]
    print(
        f"{len(objects)} GCS objects under gs://{remote}; {sum(int(o['size']) for o in objects.values()) / 1e9:.2f} GB"
    )
    print("\n".join(bad) if bad else "OK: every local file matches GCS size and MD5")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
