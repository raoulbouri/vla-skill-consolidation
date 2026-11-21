"""Download the teachers a continual-learning config needs from the public Hugging Face repo, check every file's sha256,
and make them available at the paths the config names.
  scripts/py-policy scripts/fetch_cl_assets.py [--config configs/cl_opd.yaml] [--tasks t035,t063] [--dry-run]
The config's teacher paths are outputs/teachers/<variant>/<task>; the HF repo stores the same files (byte-identical) at
outputs/teacher_deltas/<variant>/<task>. Files go to <repo>/outputs/teacher_deltas/..., and outputs/teachers/<variant>/<task>
becomes a relative symlink to them. A teacher that already exists at the config path is hash-checked and left alone.
About 210 MB per LoRA teacher. The base pi0.5 checkpoint and the demonstrations are separate (README, Setup).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
REPO = os.environ.get("CLVLA_HF_REPO", "raoulb091602/vla-skill-consolidation")
SUMS = "checksums/vla-skill-consolidation_pkg3_teacher-deltas_20261102.SHA256SUMS"
CFG_PREFIX, HF_PREFIX = "outputs/teachers/", "outputs/teacher_deltas/"
FILES = ("delta.npz", "leaf_sha.json", "summary.json", "train_spec.yaml")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/cl_opd.yaml")
    ap.add_argument("--tasks", help="comma-separated task ids (default: every task in the config)")
    ap.add_argument("--dry-run", action="store_true", help="only list what would be downloaded")
    a = ap.parse_args()
    from huggingface_hub import hf_hub_download

    tasks = yaml.safe_load((ROOT / a.config).read_text())["tasks"]
    if a.tasks:
        want = set(a.tasks.split(","))
        assert want <= {t["id"] for t in tasks}, f"unknown task ids: {sorted(want - {t['id'] for t in tasks})}"
        tasks = [t for t in tasks if t["id"] in want]
    sums_path = Path(hf_hub_download(REPO, SUMS))
    sums = {}
    for line in sums_path.read_text().splitlines():
        h, p = line.split(None, 1)
        sums[p.strip()] = h
    bad = 0
    for t in tasks:
        cfg_dir = t["teacher"]
        assert cfg_dir.startswith(CFG_PREFIX), f"{t['id']}: teacher path {cfg_dir} is not under {CFG_PREFIX}"
        hf_dir = HF_PREFIX + cfg_dir[len(CFG_PREFIX):]
        want = {f: sums.get(f"{hf_dir}/{f}") for f in FILES}
        assert want["delta.npz"] and want["leaf_sha.json"], f"{t['id']}: {hf_dir} is not in {SUMS}"
        dest = ROOT / cfg_dir
        if dest.exists() and not dest.is_symlink():
            ok = all(sha256(dest / f) == h for f, h in want.items() if h and (dest / f).exists())
            print(f"{t['id']}: present at {cfg_dir}, hashes {'match' if ok else 'DIFFER'}")
            bad += not ok
            continue
        if a.dry_run:
            print(f"{t['id']}: would download {hf_dir}/ and link {cfg_dir}")
            continue
        for f, h in want.items():
            if not h:
                continue
            p = Path(hf_hub_download(REPO, f"{hf_dir}/{f}", local_dir=ROOT))
            if sha256(p) != h:
                print(f"{t['id']}: {f} sha256 MISMATCH")
                bad += 1
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(os.path.relpath(ROOT / hf_dir, dest.parent))
        print(f"{t['id']}: ok ({hf_dir} -> {cfg_dir})")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
