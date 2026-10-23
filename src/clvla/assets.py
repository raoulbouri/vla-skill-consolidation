"""Checkpoint and dataset presence/integrity checks.

Integrity is checked against lock files committed to configs/:
  * configs/checkpoint.lock.json  - every file of the pi05_libero checkpoint: relative path, size, sha256
  * configs/dataset.lock.json     - every LIBERO-90 HDF5: size and sha256 as published by the HF LFS API at a pinned revision

`verify_tree(..., deep=False)` checks presence and exact sizes (seconds); `deep=True` also hashes every file.
Leftover partial/temporary files from an interrupted download are reported as errors.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

PARTIAL_MARKERS = (".partial", ".incomplete", ".tmp", ".lock", ".download")
CHECKPOINT_REQUIRED = ("params/_METADATA", "params/manifest.ocdbt")


class AssetError(RuntimeError):
    pass


def sha256_file(path: os.PathLike | str, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclasses.dataclass
class Report:
    """Collects problems instead of failing on the first one, so one run lists everything that is wrong."""

    name: str
    errors: List[str] = dataclasses.field(default_factory=list)
    checked: int = 0
    total_bytes: int = 0

    @property
    def ok(self) -> bool:
        return not self.errors

    def raise_if_failed(self) -> None:
        if self.errors:
            shown = "\n  - ".join(self.errors[:20])
            more = f"\n  ... and {len(self.errors) - 20} more" if len(self.errors) > 20 else ""
            raise AssetError(f"{self.name}: {len(self.errors)} problem(s):\n  - {shown}{more}")

    def summary(self) -> str:
        status = "OK" if self.ok else f"FAILED ({len(self.errors)} problems)"
        return f"{self.name}: {status}; {self.checked} files, {self.total_bytes / 1e9:.2f} GB checked"


def build_lock(
    root: os.PathLike | str, files: Optional[Iterable[str]] = None, with_sha256: bool = True
) -> Dict[str, Dict]:
    """Record size (+ sha256) of every regular file under root (or of the given relative paths)."""
    root = Path(root)
    rels = (
        sorted(files) if files is not None else sorted(str(p.relative_to(root)) for p in root.rglob("*") if p.is_file())
    )
    out = {}
    for rel in rels:
        p = root / rel
        out[rel] = {"size": p.stat().st_size}
        if with_sha256:
            out[rel]["sha256"] = sha256_file(p)
    return out


def find_partial_files(root: os.PathLike | str) -> List[str]:
    root = Path(root)
    if not root.exists():
        return []
    return sorted(
        str(p.relative_to(root))
        for p in root.rglob("*")
        if p.is_file() and any(p.name.endswith(m) or m + "." in p.name for m in PARTIAL_MARKERS)
    )


def verify_tree(
    root: os.PathLike | str,
    lock: Dict[str, Dict],
    name: str,
    deep: bool = False,
    only: Optional[Iterable[str]] = None,
    check_partials: bool = True,
) -> Report:
    """Verify files under root against a lock mapping {relpath: {size, sha256}}."""
    root = Path(root)
    rep = Report(name)
    if not root.is_dir():
        rep.errors.append(f"directory missing: {root}")
        return rep
    rels = sorted(only) if only is not None else sorted(lock)
    for rel in rels:
        if rel not in lock:
            rep.errors.append(f"{rel}: not in lock file")
            continue
        exp, p = lock[rel], root / rel
        if not p.is_file():
            rep.errors.append(f"{rel}: missing")
            continue
        size = p.stat().st_size
        rep.checked += 1
        rep.total_bytes += size
        if size != exp["size"]:
            kind = "incomplete download (truncated)" if size < exp["size"] else "unexpected size"
            rep.errors.append(f"{rel}: {kind}: {size} bytes, expected {exp['size']}")
            continue
        if deep and exp.get("sha256"):
            got = sha256_file(p)
            if got != exp["sha256"]:
                rep.errors.append(f"{rel}: sha256 mismatch (corrupt): {got[:16]}... expected {exp['sha256'][:16]}...")
    if check_partials:
        for rel in find_partial_files(root):
            rep.errors.append(f"{rel}: leftover partial/temporary download file")
    return rep


# ---------------------------------------------------------------------------------------------------------------------
# Checkpoint


def load_norm_stats_json(path: os.PathLike | str) -> Dict[str, Dict[str, List[float]]]:
    """Parse OpenPI's norm_stats.json ({"norm_stats": {key: {mean, std, q01, q99}}}) without importing OpenPI."""
    data = json.loads(Path(path).read_text())
    if "norm_stats" not in data:
        raise AssetError(f"{path}: not an OpenPI norm_stats.json (no 'norm_stats' key)")
    return data["norm_stats"]


def check_norm_stats(
    stats: Dict[str, Dict[str, List[float]]], state_dim: int, action_dim: int, quantiles: bool = True
) -> List[str]:
    """Return problems with the bundled normalization statistics (empty list = compatible)."""
    errs = []
    for key, dim in (("state", state_dim), ("actions", action_dim)):
        if key not in stats:
            errs.append(f"norm stats have no '{key}' entry (found {sorted(stats)})")
            continue
        fields = ("mean", "std", "q01", "q99") if quantiles else ("mean", "std")
        for f in fields:
            v = stats[key].get(f)
            if v is None:
                errs.append(f"norm stats '{key}' missing '{f}' (pi05 uses quantile normalization)")
            elif len(v) != dim:
                errs.append(f"norm stats '{key}.{f}' has dim {len(v)}, expected {dim}")
            elif not all(isinstance(x, (int, float)) and x == x and abs(x) != float("inf") for x in v):
                errs.append(f"norm stats '{key}.{f}' contains non-finite values")
        if (
            quantiles
            and all(stats[key].get(f) is not None for f in ("q01", "q99"))
            and any(lo > hi for lo, hi in zip(stats[key]["q01"], stats[key]["q99"]))
        ):
            errs.append(f"norm stats '{key}': q01 > q99 in some dimension")
    return errs


def verify_checkpoint(
    ckpt_dir: os.PathLike | str,
    lock: Dict[str, Dict],
    asset_id: str,
    state_dim: int,
    action_dim: int,
    deep: bool = False,
) -> Report:
    ckpt_dir = Path(ckpt_dir)
    rep = verify_tree(ckpt_dir, lock, f"checkpoint {ckpt_dir.name}", deep=deep)
    for rel in CHECKPOINT_REQUIRED:
        if not (ckpt_dir / rel).is_file() and f"{rel}: missing" not in rep.errors:
            rep.errors.append(f"{rel}: missing (required checkpoint file)")
    ns = ckpt_dir / "assets" / asset_id / "norm_stats.json"
    if not ns.is_file():
        rep.errors.append(f"bundled norm stats missing: assets/{asset_id}/norm_stats.json")
    else:
        try:
            rep.errors.extend(check_norm_stats(load_norm_stats_json(ns), state_dim, action_dim))
        except (AssetError, json.JSONDecodeError) as e:
            rep.errors.append(f"bundled norm stats unreadable: {e}")
    return rep


def checkpoint_fingerprint(lock: Dict[str, Dict]) -> str:
    """Identity of a checkpoint = sha256 over its lock (paths, sizes, file hashes)."""
    return hashlib.sha256(json.dumps(lock, sort_keys=True).encode()).hexdigest()


def load_lock(path: os.PathLike | str) -> Dict:
    p = Path(path)
    if not p.is_file():
        raise AssetError(f"lock file missing: {p}")
    return json.loads(p.read_text())
