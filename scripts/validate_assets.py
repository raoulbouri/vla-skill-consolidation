"""Validate config, manifest, checkpoint and demonstrations. Exit code 1 if anything is wrong. Safe to rerun.

Default: checkpoint file sizes + norm stats; all 90 LIBERO-90 file sizes; sha256 + full schema/value validation of
the 10 manifest tasks' demo files. --deep additionally sha256-hashes every checkpoint file; --deep-all every
LIBERO-90 file (~67 GB).

Usage: scripts/py-policy scripts/validate_assets.py [--deep] [--deep-all]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from clvla.assets import Report, find_partial_files, load_lock, verify_checkpoint, verify_tree
from clvla.config import load_experiment_config
from clvla.data import DemoError, validate_demo_file
from clvla.experiment import utc_stamp
from clvla.tasks import check_against_snapshot, load_manifest

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deep", action="store_true", help="sha256 every checkpoint file")
    ap.add_argument("--deep-all", action="store_true", help="sha256 every LIBERO-90 file too")
    ap.add_argument("--config", default=str(ROOT / "configs/experiment.yaml"))
    args = ap.parse_args()
    t0 = time.time()
    cfg = load_experiment_config(args.config)
    manifest = load_manifest(cfg.resolve(cfg.task_manifest))
    check_against_snapshot(manifest, ROOT / "configs/libero90_tasks.json")
    print(
        f"config OK (protocol {cfg.protocol_version}, sha {cfg.protocol_sha256[:12]}); manifest OK "
        f"(v{manifest.version}, sha {manifest.content_sha256[:12]}, {len(manifest.tasks)} tasks)"
    )

    reports = []
    ck_lock = load_lock(cfg.resolve(cfg.checkpoint.lock_file))
    rep = verify_checkpoint(
        cfg.checkpoint.local_dir,
        ck_lock["files"],
        cfg.checkpoint.norm_stats_asset_id,
        cfg.policy.state_dim,
        cfg.policy.action_dim,
        deep=args.deep or args.deep_all,
    )
    reports.append(rep)
    print(rep.summary(), "(sha256)" if args.deep or args.deep_all else "(sizes)")

    ds_lock = load_lock(cfg.resolve(cfg.data.lock_file))
    root = Path(cfg.data.root)
    rep = verify_tree(
        root, ds_lock["files"], "LIBERO-90 demos (all 90, sizes)", deep=args.deep_all, check_partials=False
    )
    rep.errors += [f"libero_90/{p}: leftover partial download file" for p in find_partial_files(root / "libero_90")]
    reports.append(rep)
    print(rep.summary())

    sel = [t.demo_file for t in manifest.tasks]
    rep = verify_tree(root, ds_lock["files"], "manifest demos (sha256)", deep=True, only=sel, check_partials=False)
    for t in manifest.tasks:
        if t.demo_sha256 != ds_lock["files"][t.demo_file]["sha256"]:
            rep.errors.append(f"{t.task_id}: manifest demo_sha256 disagrees with dataset lock")
    reports.append(rep)
    print(rep.summary())

    rep = Report("manifest demos (schema, shapes, finite values, instruction)")
    demo_stats = {}
    for t in manifest.tasks:
        try:
            idx = validate_demo_file(
                root / t.demo_file,
                expected_instruction=t.instruction,
                expected_demos=cfg.data.demos_per_task,
                expected_image_hw=(cfg.data.image_size, cfg.data.image_size),
            )
            rep.checked += 1
            demo_stats[t.task_id] = {
                "n_demos": idx.n_demos,
                "len_min": min(idx.lengths),
                "len_max": max(idx.lengths),
                "train": len(idx.split(cfg.splits.train_demo_indices)),
                "val": len(idx.split(cfg.splits.val_demo_indices)),
            }
        except DemoError as e:
            rep.errors.append(str(e))
    reports.append(rep)
    print(rep.summary())

    ok = all(r.ok for r in reports)
    out_dir = Path(cfg.output_root) / "asset_validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{utc_stamp()}.json"
    out.write_text(
        json.dumps(
            {
                "ok": ok,
                "elapsed_s": round(time.time() - t0, 1),
                "deep": args.deep,
                "deep_all": args.deep_all,
                "checkpoint": {
                    "local_dir": cfg.checkpoint.local_dir,
                    "fingerprint_sha256": ck_lock["fingerprint_sha256"],
                },
                "dataset": {"root": str(root), "repo": ds_lock["repo"], "revision": ds_lock["revision"]},
                "manifest": {"version": manifest.version, "content_sha256": manifest.content_sha256},
                "demo_stats": demo_stats,
                "reports": [
                    {"name": r.name, "ok": r.ok, "checked": r.checked, "bytes": r.total_bytes, "errors": r.errors}
                    for r in reports
                ],
            },
            indent=1,
        )
    )
    for r in reports:
        for e in r.errors:
            print(f"  ERROR [{r.name}] {e}")
    print(f"{'ALL ASSETS VALID' if ok else 'ASSET VALIDATION FAILED'} ({time.time() - t0:.0f}s); report: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
