"""Re-render official LIBERO-90 demos with pre-action (deployment-aligned) observations at 256x256.

DEFAULT IS A DRY RUN: prints scope, inputs, outputs and disk estimate; touches nothing. Writing requires --execute
AND an explicit task selection (--positions / --indices, or --all). Raw HDF5 files are opened read-only; existing
outputs are never overwritten (a leftover *.partial also blocks, so a crashed run is never silently mixed).

  scripts/py-sim scripts/rerender_demos.py --positions 0                       # plan only
  scripts/py-sim scripts/rerender_demos.py --positions 0 --execute             # render task 78
  scripts/py-sim scripts/rerender_demos.py --finalize                          # seal <version>/manifest.json once

Runs in the simulator env (Python 3.8). See clvla/rerender.py for the alignment derivation and audits.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from pathlib import Path

import yaml

from clvla.config import load_experiment_config
from clvla.experiment import git_revision, package_versions, utc_stamp
from clvla.rerender import (
    AlignmentAmbiguity,
    ReconstructionError,
    Thresholds,
    build_processed_manifest,
    estimate_disk_bytes,
    processed_paths,
    refuse_existing,
    rerender_episode,
    sha256_file,
)
from clvla.tasks import load_manifest

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default=str(ROOT / "configs/datasets/processed_v1.yaml"))
    ap.add_argument("--manifest", default=None, help="task manifest for --positions (default: protocol manifest)")
    sel = ap.add_mutually_exclusive_group()
    sel.add_argument("--positions", help="comma-separated manifest positions")
    sel.add_argument("--indices", help="comma-separated LIBERO-90 indices (any task, e.g. v1.1 candidates)")
    sel.add_argument("--all", action="store_true", help="all manifest tasks")
    ap.add_argument("--execute", action="store_true", help="actually render and write (default: plan only)")
    ap.add_argument("--max-demos", type=int, default=None, help="debug: only the first N demos (recorded)")
    ap.add_argument("--finalize", action="store_true", help="write the version manifest from finished task reports")
    return ap.parse_args(argv)


def load_data_config(path):
    d = yaml.safe_load(Path(path).read_text())
    for key in ("version", "camera_resolution", "validation_resolution", "source", "output", "thresholds"):
        if key not in d:
            raise SystemExit(f"{path}: missing key {key}")
    return d


def resolve_selection(args, cfg, snapshot):
    """List of dicts: libero_90_index, name, instruction, bddl_file, demo_file, approx frames."""
    rows = {r["idx"]: r for r in snapshot["libero_90"]}
    manifest = load_manifest(args.manifest or cfg.resolve(cfg.task_manifest))
    by_index = {t.libero_90_index: t for t in manifest.tasks}
    if args.positions:
        indices = [manifest.by_position(int(p)).libero_90_index for p in args.positions.split(",")]
    elif args.indices:
        indices = [int(i) for i in args.indices.split(",")]
    else:
        indices = [t.libero_90_index for t in manifest.tasks]
    out = []
    for i in indices:
        if i not in rows:
            raise SystemExit(f"LIBERO-90 index {i} not in configs/libero90_tasks.json")
        r = rows[i]
        t = by_index.get(i)
        approx = (
            int(round(t.extra["demo_len_mean"] * t.extra["n_demos"]))
            if t and t.extra and "demo_len_mean" in t.extra and "n_demos" in t.extra
            else None
        )
        out.append(
            {
                "libero_90_index": i,
                "name": r["name"],
                "instruction": r["language"],
                "bddl_file": f"{r['problem_folder']}/{r['bddl_file']}",
                "demo_file": f"libero_90/{r['name']}_demo.hdf5",
                "approx_frames": approx,
                "manifest_position": t.position if t else None,
            }
        )
    return out


def plan(selection, dcfg, cfg, lock, processed_root):
    total = 0
    print(f"[plan] version {dcfg['version']}  ->  {processed_root}/{dcfg['version']}/   (nothing written)")
    for s in selection:
        src = Path(cfg.data.root) / s["demo_file"]
        frames = s["approx_frames"] or 50 * 200  # conservative when the task is outside the manifest
        est = estimate_disk_bytes(frames, dcfg["camera_resolution"])
        total += est
        out = processed_paths(processed_root, dcfg["version"], s["name"])
        state = "EXISTS (would refuse)" if out["hdf5"].exists() else "new"
        print(
            f"  {s['libero_90_index']:02d} {s['name']}\n      in : {src} ({lock['files'][s['demo_file']]['size'] / 1e9:.2f} GB,"
            f" sha256 {lock['files'][s['demo_file']]['sha256'][:12]}...)\n      out: {out['hdf5']} [{state}]"
            f"\n      ~{frames} frames, <= {est / 1e9:.2f} GB uncompressed images"
        )
    print(
        f"[plan] {len(selection)} task(s); <= {total / 1e9:.1f} GB uncompressed (gzip output is smaller). "
        "Re-run with --execute and an explicit selection to render."
    )


def render_task(s, dcfg, cfg, lock, processed_root, max_demos=None):
    import h5py
    import torch  # noqa: F401  (LIBERO imports it; keep import order identical to evaluation)

    from clvla.data import DEMO_SCHEMA
    from clvla.evaluation import make_libero_env
    from libero.libero import get_libero_path

    th = Thresholds.from_dict(dcfg["thresholds"])
    # Debug renders (--max-demos) never occupy the canonical version directory.
    version = dcfg["version"] + (f"-debug{max_demos}" if max_demos else "")
    paths = processed_paths(processed_root, version, s["name"])
    refuse_existing(paths["hdf5"], paths["report"])
    src = Path(cfg.data.root) / s["demo_file"]
    expected = lock["files"][s["demo_file"]]
    t0 = time.time()
    src_sha = sha256_file(src)
    if src_sha != expected["sha256"]:
        raise SystemExit(f"{src}: sha256 {src_sha} != dataset lock {expected['sha256']}; refusing")
    bddl = Path(get_libero_path("bddl_files")) / s["bddl_file"]
    env = make_libero_env(bddl, dcfg["camera_resolution"], cfg.simulator.env_seed)
    env_small = make_libero_env(bddl, dcfg["validation_resolution"], cfg.simulator.env_seed)
    paths["hdf5"].parent.mkdir(parents=True, exist_ok=True)
    partial = Path(str(paths["hdf5"]) + ".partial")
    comp = {
        "compression": dcfg["output"].get("image_compression"),
        "compression_opts": dcfg["output"].get("image_compression_level"),
    }
    demo_reports = {}
    try:
        with h5py.File(str(src), "r") as fin, h5py.File(str(partial), "w-") as fout:
            src_data = fin["data"]
            keys = sorted(src_data.keys(), key=lambda k: int(k.split("_")[1]))
            if max_demos:
                keys = keys[:max_demos]
            instr = json.loads(src_data.attrs["problem_info"])["language_instruction"]
            if instr != s["instruction"]:
                raise SystemExit(f"instruction mismatch: {instr!r} != {s['instruction']!r}")
            dout = fout.create_group("data")
            for a in ("problem_info", "bddl_file_name", "env_name", "env_args", "tag"):
                if a in src_data.attrs:
                    dout.attrs[a] = src_data.attrs[a]
            dout.attrs.update(
                {
                    "clvla_format": "clvla-aligned-hdf5/1",
                    "clvla_version": version,
                    "alignment": "pre_action: obs[k] = render(states[k]) paired with actions[k]",
                    "image_convention": "opengl (raw LIBERO env obs; OpenPI rotates 180 degrees)",
                    "camera_resolution": dcfg["camera_resolution"],
                    "source_file": s["demo_file"],
                    "source_sha256": src_sha,
                    "num_demos": len(keys),
                }
            )
            for key in keys:
                g = src_data[key]
                ep = {name: g[name][()] for name in DEMO_SCHEMA}
                init_state = g.attrs.get("init_state")
                arrays, rep = rerender_episode(env, env_small, ep, init_state, th)
                go = dout.create_group(key)
                for name, arr in arrays.items():
                    kw = comp if name.endswith("_rgb") else {}
                    if name.endswith("_rgb"):
                        kw = dict(kw, chunks=(1,) + arr.shape[1:])
                    go.create_dataset(name, data=arr, **kw)
                go.attrs["source_demo_key"] = key
                go.attrs["num_samples"] = len(arrays["actions"])
                go.attrs["final_success"] = bool(rep["final"]["success"])
                demo_reports[key] = rep
                print(
                    f"  {s['libero_90_index']:02d} {key}: T={rep['T']} offset={rep['offset']} "
                    f"res={rep['aligned_state_residual_max']:.2e} replay_qpos_med={rep['replay']['qpos_median']:.1e}{'(flagged)' if rep['replay']['suspect'] else ''} "
                    f"final_success={rep['final']['success']}",
                    flush=True,
                )
    except (AlignmentAmbiguity, ReconstructionError) as e:
        print(
            f"STOPPED task {s['libero_90_index']}: {type(e).__name__}: {e}\n"
            f"  partial output left for inspection (never used): {partial}",
            file=sys.stderr,
        )
        raise SystemExit(3)
    finally:
        env.close()
        env_small.close()
    os.replace(partial, paths["hdf5"])
    report = {
        "libero_90_index": s["libero_90_index"],
        "name": s["name"],
        "instruction": s["instruction"],
        "manifest_position": s["manifest_position"],
        "source_file": s["demo_file"],
        "source_sha256": src_sha,
        "output_file": str(paths["hdf5"].relative_to(Path(processed_root) / version)),
        "version": version,
        "output_sha256": sha256_file(paths["hdf5"]),
        "demo_keys": list(demo_reports),
        "num_frames": int(sum(r["T"] for r in demo_reports.values())),
        "max_demos_debug_limit": max_demos,
        "demos": demo_reports,
        "validation": {
            "all_offsets": sorted({r["offset"] for r in demo_reports.values()}),
            "aligned_state_residual_max": max(r["aligned_state_residual_max"] for r in demo_reports.values()),
            "replay_qpos_median_max": max(r["replay"]["qpos_median"] for r in demo_reports.values()),
            "replay_flagged_demos": [k for k, r in demo_reports.items() if r["replay"].get("suspect")],
            "final_success_all": all(r["final"]["success"] for r in demo_reports.values()),
        },
        "created_utc": utc_stamp(),
        "elapsed_s": round(time.time() - t0, 1),
        "hostname": socket.gethostname(),
    }
    refuse_existing(paths["report"])
    paths["report"].write_text(json.dumps(report, indent=1, sort_keys=True))
    print(f"[done] {paths['hdf5']} ({report['num_frames']} frames, {report['elapsed_s']}s)")
    return report


def finalize(dcfg, cfg, processed_root):
    vdir = Path(processed_root) / dcfg["version"]
    reports = [json.loads(p.read_text()) for p in sorted((vdir / "hdf5").glob("*.report.json"))]
    if not reports:
        raise SystemExit(f"no task reports under {vdir}/hdf5")
    if any(r.get("max_demos_debug_limit") for r in reports):
        raise SystemExit("a task was rendered with --max-demos (debug); it cannot enter a sealed version")
    for r in reports:  # outputs must still be intact
        if sha256_file(vdir / r["output_file"]) != r["output_sha256"]:
            raise SystemExit(f"{r['output_file']}: sha256 changed since rendering")
    lock = json.loads((ROOT / dcfg["source"]["lock_file"]).read_text())
    m = build_processed_manifest(
        dcfg["version"],
        reports,
        source={
            "dataset_repo": lock["repo"],
            "dataset_revision": lock["revision"],
            "lock_file": dcfg["source"]["lock_file"],
        },
        thresholds=Thresholds.from_dict(dcfg["thresholds"]),
        code={
            "project": git_revision(ROOT),
            "libero": git_revision(ROOT / "third_party/openpi/third_party/libero"),
            "packages": package_versions(["numpy", "mujoco", "robosuite", "libero", "h5py"]),
        },
    )
    out = vdir / "manifest.json"
    if out.exists():
        old = json.loads(out.read_text())
        if old["content_sha256"] == m["content_sha256"]:
            print(f"{out}: unchanged ({m['content_sha256'][:12]})")
            return
        raise SystemExit(f"{out} exists with different content; versions are immutable (use a new version string)")
    out.write_text(json.dumps(m, indent=1, sort_keys=True))
    print(f"[finalize] {out}: {len(reports)} tasks, content_sha256 {m['content_sha256']}")


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    dcfg = load_data_config(args.data_config)
    processed_root = os.environ.get(dcfg["output"]["processed_root_env"])
    if not processed_root:
        raise SystemExit(f"{dcfg['output']['processed_root_env']} not set (source env.sh)")
    if args.finalize:
        finalize(dcfg, cfg, processed_root)
        return 0
    lock = json.loads((ROOT / dcfg["source"]["lock_file"]).read_text())
    snapshot = json.loads((ROOT / "configs/libero90_tasks.json").read_text())
    selection = resolve_selection(args, cfg, snapshot)
    if not args.execute:
        plan(selection, dcfg, cfg, lock, processed_root)
        return 0
    if not (args.positions or args.indices or args.all):
        raise SystemExit("--execute requires an explicit selection (--positions, --indices or --all)")
    for s in selection:
        render_task(s, dcfg, cfg, lock, processed_root, max_demos=args.max_demos)
    return 0


if __name__ == "__main__":
    sys.exit(main())
