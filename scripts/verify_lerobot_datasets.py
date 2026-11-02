"""Verify the converted LeRobot datasets end to end (CPU, read-only). Exit code 1 if anything is wrong.

Per task and per split (train / val) it checks:
  1. meta/info.json: image + wrist_image declared (256,256,3), state (8,), actions (7,), fps, episode/frame counts
  2. EVERY image (PNG header, read from the parquet files where this LeRobot version embeds them): 256 x 256,
     and image count == 2 x frames
  3. tasks.jsonl holds exactly the task's instruction
  4. content read back through LeRobotDataset (the loader OpenPI uses) for sampled frames, incl. first/last frames of
     every episode boundary sampled: image/wrist_image == processed HDF5 frame rotated 180 deg (lossless PNG), state ==
     obs/state, action chunk rows == actions[t:t+10], train/val demo keys disjoint and in protocol order
  5. dataset fingerprints equal the ones recorded at conversion (skip with --skip-fingerprint)

  scripts/py-policy scripts/verify_lerobot_datasets.py [--indices 78,84] [--samples 24] [--skip-fingerprint]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import struct
import sys
from pathlib import Path

import h5py
import numpy as np
import yaml

from clvla.config import load_experiment_config
from clvla.lerobot_convert import FEATURES, dataset_fingerprint, lerobot_manifest_name, resolve_split
from clvla.tasks import load_manifest

ROOT = Path(__file__).resolve().parents[1]
HORIZON = 10


def png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    if head[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"not a PNG: {path}")
    return struct.unpack(">II", head[16:24])


def check_images(root):
    """Every image of every frame, embedded in the parquet files (this LeRobot version) or stored as PNG files:
    returns (number of images, list of the ones that are not 256x256 PNGs)."""
    import pyarrow.parquet as pq

    n, wrong = 0, []
    for pf in sorted((root / "data").glob("chunk-*/episode_*.parquet")):
        table = pq.read_table(pf, columns=["image", "wrist_image"])
        for col in ("image", "wrist_image"):
            for cell in table[col].to_pylist():
                n += 1
                try:
                    data = cell.get("bytes")
                    if data is not None:
                        if data[:8] != b"\x89PNG\r\n\x1a\n":
                            raise ValueError("not a PNG")
                        size = struct.unpack(">II", data[16:24])
                    else:
                        size = png_size(root / cell["path"])
                except Exception as e:  # noqa: BLE001 - reported below
                    size = ("error", str(e))
                if size != (256, 256):
                    wrong.append((pf.name, col, size))
    return n, wrong


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indices", default=None, help="comma-separated LIBERO-90 indices (default: all manifest tasks)")
    ap.add_argument("--samples", type=int, default=24)
    ap.add_argument("--skip-fingerprint", action="store_true")
    ap.add_argument("--profile", default=None, help="split profile (e.g. n25); default: the protocol split")
    args = ap.parse_args(argv)

    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    manifest = load_manifest(cfg.resolve(cfg.task_manifest))
    dcfg = yaml.safe_load((ROOT / "configs/datasets/processed_v1.yaml").read_text())
    proot = Path(os.environ[dcfg["output"]["processed_root_env"]])
    version = dcfg["version"]
    train_idx, val_idx = resolve_split(dcfg, cfg, args.profile)
    wanted = {int(i) for i in args.indices.split(",")} if args.indices else None
    rng = random.Random(0)
    problems, rows = [], []

    for t in manifest.tasks:
        if wanted is not None and t.libero_90_index not in wanted:
            continue
        tag = f"t{t.libero_90_index:03d}"
        lm_path = proot / version / "lerobot" / lerobot_manifest_name(t.name, args.profile)
        if not lm_path.is_file():
            problems.append(f"{tag}: not converted (no {lm_path.name})")
            continue
        lm = json.loads(lm_path.read_text())
        hdf5 = proot / version / lm["source"]["processed_file"]
        tr_keys, va_keys = lm["splits"]["train"]["demo_keys"], lm["splits"]["val"]["demo_keys"]
        if set(tr_keys) & set(va_keys) or tr_keys != [f"demo_{i}" for i in train_idx] \
                or va_keys != [f"demo_{i}" for i in val_idx]:
            problems.append(f"{tag}: train/val demo keys are not the expected split "
                            f"({train_idx[0]}-{train_idx[-1]} / {val_idx[0]}-{val_idx[-1]}) or overlap")
        for part in ("train", "val"):
            s, root = lm["splits"][part], Path(lm["splits"][part]["root"])
            ptag, bad0 = f"{tag}/{part}", len(problems)
            info = json.loads((root / "meta/info.json").read_text())
            for name, spec in FEATURES.items():
                f = info["features"].get(name)
                if f is None or list(f["shape"]) != list(spec["shape"]) or f["dtype"] != spec["dtype"]:
                    problems.append(f"{ptag}: feature {name} is {f}, expected shape {list(spec['shape'])} dtype {spec['dtype']}")
            if info["fps"] != dcfg["lerobot"]["fps"]:
                problems.append(f"{ptag}: fps {info['fps']}")
            if info["total_episodes"] != s["num_episodes"] or info["total_frames"] != s["num_frames"]:
                problems.append(f"{ptag}: counts {info['total_episodes']}/{info['total_frames']} != manifest {s['num_episodes']}/{s['num_frames']}")
            tasks = [json.loads(line)["task"] for line in (root / "meta/tasks.jsonl").read_text().splitlines() if line.strip()]
            if tasks != [t.instruction]:
                problems.append(f"{ptag}: tasks.jsonl {tasks} != [{t.instruction!r}]")
            n_img, wrong = check_images(root)
            if n_img != 2 * s["num_frames"]:
                problems.append(f"{ptag}: {n_img} images found, expected {2 * s['num_frames']}")
            if wrong:
                problems.append(f"{ptag}: {len(wrong)} images are not 256x256, e.g. {wrong[0]}")

            lengths = s["episode_lengths"]
            starts = np.concatenate([[0], np.cumsum(lengths)])
            ds = LeRobotDataset(s["repo_id"], root=root, delta_timestamps={"actions": [j / info["fps"] for j in range(HORIZON)]})
            picks = {int(starts[e]) for e in range(len(lengths))} | {int(starts[e + 1] - 1) for e in range(len(lengths))}
            picks |= {rng.randrange(s["num_frames"]) for _ in range(args.samples)}
            checked = 0
            with h5py.File(str(hdf5), "r") as h:
                for gi in sorted(picks):
                    e = int(np.searchsorted(starts, gi, side="right") - 1)
                    ti = gi - int(starts[e])
                    g = h["data"][s["demo_keys"][e]]
                    item = ds[gi]
                    for cam, src in (("image", "obs/agentview_rgb"), ("wrist_image", "obs/eye_in_hand_rgb")):
                        arr = np.asarray(item[cam])
                        if arr.shape != (3, 256, 256):
                            problems.append(f"{ptag}: item {gi} {cam} has shape {arr.shape}")
                            continue
                        got = (arr.transpose(1, 2, 0) * 255).round().astype(np.uint8)
                        if not np.array_equal(got, np.ascontiguousarray(g[src][ti][::-1, ::-1])):
                            problems.append(f"{ptag}: item {gi} ({s['demo_keys'][e]} t={ti}) {cam} differs from the processed HDF5 (rotated 180)")
                    if not np.allclose(np.asarray(item["state"]), g["obs/state"][ti], atol=1e-6):
                        problems.append(f"{ptag}: item {gi} state differs")
                    acts = np.asarray(item["actions"])
                    pad = np.asarray(item["actions_is_pad"]).astype(bool) if "actions_is_pad" in item else np.zeros(HORIZON, bool)
                    ref = g["actions"][()].astype(np.float32)
                    if acts.shape != (HORIZON, 7) or any(not np.allclose(acts[j], ref[ti + j], atol=1e-6) for j in range(HORIZON) if not pad[j] and ti + j < len(ref)):
                        problems.append(f"{ptag}: item {gi} action chunk differs from actions[{ti}:{ti + HORIZON}]")
                    if item["task"] != t.instruction:
                        problems.append(f"{ptag}: item {gi} task {item['task']!r}")
                    checked += 1
            fp_ok = "skipped"
            if not args.skip_fingerprint:
                fp = dataset_fingerprint(root)["sha256"]
                fp_ok = "ok" if fp == s["fingerprint"]["sha256"] else "MISMATCH"
                if fp_ok != "ok":
                    problems.append(f"{ptag}: fingerprint changed since conversion")
            rows.append(f"  {ptag:12s} eps {s['num_episodes']:2d} frames {s['num_frames']:5d} | {n_img} images, "
                        f"{'all 256x256' if not wrong else str(len(wrong)) + ' WRONG'} | content {checked} frames "
                        f"{'ok' if len(problems) == bad0 else 'PROBLEMS'} | fingerprint {fp_ok}")
            print(rows[-1], flush=True)

    print(f"\n{len(rows)} datasets checked")
    for p in problems:
        print("PROBLEM:", p)
    print("ALL LEROBOT DATASETS VERIFIED (256 px, aligned, split-isolated)" if not problems and rows else "VERIFICATION FAILED")
    return 0 if not problems and rows else 1


if __name__ == "__main__":
    sys.exit(main())
