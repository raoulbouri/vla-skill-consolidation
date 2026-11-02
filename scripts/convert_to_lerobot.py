"""Convert re-rendered demos (scripts/rerender_demos.py output) of selected tasks into separate train / val LeRobot
datasets. DEFAULT IS A DRY RUN; writing requires --execute and an explicit selection.

  scripts/py-policy scripts/convert_to_lerobot.py --indices 78              # plan
  scripts/py-policy scripts/convert_to_lerobot.py --indices 78 --execute    # write t078 train + val datasets

Runs in the policy env (lerobot lives there). Writes <processed_root>/<version>/lerobot/<task>.lerobot.json per task
(immutable; records source/output hashes, demo ids per split and dataset fingerprints) for the trainer to verify.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import yaml

from clvla.config import load_experiment_config
from clvla.experiment import git_revision, package_versions, utc_stamp
from clvla.lerobot_convert import (
    canonical_json,
    dataset_fingerprint,
    lerobot_root,
    repo_ids,
    lerobot_manifest_name,
    resolve_split,
    split_demo_keys,
    verify_lerobot_dataset,
    write_lerobot_dataset,
)
from clvla.rerender import processed_paths, refuse_existing, sha256_file

ROOT = Path(__file__).resolve().parents[1]


def task_lerobot_manifest_path(processed_root, version, name, profile=None):
    return Path(processed_root) / version / "lerobot" / lerobot_manifest_name(name, profile)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-config", default=str(ROOT / "configs/datasets/processed_v1.yaml"))
    ap.add_argument("--indices", required=True, help="comma-separated LIBERO-90 indices (already re-rendered)")
    ap.add_argument("--split-profile", default=None,
                    help="named demo split from the data config (e.g. n25); default: the frozen protocol split")
    ap.add_argument("--execute", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    dcfg = yaml.safe_load(Path(args.data_config).read_text())
    version, lr = dcfg["version"], dcfg["lerobot"]
    processed_root = os.environ.get(dcfg["output"]["processed_root_env"])
    if not processed_root:
        raise SystemExit(f"{dcfg['output']['processed_root_env']} not set (source env.sh)")
    train_idx, val_idx = resolve_split(dcfg, cfg, args.split_profile)
    snapshot = {r["idx"]: r for r in json.loads((ROOT / "configs/libero90_tasks.json").read_text())["libero_90"]}

    for idx in [int(i) for i in args.indices.split(",")]:
        name, instruction = snapshot[idx]["name"], snapshot[idx]["language"]
        paths = processed_paths(processed_root, version, name)
        report = json.loads(paths["report"].read_text()) if paths["report"].is_file() else None
        rid = repo_ids(lr["namespace"], version, idx, args.split_profile)
        roots = {k: lerobot_root(v) for k, v in rid.items()}
        out_manifest = task_lerobot_manifest_path(processed_root, version, name, args.split_profile)
        if not args.execute:
            print(f"[plan] t{idx:03d} {name}\n  in : {paths['hdf5']} ({'ok' if report else 'NOT RENDERED YET'})")
            for k in ("train", "val"):
                print(f"  out: {roots[k]} [{'EXISTS (would refuse)' if roots[k].exists() else 'new'}]")
            print(f"  manifest: {out_manifest}")
            continue
        if report is None:
            raise SystemExit(f"t{idx}: no re-render report at {paths['report']}; run rerender_demos.py first")
        if report.get("max_demos_debug_limit"):
            raise SystemExit(f"t{idx}: report comes from a --max-demos debug render; refusing")
        if report["instruction"] != instruction:
            raise SystemExit(f"t{idx}: instruction mismatch with snapshot")
        refuse_existing(out_manifest, roots["train"], roots["val"])
        if sha256_file(paths["hdf5"]) != report["output_sha256"]:
            raise SystemExit(f"{paths['hdf5']}: sha256 differs from its re-render report; refusing")

        import h5py

        with h5py.File(str(paths["hdf5"]), "r") as f:
            data = f["data"]
            if data.attrs["alignment"] != "pre_action: obs[k] = render(states[k]) paired with actions[k]":
                raise SystemExit(f"{paths['hdf5']}: not a pre-action aligned file")
            split = split_demo_keys(list(data.keys()), train_idx, val_idx)
            written, verified, fps = {}, {}, {}
            for part in ("train", "val"):
                written[part] = write_lerobot_dataset(
                    rid[part], roots[part], [data[k] for k in split[part]], instruction, fps=lr["fps"],
                    robot_type=lr["robot_type"], threads=lr["image_writer_threads"],
                    processes=lr["image_writer_processes"])
                verified[part] = verify_lerobot_dataset(rid[part], roots[part], written[part]["num_episodes"],
                                                        written[part]["num_frames"], instruction)
                fps[part] = dataset_fingerprint(roots[part])
        manifest = {
            "libero_90_index": idx, "name": name, "instruction": instruction, "version": version,
            "split_profile": args.split_profile,
            "source": {"processed_file": report["output_file"], "processed_sha256": report["output_sha256"],
                       "raw_file": report["source_file"], "raw_sha256": report["source_sha256"]},
            "splits": {part: {"repo_id": rid[part], "root": str(roots[part]), "demo_keys": split[part],
                              "protocol_indices": list(train_idx if part == "train" else val_idx),
                              **written[part], "readback": verified[part], "fingerprint": fps[part]}
                       for part in ("train", "val")},
            "format": "OpenPI physical-intelligence/libero LeRobot format (image/wrist_image rotated 180, 8-D state, "
                      "7-D actions, task=instruction)",
            "normalization": "none computed; training uses the checkpoint's bundled norm_stats.json",
            "code": {"project": git_revision(ROOT), "packages": package_versions(["lerobot", "numpy", "h5py"])},
            "created_utc": utc_stamp(),
        }
        out_manifest.parent.mkdir(parents=True, exist_ok=True)
        out_manifest.write_text(canonical_json(manifest))
        print(f"[done] t{idx:03d}: train {written['train']['num_episodes']} eps / {written['train']['num_frames']} "
              f"frames, val {written['val']['num_episodes']} eps; {out_manifest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
