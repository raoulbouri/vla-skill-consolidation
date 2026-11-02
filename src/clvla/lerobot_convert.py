"""Convert re-rendered (aligned, 256 px) demos into LeRobot datasets in OpenPI's physical-intelligence/libero format.

One task -> TWO physically separate LeRobot datasets with distinct repo ids:
    <namespace>/<version>-t<idx:03d>-train   demos at protocol train_demo_indices (demo_0..demo_44)
    <namespace>/<version>-t<idx:03d>-val     demos at protocol val_demo_indices   (demo_45..demo_49)
OpenPI's loader (data_loader.create_torch_dataset) reads every episode of the repo it is given, so giving the trainer
only the train repo makes sampling validation demos impossible by construction.

Frame format = examples/libero/convert_libero_data_to_lerobot.py at the pinned OpenPI commit:
    image, wrist_image : 256x256x3 uint8, rotated 180 degrees (raw LIBERO obs are OpenGL-oriented; OpenPI's eval client
                         applies the same rotation "to match train preprocessing")
    state              : 8 float32 = eef pos + axis-angle + gripper qpos of the PRE-action observation
    actions            : 7 float32, the original LIBERO action (no transformation)
    task               : exact instruction (used by OpenPI's PromptFromLeRobotTask)
LeRobot writes its own per-episode statistics as part of its format; OpenPI does not use them. Normalization comes
only from the checkpoint's bundled norm_stats.json (never computed here).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

FEATURES = {
    "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "wrist_image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
    "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
    "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
}


def repo_ids(namespace: str, version: str, libero_90_index: int, profile: Optional[str] = None) -> Dict[str, str]:
    base = f"{namespace}/{version}-t{libero_90_index:03d}" + (f"-{profile}" if profile else "")
    return {"train": f"{base}-train", "val": f"{base}-val"}


def resolve_split(data_cfg: Dict[str, Any], cfg: Any, profile: Optional[str] = None) -> Tuple[List[int], List[int]]:
    """(train, val) demo indices: the frozen protocol split, or a named profile from the data config."""
    if not profile:
        return list(cfg.splits.train_demo_indices), list(cfg.splits.val_demo_indices)
    profiles = data_cfg.get("split_profiles") or {}
    if profile not in profiles:
        raise ValueError(f"unknown split profile {profile!r}; defined: {sorted(profiles)}")
    p = profiles[profile]
    tr = list(range(p["train_demo_indices"]["start"], p["train_demo_indices"]["stop"]))
    va = list(range(p["val_demo_indices"]["start"], p["val_demo_indices"]["stop"]))
    if not tr or not va or set(tr) & set(va):
        raise ValueError(f"split profile {profile!r}: empty or overlapping train/val")
    return tr, va


def lerobot_manifest_name(task_name: str, profile: Optional[str] = None) -> str:
    return f"{task_name}.{profile}.lerobot.json" if profile else f"{task_name}.lerobot.json"


def split_demo_keys(demo_keys: Sequence[str], train_idx: Sequence[int], val_idx: Sequence[int]) -> Dict[str, List[str]]:
    """Positional split over numerically ordered demo keys (demo_0, demo_1, ..., demo_49)."""
    from clvla.rerender import check_split_isolation

    ordered = sorted(demo_keys, key=lambda k: int(k.split("_")[1]))
    bad = [i for i in list(train_idx) + list(val_idx) if not 0 <= i < len(ordered)]
    if bad:
        raise ValueError(f"split indices {bad} out of range for {len(ordered)} demos")
    split = {"train": [ordered[i] for i in train_idx], "val": [ordered[i] for i in val_idx]}
    check_split_isolation(split["train"], split["val"], ordered)
    return split


def dataset_fingerprint(root: os.PathLike) -> Dict[str, Any]:
    """sha256 over (relative path, size, sha256) of every file under a LeRobot dataset directory."""
    root = Path(root)
    files = sorted(p for p in root.rglob("*") if p.is_file())
    if not files:
        raise FileNotFoundError(f"empty or missing dataset directory: {root}")
    h = hashlib.sha256()
    total = 0
    for p in files:
        rel = str(p.relative_to(root))
        fh = hashlib.sha256()
        with p.open("rb") as f:
            for block in iter(lambda: f.read(1 << 24), b""):
                fh.update(block)
        size = p.stat().st_size
        total += size
        h.update(f"{rel}\0{size}\0{fh.hexdigest()}\n".encode())
    return {"sha256": h.hexdigest(), "num_files": len(files), "total_bytes": total}


def lerobot_root(repo_id: str, lerobot_home: Optional[str] = None) -> Path:
    """Where OpenPI's loader will look: $HF_LEROBOT_HOME/<repo_id> (lerobot.common.constants.HF_LEROBOT_HOME)."""
    home = lerobot_home or os.environ.get("HF_LEROBOT_HOME")
    if not home:
        raise RuntimeError("HF_LEROBOT_HOME is not set (source env.sh); the trainer would otherwise use ~/.cache")
    return Path(home) / repo_id


def frames_from_processed_demo(g: Any, instruction: str) -> List[Dict[str, Any]]:
    """Frames of one processed (aligned) demo group in OpenPI's LIBERO LeRobot format."""
    av, wr = g["obs/agentview_rgb"][()], g["obs/eye_in_hand_rgb"][()]
    state, actions = g["obs/state"][()].astype(np.float32), g["actions"][()].astype(np.float32)
    T = len(actions)
    if not (len(av) == len(wr) == len(state) == T):
        raise ValueError(f"length mismatch in processed demo: {len(av)}, {len(wr)}, {len(state)}, {T}")
    if not (np.all(np.isfinite(state)) and np.all(np.isfinite(actions))):
        raise ValueError("non-finite state/actions in processed demo")
    return [{
        "image": np.ascontiguousarray(av[t][::-1, ::-1]),
        "wrist_image": np.ascontiguousarray(wr[t][::-1, ::-1]),
        "state": state[t],
        "actions": actions[t],
        "task": instruction,
    } for t in range(T)]


def write_lerobot_dataset(repo_id: str, root: Path, demos: List[Any], instruction: str, *, fps: int, robot_type: str,
                          threads: int, processes: int) -> Dict[str, Any]:
    """Create one LeRobot dataset (refuses if `root` exists) and add each processed demo as one episode, in order."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    if root.exists():
        raise FileExistsError(f"refusing to overwrite existing LeRobot dataset: {root}")
    ds = LeRobotDataset.create(repo_id=repo_id, root=root, robot_type=robot_type, fps=fps, features=FEATURES,
                               image_writer_threads=threads, image_writer_processes=processes)
    lengths = []
    for g in demos:
        frames = frames_from_processed_demo(g, instruction)
        for fr in frames:
            ds.add_frame(fr)
        ds.save_episode()
        lengths.append(len(frames))
    if hasattr(ds, "stop_image_writer"):
        ds.stop_image_writer()
    return {"num_episodes": len(lengths), "num_frames": int(sum(lengths)), "episode_lengths": lengths}


def verify_lerobot_dataset(repo_id: str, root: Path, expected_episodes: int, expected_frames: int,
                           instruction: str) -> Dict[str, Any]:
    """Read-back check of a written dataset through LeRobot's metadata (what OpenPI's loader will see)."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    meta = LeRobotDatasetMetadata(repo_id, root=root)
    problems = []
    if meta.total_episodes != expected_episodes:
        problems.append(f"episodes {meta.total_episodes} != {expected_episodes}")
    if meta.total_frames != expected_frames:
        problems.append(f"frames {meta.total_frames} != {expected_frames}")
    if list(meta.tasks.values()) != [instruction]:
        problems.append(f"tasks {list(meta.tasks.values())} != [{instruction!r}]")
    for k, spec in FEATURES.items():
        if k not in meta.features or tuple(meta.features[k]["shape"]) != tuple(spec["shape"]):
            problems.append(f"feature {k}: {meta.features.get(k)}")
    if problems:
        raise ValueError(f"{repo_id}: " + "; ".join(problems))
    return {"total_episodes": meta.total_episodes, "total_frames": meta.total_frames, "fps": meta.fps}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, indent=1, sort_keys=True)
