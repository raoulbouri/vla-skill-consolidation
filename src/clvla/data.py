"""Demonstration indexing, episode loading and schema checks for the official LIBERO HDF5 demos.

Mapping to pi05_libero's training samples. Frames use the LeRobot dataset keys of physical-intelligence/libero
(examples/libero/convert_libero_data_to_lerobot.py); OpenPI's own RepackTransform then maps them to the keys the eval
client sends (image -> observation/image, ...):
  image       = obs/agentview_rgb[t][::-1, ::-1]     (180 deg rotation, as OpenPI main.py does to env obs)
  wrist_image = obs/eye_in_hand_rgb[t][::-1, ::-1]
  state       = concat(obs/ee_states[t] (eef pos 3 + axis-angle 3), obs/gripper_states[t] (2))  -> 8
  actions     = actions[t : t + H]  (raw 7-D LIBERO delta actions; chunk clamped at episode end)
  prompt      = task instruction
No other transformation is applied here; normalization and resizing are done by OpenPI's own transforms.

Known LIBERO data quirk (verified by tests/integration/test_simulator.py; see docs/EXPERIMENT_PROTOCOL.md): LIBERO's
scripts/create_dataset.py stores the observation returned by env.step(actions[t]) as obs[t], i.e. HDF5 obs[t] is the
state AFTER actions[t] (= sim(states[t+1])). Frames here keep LIBERO's own (obs[t], actions[t]) pairing unchanged; any
re-alignment is a protocol decision, not a silent loader change.

Known gap (documented in docs/EXPERIMENT_PROTOCOL.md): the official HDF5 images are 128x128 while pi05_libero was
trained on 256x256 re-rendered images (OpenVLA's regenerated LIBERO data). OpenPI resizes both to 224.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from typing import Dict, List, Optional, Sequence

import numpy as np

# key -> (trailing shape or None for "any", dtype kind). Leading dim is always T (episode length).
DEMO_SCHEMA = {
    "actions": ((7,), "f"),
    "dones": ((), "u"),
    "rewards": ((), "u"),
    "states": (None, "f"),
    "robot_states": ((9,), "f"),
    "obs/agentview_rgb": (("H", "W", 3), "u"),
    "obs/eye_in_hand_rgb": (("H", "W", 3), "u"),
    "obs/ee_states": ((6,), "f"),
    "obs/ee_pos": ((3,), "f"),
    "obs/ee_ori": ((3,), "f"),
    "obs/gripper_states": ((2,), "f"),
    "obs/joint_states": ((7,), "f"),
}
FINITE_KEYS = ("actions", "states", "obs/ee_states", "obs/gripper_states", "obs/joint_states")


class DemoError(ValueError):
    pass


def _demo_sort_key(k: str) -> int:
    m = re.fullmatch(r"demo_(\d+)", k)
    if not m:
        raise DemoError(f"unexpected demo key {k!r} (expected demo_<n>)")
    return int(m.group(1))


@dataclasses.dataclass(frozen=True)
class DemoIndex:
    path: str
    instruction: str
    demo_keys: List[str]  # numeric order: demo_0, demo_1, ..., demo_49
    lengths: List[int]
    image_hw: tuple

    @property
    def n_demos(self) -> int:
        return len(self.demo_keys)

    def split(self, indices: Sequence[int]) -> List[str]:
        """Demo keys for positional indices into the numerically ordered demo list (e.g. the frozen train split)."""
        bad = [i for i in indices if not 0 <= i < self.n_demos]
        if bad:
            raise DemoError(f"{self.path}: split indices {bad} out of range for {self.n_demos} demos")
        return [self.demo_keys[i] for i in indices]


def _h5py():
    import h5py  # imported lazily so modules that do not touch data import without h5py

    return h5py


def read_instruction(h5file) -> str:
    try:
        return json.loads(h5file["data"].attrs["problem_info"])["language_instruction"]
    except (KeyError, ValueError, TypeError) as e:
        raise DemoError(f"cannot read language instruction from data.attrs['problem_info']: {e}") from e


def validate_demo_file(
    path: os.PathLike | str,
    expected_instruction: Optional[str] = None,
    expected_demos: Optional[int] = None,
    expected_image_hw: Optional[tuple] = None,
    check_values: bool = True,
) -> DemoIndex:
    """Validate structure, shapes, dtypes, finite values and task->instruction mapping. Returns the demo index."""
    path = str(path)
    if not os.path.isfile(path):
        raise DemoError(f"demo file missing: {path}")
    h5py = _h5py()
    try:
        f = h5py.File(path, "r")
    except OSError as e:
        raise DemoError(f"{path}: not a readable HDF5 file (corrupt or incomplete download?): {e}") from e
    with f:
        if "data" not in f:
            raise DemoError(f"{path}: no 'data' group")
        instruction = read_instruction(f)
        if expected_instruction is not None and instruction != expected_instruction:
            raise DemoError(f"{path}: instruction {instruction!r} != expected {expected_instruction!r}")
        keys = sorted(f["data"].keys(), key=_demo_sort_key)
        if not keys:
            raise DemoError(f"{path}: contains no demos")
        if expected_demos is not None and len(keys) != expected_demos:
            raise DemoError(f"{path}: {len(keys)} demos, expected {expected_demos}")
        lengths, hw = [], None
        for k in keys:
            g = f["data"][k]
            T = None
            for name, (trail, kind) in DEMO_SCHEMA.items():
                if name not in g:
                    raise DemoError(f"{path}:{k}: missing dataset {name}")
                ds = g[name]
                if T is None:
                    T = ds.shape[0]
                if ds.shape[0] != T:
                    raise DemoError(f"{path}:{k}/{name}: length {ds.shape[0]} != episode length {T}")
                if ds.dtype.kind != kind:
                    raise DemoError(
                        f"{path}:{k}/{name}: dtype {ds.dtype} (kind {ds.dtype.kind!r}), expected kind {kind!r}"
                    )
                if trail is None:
                    continue
                shape = ds.shape[1:]
                if name.endswith("_rgb"):
                    if len(shape) != 3 or shape[2] != 3:
                        raise DemoError(f"{path}:{k}/{name}: image shape {shape}, expected (H, W, 3)")
                    hw = hw or shape[:2]
                    if shape[:2] != hw:
                        raise DemoError(f"{path}:{k}/{name}: image size {shape[:2]} != {hw} of other images")
                elif shape != tuple(trail):
                    raise DemoError(f"{path}:{k}/{name}: shape {shape}, expected {tuple(trail)}")
            if T is None or T < 1:
                raise DemoError(f"{path}:{k}: empty episode")
            if check_values:
                for name in FINITE_KEYS:
                    if not np.all(np.isfinite(g[name][()])):
                        raise DemoError(f"{path}:{k}/{name}: contains NaN/inf")
                a = g["actions"][()]
                if np.abs(a).max() > 1.0 + 1e-6:
                    raise DemoError(f"{path}:{k}/actions: |action| > 1 (LIBERO OSC actions are clipped to [-1, 1])")
            lengths.append(int(T))
        if expected_image_hw is not None and tuple(hw) != tuple(expected_image_hw):
            raise DemoError(f"{path}: images are {tuple(hw)}, expected {tuple(expected_image_hw)}")
    return DemoIndex(path=path, instruction=instruction, demo_keys=keys, lengths=lengths, image_hw=tuple(hw))


def load_episode(path: os.PathLike | str, demo_key: str) -> Dict[str, np.ndarray]:
    """Load one demo into memory as {schema key: array}."""
    h5py = _h5py()
    with h5py.File(str(path), "r") as f:
        if demo_key not in f["data"]:
            raise DemoError(f"{path}: no demo {demo_key!r}")
        g = f["data"][demo_key]
        return {name: g[name][()] for name in DEMO_SCHEMA}


def episode_state(ep: Dict[str, np.ndarray]) -> np.ndarray:
    """8-D proprio state in pi05_libero's convention: eef pos (3), eef axis-angle (3), gripper qpos (2)."""
    return np.concatenate([ep["obs/ee_states"], ep["obs/gripper_states"]], axis=-1).astype(np.float32)


def action_chunk(actions: np.ndarray, t: int, horizon: int) -> np.ndarray:
    """actions[t : t+horizon], repeating the last action past the episode end (LeRobot delta_timestamps clamping)."""
    idx = np.minimum(np.arange(t, t + horizon), len(actions) - 1)
    return actions[idx].astype(np.float32)


def rotate180(img: np.ndarray) -> np.ndarray:
    """The 180-degree rotation OpenPI's LIBERO client applies to raw (OpenGL-convention) LIBERO images."""
    return np.ascontiguousarray(img[::-1, ::-1])


def episode_to_openpi_frames(
    ep: Dict[str, np.ndarray], instruction: str, horizon: int, timesteps: Optional[Sequence[int]] = None
) -> List[Dict]:
    """Frames keyed like physical-intelligence/libero samples, ready for OpenPI's full training transform chain."""
    T = len(ep["actions"])
    state = episode_state(ep)
    out = []
    for t in range(T) if timesteps is None else timesteps:
        if not 0 <= t < T:
            raise DemoError(f"timestep {t} out of range for episode of length {T}")
        out.append(
            {
                "image": rotate180(ep["obs/agentview_rgb"][t]),
                "wrist_image": rotate180(ep["obs/eye_in_hand_rgb"][t]),
                "state": state[t],
                "actions": action_chunk(ep["actions"], t, horizon),
                "prompt": instruction,
            }
        )
    return out
