"""Shared fixtures. Unit tests use only synthetic data; integration tests are marked and skip with a reason when the
environment they need is absent (see docs/SETUP.md "Tests")."""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
FAKE_ENV = {"OPENPI_DATA_HOME": "/fake/openpi", "LIBERO_DATA_ROOT": "/fake/libero", "CLVLA_OUTPUT_ROOT": "/fake/out"}


@pytest.fixture
def repo_root() -> Path:
    return ROOT


@pytest.fixture
def experiment_raw() -> dict:
    return yaml.safe_load((ROOT / "configs/experiment.yaml").read_text())


@pytest.fixture
def manifest_raw() -> dict:
    return yaml.safe_load((ROOT / "configs/task_manifest.yaml").read_text())


@pytest.fixture
def fake_env() -> dict:
    return dict(FAKE_ENV)


def write_demo_hdf5(
    path: Path, instruction: str = "open the microwave", n_demos: int = 3, T: int = 12, hw: int = 8, mutate=None
) -> Path:
    """Tiny HDF5 with the official LIBERO demo schema. `mutate(name, array) -> array` corrupts selected fields."""
    import h5py

    rng = np.random.default_rng(0)
    mutate = mutate or (lambda name, a: a)
    with h5py.File(path, "w") as f:
        d = f.create_group("data")
        d.attrs["problem_info"] = json.dumps({"language_instruction": instruction})
        for i in range(n_demos):
            g = d.create_group(f"demo_{i}")
            fields = {
                "actions": rng.uniform(-1, 1, (T, 7)),
                "dones": np.r_[np.zeros(T - 1), 1].astype(np.uint8),
                "rewards": np.r_[np.zeros(T - 1), 1].astype(np.uint8),
                "states": rng.normal(size=(T, 77)),
                "robot_states": rng.normal(size=(T, 9)),
                "obs/agentview_rgb": rng.integers(0, 255, (T, hw, hw, 3), dtype=np.uint8),
                "obs/eye_in_hand_rgb": rng.integers(0, 255, (T, hw, hw, 3), dtype=np.uint8),
                "obs/ee_pos": rng.normal(size=(T, 3)),
                "obs/ee_ori": rng.normal(size=(T, 3)),
                "obs/gripper_states": rng.normal(size=(T, 2)) * 0.02,
                "obs/joint_states": rng.normal(size=(T, 7)),
            }
            fields["obs/ee_states"] = np.concatenate([fields["obs/ee_pos"], fields["obs/ee_ori"]], axis=1)
            for name, arr in fields.items():
                arr = mutate(f"demo_{i}/{name}", arr)
                if arr is not None:
                    g.create_dataset(name, data=arr)
    return path


@pytest.fixture
def demo_file(tmp_path) -> Path:
    return write_demo_hdf5(tmp_path / "KITCHEN_SCENE7_open_the_microwave_demo.hdf5")


NORM_STATS = {
    "norm_stats": {
        "state": {"mean": [0.0] * 8, "q01": [0.0] * 8, "std": [1.0] * 8, "q99": [1.0] * 8},
        "actions": {"mean": [0.0] * 7, "q01": [0.0] * 7, "std": [1.0] * 7, "q99": [1.0] * 7},
    }
}


@pytest.fixture
def fake_checkpoint(tmp_path):
    """Synthetic checkpoint tree shaped like pi05_libero, plus its lock."""
    from clvla.assets import build_lock

    root = tmp_path / "pi05_libero"
    (root / "params/d").mkdir(parents=True)
    (root / "params/_METADATA").write_text("{}")
    (root / "params/manifest.ocdbt").write_bytes(b"\x00" * 64)
    (root / "params/d/shard0").write_bytes(os.urandom(4096))
    ns = root / "assets/physical-intelligence/libero"
    ns.mkdir(parents=True)
    (ns / "norm_stats.json").write_text(json.dumps(copy.deepcopy(NORM_STATS)))
    return root, build_lock(root)
