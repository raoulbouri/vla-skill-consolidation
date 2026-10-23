import numpy as np
import pytest

from clvla.data import (
    DemoError,
    action_chunk,
    episode_state,
    episode_to_openpi_frames,
    load_episode,
    rotate180,
    validate_demo_file,
)
from conftest import write_demo_hdf5


def test_valid_demo_file_indexed_in_numeric_order(tmp_path):
    p = write_demo_hdf5(tmp_path / "x.hdf5", n_demos=12, T=5)
    idx = validate_demo_file(p, expected_instruction="open the microwave", expected_demos=12, expected_image_hw=(8, 8))
    assert idx.demo_keys[:3] == ["demo_0", "demo_1", "demo_2"] and idx.demo_keys[-1] == "demo_11"  # not lexicographic
    assert idx.lengths == [5] * 12 and idx.image_hw == (8, 8)
    assert idx.split([0, 10]) == ["demo_0", "demo_10"]
    with pytest.raises(DemoError, match="out of range"):
        idx.split([12])


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda n, a: a[:, :6] if n.endswith("/actions") else a, r"actions: shape \(6,\), expected \(7,\)"),
        (lambda n, a: None if n.endswith("obs/gripper_states") else a, "missing dataset obs/gripper_states"),
        (lambda n, a: a[:-1] if n == "demo_1/obs/ee_states" else a, "length 11 != episode length 12"),
        (
            lambda n, a: np.where(np.arange(a.size).reshape(a.shape) == 3, np.nan, a) if n == "demo_2/actions" else a,
            "NaN/inf",
        ),
        (lambda n, a: a * 3 if n == "demo_0/actions" else a, r"\|action\| > 1"),
        (lambda n, a: a.astype(np.float32) if n.endswith("agentview_rgb") else a, "dtype"),
        (lambda n, a: a[..., :1] if n.endswith("eye_in_hand_rgb") else a, "image shape"),
    ],
)
def test_schema_violations_detected(tmp_path, mutate, msg):
    p = write_demo_hdf5(tmp_path / "bad.hdf5", mutate=mutate)
    with pytest.raises(DemoError, match=msg):
        validate_demo_file(p)


def test_task_to_instruction_mapping_enforced(demo_file):
    with pytest.raises(DemoError, match="instruction 'open the microwave' != expected 'close the microwave'"):
        validate_demo_file(demo_file, expected_instruction="close the microwave")


def test_wrong_counts_and_sizes_detected(demo_file):
    with pytest.raises(DemoError, match="3 demos, expected 50"):
        validate_demo_file(demo_file, expected_demos=50)
    with pytest.raises(DemoError, match=r"images are \(8, 8\), expected \(128, 128\)"):
        validate_demo_file(demo_file, expected_image_hw=(128, 128))


def test_missing_and_corrupt_files(tmp_path):
    with pytest.raises(DemoError, match="demo file missing"):
        validate_demo_file(tmp_path / "none.hdf5")
    bad = tmp_path / "corrupt.hdf5"
    bad.write_bytes(b"\x89HDF\r\n" + b"\x00" * 100)
    with pytest.raises(DemoError, match="not a readable HDF5"):
        validate_demo_file(bad)
    full = write_demo_hdf5(tmp_path / "full.hdf5")
    trunc = tmp_path / "trunc.hdf5"
    trunc.write_bytes(full.read_bytes()[: full.stat().st_size // 3])
    with pytest.raises(DemoError):
        validate_demo_file(trunc)


def test_episode_to_openpi_frames_follows_openpi_conventions(demo_file):
    ep = load_episode(demo_file, "demo_1")
    T = len(ep["actions"])
    frames = episode_to_openpi_frames(ep, "open the microwave", horizon=10)
    assert len(frames) == T
    f0 = frames[0]
    assert set(f0) == {"image", "wrist_image", "state", "actions", "prompt"}  # physical-intelligence/libero keys
    assert np.array_equal(f0["image"], ep["obs/agentview_rgb"][0][::-1, ::-1])  # 180-degree rotation, nothing else
    assert np.array_equal(f0["wrist_image"], ep["obs/eye_in_hand_rgb"][0][::-1, ::-1])
    assert f0["image"].dtype == np.uint8 and f0["image"].flags["C_CONTIGUOUS"]
    assert f0["state"].shape == (8,) and np.allclose(f0["state"][:6], ep["obs/ee_states"][0])
    assert np.allclose(f0["state"][6:], ep["obs/gripper_states"][0])
    assert f0["actions"].shape == (10, 7) and np.allclose(
        f0["actions"], ep["actions"][:10]
    )  # raw actions, no transform
    last = frames[-1]["actions"]
    assert np.allclose(last, np.repeat(ep["actions"][-1:], 10, axis=0))  # clamped at episode end (LeRobot behaviour)
    with pytest.raises(DemoError, match="out of range"):
        episode_to_openpi_frames(ep, "x", 10, timesteps=[T])


def test_helpers():
    a = np.arange(5 * 7, dtype=float).reshape(5, 7)
    assert np.array_equal(action_chunk(a, 3, 4), a[[3, 4, 4, 4]])
    img = np.arange(2 * 3 * 3).reshape(2, 3, 3)
    assert np.array_equal(rotate180(rotate180(img)), img)
    assert episode_state({"obs/ee_states": np.ones((2, 6)), "obs/gripper_states": np.zeros((2, 2))}).shape == (2, 8)
