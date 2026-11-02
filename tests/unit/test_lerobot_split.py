"""LeRobot conversion helpers: repo naming, split isolation, frame format, fingerprints (no lerobot import)."""

import numpy as np
import pytest

from clvla.lerobot_convert import (
    FEATURES,
    dataset_fingerprint,
    frames_from_processed_demo,
    lerobot_root,
    repo_ids,
    split_demo_keys,
)


def test_train_and_val_are_distinct_repos():
    r = repo_ids("clvla", "libero90_aligned256_v1", 78)
    assert r == {"train": "clvla/libero90_aligned256_v1-t078-train", "val": "clvla/libero90_aligned256_v1-t078-val"}
    assert r["train"] != r["val"]


def test_split_uses_numeric_demo_order_and_protocol_indices():
    keys = [f"demo_{i}" for i in (10, 2, 0, 1, 11, 3)]  # deliberately unordered, lexicographic trap demo_10 < demo_2
    s = split_demo_keys(keys, [0, 1, 2, 3], [4, 5])
    assert s["train"] == ["demo_0", "demo_1", "demo_2", "demo_3"] and s["val"] == ["demo_10", "demo_11"]
    with pytest.raises(ValueError, match="leakage"):
        split_demo_keys(keys, [0, 1, 2, 3, 4], [4, 5])
    with pytest.raises(ValueError, match="out of range"):
        split_demo_keys(keys, [0], [6])


def test_protocol_split_has_45_train_5_val():
    keys = [f"demo_{i}" for i in range(50)]
    s = split_demo_keys(keys, list(range(45)), list(range(45, 50)))
    assert len(s["train"]) == 45 and s["val"] == [f"demo_{i}" for i in range(45, 50)]
    assert not set(s["train"]) & set(s["val"])


class _Group(dict):
    def __getitem__(self, k):
        return _DS(dict.__getitem__(self, k))


class _DS:
    def __init__(self, a):
        self.a = a

    def __getitem__(self, _):
        return self.a


def test_frames_follow_openpi_libero_format():
    T = 4
    av = np.arange(T * 256 * 256 * 3, dtype=np.uint32).reshape(T, 256, 256, 3).astype(np.uint8)
    g = _Group({"obs/agentview_rgb": av, "obs/eye_in_hand_rgb": av[:, ::-1].copy(),
                "obs/state": np.ones((T, 8)), "actions": np.full((T, 7), 0.5)})
    frames = frames_from_processed_demo(g, "open the microwave")
    assert len(frames) == T and set(frames[0]) == set(FEATURES) | {"task"}
    assert np.array_equal(frames[1]["image"], av[1][::-1, ::-1])  # rotated 180 like OpenPI's training images
    assert frames[0]["state"].dtype == np.float32 and frames[0]["actions"].dtype == np.float32
    assert np.allclose(frames[2]["actions"], 0.5) and frames[3]["task"] == "open the microwave"
    bad = _Group(dict(g, actions=np.full((T, 7), np.nan)))
    with pytest.raises(ValueError, match="non-finite"):
        frames_from_processed_demo(bad, "x")


def test_lerobot_root_requires_env(monkeypatch, tmp_path):
    monkeypatch.delenv("HF_LEROBOT_HOME", raising=False)
    with pytest.raises(RuntimeError, match="HF_LEROBOT_HOME"):
        lerobot_root("clvla/x")
    assert lerobot_root("clvla/x", str(tmp_path)) == tmp_path / "clvla/x"


def test_fingerprint_detects_any_change(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data/a.parquet").write_bytes(b"abc")
    (tmp_path / "meta.json").write_text("{}")
    fp = dataset_fingerprint(tmp_path)
    assert fp["num_files"] == 2 and dataset_fingerprint(tmp_path) == fp
    (tmp_path / "data/a.parquet").write_bytes(b"abd")
    assert dataset_fingerprint(tmp_path)["sha256"] != fp["sha256"]
    with pytest.raises(FileNotFoundError):
        dataset_fingerprint(tmp_path / "missing")
