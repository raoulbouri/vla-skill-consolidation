import json

import pytest

from clvla.assets import (
    AssetError,
    build_lock,
    check_norm_stats,
    checkpoint_fingerprint,
    find_partial_files,
    load_lock,
    sha256_file,
    verify_checkpoint,
    verify_tree,
)

ASSET = "physical-intelligence/libero"


def test_intact_checkpoint_passes_shallow_and_deep(fake_checkpoint):
    root, lock = fake_checkpoint
    for deep in (False, True):
        rep = verify_checkpoint(root, lock, ASSET, 8, 7, deep=deep)
        assert rep.ok, rep.errors
        assert rep.checked == len(lock)


def test_missing_file_detected(fake_checkpoint):
    root, lock = fake_checkpoint
    (root / "params/_METADATA").unlink()
    rep = verify_checkpoint(root, lock, ASSET, 8, 7)
    assert any("params/_METADATA: missing" in e for e in rep.errors)
    with pytest.raises(AssetError, match="_METADATA"):
        rep.raise_if_failed()


def test_truncated_download_detected_without_hashing(fake_checkpoint):
    root, lock = fake_checkpoint
    shard = root / "params/d/shard0"
    shard.write_bytes(shard.read_bytes()[:100])
    rep = verify_checkpoint(root, lock, ASSET, 8, 7, deep=False)
    assert any("incomplete download (truncated)" in e for e in rep.errors)


def test_same_size_corruption_needs_and_gets_deep_check(fake_checkpoint):
    root, lock = fake_checkpoint
    shard = root / "params/d/shard0"
    data = bytearray(shard.read_bytes())
    data[10] ^= 0xFF
    shard.write_bytes(bytes(data))
    assert verify_checkpoint(root, lock, ASSET, 8, 7, deep=False).ok
    rep = verify_checkpoint(root, lock, ASSET, 8, 7, deep=True)
    assert any("sha256 mismatch (corrupt)" in e for e in rep.errors)


def test_leftover_partial_files_detected(fake_checkpoint):
    root, lock = fake_checkpoint
    (root / "params/d/shard1.partial").write_bytes(b"x")
    (root / "params/d/tmpdownload.tmp").write_bytes(b"x")
    assert find_partial_files(root) == ["params/d/shard1.partial", "params/d/tmpdownload.tmp"]
    rep = verify_checkpoint(root, lock, ASSET, 8, 7)
    assert sum("leftover partial" in e for e in rep.errors) == 2


def test_missing_or_wrong_norm_stats_detected(fake_checkpoint):
    root, lock = fake_checkpoint
    ns = root / "assets" / ASSET / "norm_stats.json"
    stats = json.loads(ns.read_text())
    stats["norm_stats"]["actions"]["q99"] = [1.0] * 6
    ns.write_text(json.dumps(stats))
    rep = verify_checkpoint(root, build_lock(root), ASSET, 8, 7)
    assert any("actions.q99' has dim 6, expected 7" in e for e in rep.errors)
    ns.unlink()
    rep = verify_checkpoint(root, lock, ASSET, 8, 7)
    assert any("bundled norm stats missing" in e for e in rep.errors)


def test_norm_stats_checks():
    good = {
        "state": {f: [0.0] * 8 for f in ("mean", "std", "q01", "q99")},
        "actions": {f: [0.0] * 7 for f in ("mean", "std", "q01", "q99")},
    }
    assert check_norm_stats(good, 8, 7) == []
    bad = json.loads(json.dumps(good))
    bad["actions"]["mean"][0] = float("nan")
    bad["state"]["q01"] = [2.0] * 8
    errs = check_norm_stats(bad, 8, 7)
    assert any("non-finite" in e for e in errs) and any("q01 > q99" in e for e in errs)
    assert any("no 'state'" in e for e in check_norm_stats({"actions": good["actions"]}, 8, 7))


def test_dataset_tree_against_lock(tmp_path):
    (tmp_path / "libero_90").mkdir()
    a = tmp_path / "libero_90/A_demo.hdf5"
    a.write_bytes(b"a" * 1000)
    lock = {
        "libero_90/A_demo.hdf5": {"size": 1000, "sha256": sha256_file(a)},
        "libero_90/B_demo.hdf5": {"size": 5, "sha256": "0" * 64},
    }
    rep = verify_tree(tmp_path, lock, "ds", deep=True)
    assert rep.errors == ["libero_90/B_demo.hdf5: missing"]
    assert verify_tree(tmp_path, lock, "ds", only=["libero_90/A_demo.hdf5"]).ok
    assert "not in lock" in verify_tree(tmp_path, lock, "ds", only=["libero_90/C"]).errors[0]
    assert "directory missing" in verify_tree(tmp_path / "nope", lock, "ds").errors[0]


def test_fingerprint_changes_with_any_file(fake_checkpoint):
    root, lock = fake_checkpoint
    fp = checkpoint_fingerprint(lock)
    (root / "params/_METADATA").write_text('{"x": 1}')
    assert checkpoint_fingerprint(build_lock(root)) != fp
    assert checkpoint_fingerprint(lock) == fp


def test_lock_file_loading(tmp_path):
    with pytest.raises(AssetError, match="lock file missing"):
        load_lock(tmp_path / "none.json")


def test_committed_locks_are_well_formed(repo_root):
    ck = json.loads((repo_root / "configs/checkpoint.lock.json").read_text())
    assert ck["uri"] == "gs://openpi-assets/checkpoints/pi05_libero" and ck["openpi_config"] == "pi05_libero"
    assert checkpoint_fingerprint(ck["files"]) == ck["fingerprint_sha256"]
    assert f"assets/{ASSET}/norm_stats.json" in ck["files"]
    ds = json.loads((repo_root / "configs/dataset.lock.json").read_text())
    assert len(ds["files"]) == 90 and all(len(v["sha256"]) == 64 for v in ds["files"].values())
