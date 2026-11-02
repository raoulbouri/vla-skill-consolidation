"""Pre-run verification: exact checkpoint identity, norm stats, task match, dataset split isolation (no GPU)."""

import json
import types
from pathlib import Path

import pytest

from clvla.assets import build_lock, checkpoint_fingerprint
from clvla.preflight import (
    REQUIRED_CHECKPOINT_FINGERPRINT,
    PreflightError,
    verify_base_checkpoint,
    verify_datasets,
    verify_output_path,
    verify_task,
)
from clvla.tasks import parse_manifest


def _cfg(ckpt_dir, lock_rel):
    return types.SimpleNamespace(
        checkpoint=types.SimpleNamespace(lock_file=lock_rel, local_dir=str(ckpt_dir),
                                         norm_stats_asset_id="physical-intelligence/libero"),
        policy=types.SimpleNamespace(state_dim=8, action_dim=7))


def test_only_the_official_checkpoint_fingerprint_is_accepted(fake_checkpoint, tmp_path):
    root, files = fake_checkpoint
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/lock.json").write_text(json.dumps({"files": files, "fingerprint_sha256": checkpoint_fingerprint(files)}))
    assert checkpoint_fingerprint(files) != REQUIRED_CHECKPOINT_FINGERPRINT
    with pytest.raises(PreflightError, match="never train from a different checkpoint"):
        verify_base_checkpoint(_cfg(root, "configs/lock.json"), tmp_path)


def test_committed_lock_matches_required_fingerprint(repo_root):
    lock = json.loads((repo_root / "configs/checkpoint.lock.json").read_text())
    assert checkpoint_fingerprint(lock["files"]) == REQUIRED_CHECKPOINT_FINGERPRINT == lock["fingerprint_sha256"]


def test_norm_stats_tampering_detected(fake_checkpoint, tmp_path, monkeypatch):
    import clvla.preflight as pf

    root, files = fake_checkpoint
    fp = checkpoint_fingerprint(files)
    monkeypatch.setattr(pf, "REQUIRED_CHECKPOINT_FINGERPRINT", fp)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs/lock.json").write_text(json.dumps({"files": files, "fingerprint_sha256": fp}))
    ok = verify_base_checkpoint(_cfg(root, "configs/lock.json"), tmp_path)
    assert ok["fingerprint_sha256"] == fp
    ns = root / "assets/physical-intelligence/libero/norm_stats.json"
    text = ns.read_text()
    ns.write_text(text.replace("0.0", "0.5", 1))  # same size, still valid stats: only the sha256 check can tell
    assert len(ns.read_text()) == len(text)
    with pytest.raises(PreflightError, match="norm stats sha256"):
        verify_base_checkpoint(_cfg(root, "configs/lock.json"), tmp_path)


def test_task_must_match_manifest_and_snapshot(manifest_raw, repo_root):
    m = parse_manifest(manifest_raw)
    t = verify_task(m, 78, repo_root / "configs/libero90_tasks.json")
    assert t["position"] == 0 and t["instruction"].startswith("pick up the book") and t["manifest_sha256"] == m.content_sha256
    with pytest.raises(KeyError):
        verify_task(m, 1, repo_root / "configs/libero90_tasks.json")


def _lerobot_manifest(tmp_path, home, **over):
    keys = [f"demo_{i}" for i in range(50)]
    m = {"libero_90_index": 78, "instruction": "pick up", "version": "v1",
         "source": {"processed_sha256": "p" * 64, "raw_sha256": "r" * 64},
         "splits": {"train": {"repo_id": "clvla/v1-t078-train", "root": str(Path(home) / "clvla/v1-t078-train"),
                              "demo_keys": keys[:45], "protocol_indices": list(range(45)), "num_episodes": 45,
                              "num_frames": 6800, "fingerprint": {"sha256": "T" * 64}},
                    "val": {"repo_id": "clvla/v1-t078-val", "root": str(Path(home) / "clvla/v1-t078-val"),
                            "demo_keys": keys[45:], "protocol_indices": list(range(45, 50)), "num_episodes": 5,
                            "num_frames": 750, "fingerprint": {"sha256": "V" * 64}}}}
    for path, value in over.items():
        part, key = path.split(".")
        m["splits"][part][key] = value
    p = tmp_path / "t.lerobot.json"
    p.write_text(json.dumps(m))
    return p


def _fp(root):
    return {"sha256": "T" * 64 if root.endswith("train") else "V" * 64}


TASK = {"libero_90_index": 78, "instruction": "pick up"}


def test_datasets_verified_and_isolated(tmp_path, monkeypatch):
    home = tmp_path / "lerobot"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    out = verify_datasets(_lerobot_manifest(tmp_path, home), TASK, range(45), range(45, 50), fingerprint_fn=_fp)
    assert out["train"]["repo_id"].endswith("-train") and len(out["val"]["demo_keys"]) == 5


@pytest.mark.parametrize("over,msg", [
    ({"val.repo_id": "clvla/v1-t078-train"}, "distinct"),
    ({"val.demo_keys": ["demo_44", "demo_46"]}, "leakage"),
    ({"train.protocol_indices": list(range(46))}, "protocol"),
])
def test_dataset_problems_refused(tmp_path, monkeypatch, over, msg):
    home = tmp_path / "lerobot"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    with pytest.raises((PreflightError, ValueError), match=msg):
        verify_datasets(_lerobot_manifest(tmp_path, home, **over), TASK, range(45), range(45, 50), fingerprint_fn=_fp)


def test_changed_dataset_or_wrong_home_refused(tmp_path, monkeypatch):
    home = tmp_path / "lerobot"
    monkeypatch.setenv("HF_LEROBOT_HOME", str(home))
    with pytest.raises(PreflightError, match="changed since conversion"):
        verify_datasets(_lerobot_manifest(tmp_path, home), TASK, range(45), range(45, 50),
                        fingerprint_fn=lambda r: {"sha256": "X" * 64})
    monkeypatch.setenv("HF_LEROBOT_HOME", str(tmp_path / "elsewhere"))
    with pytest.raises(PreflightError, match="HF_LEROBOT_HOME"):
        verify_datasets(_lerobot_manifest(tmp_path, home), TASK, range(45), range(45, 50), fingerprint_fn=_fp)
    with pytest.raises(PreflightError, match="task/instruction"):
        verify_datasets(_lerobot_manifest(tmp_path, home), {"libero_90_index": 84, "instruction": "x"}, range(45),
                        range(45, 50), fingerprint_fn=_fp)
    with pytest.raises(PreflightError, match="no LeRobot manifest"):
        verify_datasets(tmp_path / "none.json", TASK, range(45), range(45, 50))


def test_output_path_refusal(tmp_path):
    verify_output_path(tmp_path / "new")
    with pytest.raises(PreflightError):
        verify_output_path(tmp_path)


def test_fake_checkpoint_lock_roundtrip(fake_checkpoint):
    root, files = fake_checkpoint
    assert build_lock(root) == files
