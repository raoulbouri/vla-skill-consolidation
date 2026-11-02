"""Named demo split profiles (n25) and the bounded concurrent-training lock."""

import types

import pytest

from clvla.lerobot_convert import lerobot_manifest_name, repo_ids, resolve_split
from clvla.specialists import RunnerRefused, exclusive_training_lock, max_concurrent_trainers

DCFG = {"split_profiles": {"n25": {"train_demo_indices": {"start": 0, "stop": 25},
                                   "val_demo_indices": {"start": 25, "stop": 50}}}}
CFG = types.SimpleNamespace(splits=types.SimpleNamespace(train_demo_indices=list(range(45)),
                                                         val_demo_indices=list(range(45, 50))))


def test_no_profile_is_the_protocol_split():
    assert resolve_split(DCFG, CFG, None) == (list(range(45)), list(range(45, 50)))


def test_n25_is_disjoint_and_covers_all_50():
    tr, va = resolve_split(DCFG, CFG, "n25")
    assert tr == list(range(25)) and va == list(range(25, 50)) and not set(tr) & set(va)


def test_unknown_or_overlapping_profile_is_refused():
    with pytest.raises(ValueError, match="unknown split profile"):
        resolve_split(DCFG, CFG, "n10")
    bad = {"split_profiles": {"x": {"train_demo_indices": {"start": 0, "stop": 30},
                                    "val_demo_indices": {"start": 25, "stop": 50}}}}
    with pytest.raises(ValueError, match="overlapping"):
        resolve_split(bad, CFG, "x")


def test_profile_names_never_collide_with_protocol_datasets():
    assert repo_ids("clvla", "v1", 78) == {"train": "clvla/v1-t078-train", "val": "clvla/v1-t078-val"}
    assert repo_ids("clvla", "v1", 78, "n25") == {"train": "clvla/v1-t078-n25-train", "val": "clvla/v1-t078-n25-val"}
    assert lerobot_manifest_name("T") == "T.lerobot.json" and lerobot_manifest_name("T", "n25") == "T.n25.lerobot.json"


def test_slot_lock_allows_n_concurrent_and_refuses_more(tmp_path):
    with exclusive_training_lock(tmp_path, max_concurrent=2) as a:
        with exclusive_training_lock(tmp_path, max_concurrent=2) as b:
            assert a != b
            with pytest.raises(RunnerRefused, match="all 2 training slot"):
                with exclusive_training_lock(tmp_path, max_concurrent=2):
                    pass
    assert not list(tmp_path.glob(".training*.lock"))  # released


def test_default_is_serial(monkeypatch):
    monkeypatch.delenv("CLVLA_MAX_CONCURRENT_TRAINING", raising=False)
    assert max_concurrent_trainers() == 1
    monkeypatch.setenv("CLVLA_MAX_CONCURRENT_TRAINING", "4")
    with pytest.raises(RunnerRefused):
        max_concurrent_trainers()


def test_n10_is_a_subset_of_n25_and_holds_out_the_other_40():
    import yaml
    from pathlib import Path

    dcfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "configs/datasets/processed_v1.yaml").read_text())
    tr10, va10 = resolve_split(dcfg, CFG, "n10")
    tr25, va25 = resolve_split(dcfg, CFG, "n25")
    assert tr10 == list(range(10)) and set(tr10) < set(tr25)
    assert va10 == list(range(10, 50)) and va25 == list(range(25, 50)) and not set(tr10) & set(va10)
