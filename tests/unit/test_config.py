import copy

import numpy as np
import pytest

from clvla.config import (
    ConfigError,
    compute_protocol_sha256,
    derive_seed,
    expand_env,
    load_experiment_config,
    parse_experiment_config,
    seed_everything,
)


def refreeze(raw):
    raw["protocol_sha256"] = compute_protocol_sha256(raw)
    return raw


def test_committed_config_is_valid_and_frozen(experiment_raw, fake_env):
    cfg = parse_experiment_config(experiment_raw, env=fake_env)
    assert cfg.checkpoint.openpi_config == "pi05_libero"
    assert cfg.checkpoint.uri == "gs://openpi-assets/checkpoints/pi05_libero"
    assert cfg.checkpoint.local_dir == "/fake/openpi/openpi-assets/checkpoints/pi05_libero"
    assert cfg.checkpoint.norm_stats_asset_id == "physical-intelligence/libero"
    assert (cfg.policy.action_horizon, cfg.policy.replan_steps, cfg.policy.resize_size) == (10, 5, 224)
    assert (cfg.simulator.camera_resolution, cfg.simulator.max_steps, cfg.simulator.num_steps_wait) == (256, 400, 10)
    assert cfg.simulator.dummy_action == [0, 0, 0, 0, 0, 0, -1]
    assert cfg.splits.train_demo_indices == list(range(45)) and cfg.splits.val_demo_indices == list(range(45, 50))
    assert cfg.splits.selection_init_states == list(range(10))
    assert cfg.splits.test_init_states == list(range(10, 50))


@pytest.mark.parametrize(
    "path",
    [
        ["experiment_name"],
        ["protocol"],
        ["protocol", "seeds"],
        ["protocol", "checkpoint", "uri"],
        ["protocol", "policy", "replan_steps"],
        ["protocol", "simulator", "max_steps"],
        ["protocol", "splits", "val_demo_indices"],
        ["evaluation", "episodes_per_task"],
        ["output_root"],
    ],
)
def test_missing_key_is_reported_with_its_path(experiment_raw, fake_env, path):
    raw = copy.deepcopy(experiment_raw)
    node = raw
    for k in path[:-1]:
        node = node[k]
    del node[path[-1]]
    if path[0] == "protocol":
        refreeze(raw) if len(path) > 1 else None
    with pytest.raises(ConfigError, match=path[-1]):
        parse_experiment_config(raw, env=fake_env)


@pytest.mark.parametrize(
    "section,key,value,msg",
    [
        ("policy", "replan_steps", 11, "replan_steps"),
        ("policy", "resize_size", -1, "outside allowed range"),
        ("simulator", "max_steps", "400", "expected int"),
        ("simulator", "dummy_action", [0, 0, -1], "dummy_action"),
        ("checkpoint", "openpi_config", "pi05_droid", "official pi05_libero"),
        ("checkpoint", "uri", "gs://openpi-assets/checkpoints/pi0_libero", "official pi05_libero"),
        ("data", "suite", "libero_10", "not one of"),
        ("splits", "val_demo_indices", {"start": 40, "stop": 50}, "overlap"),
        ("splits", "test_init_states", {"start": 5, "stop": 50}, "overlap"),
        ("seeds", "eval", [7, 7], "duplicates"),
    ],
)
def test_invalid_protocol_values_rejected(experiment_raw, fake_env, section, key, value, msg):
    raw = copy.deepcopy(experiment_raw)
    raw["protocol"][section][key] = value
    with pytest.raises(ConfigError, match=msg):
        parse_experiment_config(refreeze(raw), env=fake_env)


def test_invalid_evaluation_values_rejected(experiment_raw, fake_env):
    raw = copy.deepcopy(experiment_raw)
    raw["evaluation"]["episodes_per_task"] = 41  # test split has 40 init states
    with pytest.raises(ConfigError, match="exceeds"):
        parse_experiment_config(raw, env=fake_env)
    raw["evaluation"]["episodes_per_task"] = 5
    raw["evaluation"]["init_state_split"] = "train"
    with pytest.raises(ConfigError, match="not one of"):
        parse_experiment_config(raw, env=fake_env)


def test_silent_change_of_frozen_setting_is_refused(experiment_raw, fake_env):
    raw = copy.deepcopy(experiment_raw)
    raw["protocol"]["simulator"]["max_steps"] = 520  # e.g. someone copies the libero_10 cap
    with pytest.raises(ConfigError, match="frozen protocol section was modified"):
        parse_experiment_config(raw, env=fake_env)


def test_overrides_limited_to_evaluation_section(repo_root, fake_env):
    path = repo_root / "configs/experiment.yaml"
    cfg = load_experiment_config(path, env=fake_env, overrides={"evaluation.episodes_per_task": 2})
    assert cfg.evaluation.episodes_per_task == 2
    with pytest.raises(ConfigError, match="only evaluation"):
        load_experiment_config(path, env=fake_env, overrides={"protocol.simulator.max_steps": 10})


def test_env_expansion_requires_defined_variables():
    assert expand_env("${A}/x", {"A": "/r"}) == "/r/x"
    with pytest.raises(ConfigError, match="B is not set"):
        expand_env("${B}/x", {})
    with pytest.raises(ConfigError, match="B is not set"):
        expand_env("${B}/x", {"B": ""})


def test_seed_derivation_is_deterministic_and_distinct():
    assert derive_seed(7, "task", 3, 0) == derive_seed(7, "task", 3, 0)
    seeds = {derive_seed(7, "task", t, e) for t in range(10) for e in range(50)}
    assert len(seeds) == 500
    assert derive_seed(7, 1) != derive_seed(8, 1)
    assert all(0 <= s < 2**31 for s in seeds)


def test_seed_everything_reproducible():
    seed_everything(123)
    a = np.random.rand(3)
    seed_everything(123)
    assert np.array_equal(a, np.random.rand(3))
