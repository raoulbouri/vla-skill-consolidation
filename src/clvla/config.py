"""Validated loading of configs/experiment.yaml.

The `protocol` section holds every frozen setting (seeds, image sizes, episode limits, init-state splits, checkpoint
identity). Its canonical sha256 is stored next to it as `protocol_sha256`; loading fails if the two disagree, so a
frozen setting can never change silently. To change the protocol on purpose: edit it, bump `protocol_version`, run
`scripts/freeze_config.py`, and record the change in docs/EXPERIMENT_PROTOCOL.md.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import yaml

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ConfigError(ValueError):
    """Raised for any invalid, incomplete or tampered configuration."""


def load_yaml(path: os.PathLike | str) -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open() as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: top level must be a mapping, got {type(data).__name__}")
    return data


def canonical_sha256(obj: Any) -> str:
    """sha256 of a JSON-canonical encoding (sorted keys, no whitespace). Used for config and manifest hashes."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def expand_env(value: str, env: Optional[Mapping[str, str]] = None) -> str:
    """Expand ${VAR} references; an undefined variable is an error, never an empty string."""
    env = os.environ if env is None else env

    def repl(m: re.Match) -> str:
        name = m.group(1)
        if name not in env or env[name] == "":
            raise ConfigError(f"environment variable {name} is not set (needed to expand {value!r}); source env.sh")
        return env[name]

    return _ENV_REF.sub(repl, value)


# ---------------------------------------------------------------------------------------------------------------------
# Small validation helpers: every error message names the full key path.


def _get(d: Mapping[str, Any], key: str, path: str) -> Any:
    if not isinstance(d, Mapping):
        raise ConfigError(f"{path or '<root>'}: expected a mapping, got {type(d).__name__}")
    if key not in d:
        raise ConfigError(f"missing required key: {path + '.' if path else ''}{key}")
    return d[key]


def _int(d, key, path, lo=None, hi=None) -> int:
    v = _get(d, key, path)
    if isinstance(v, bool) or not isinstance(v, int):
        raise ConfigError(f"{path}.{key}: expected int, got {v!r}")
    if (lo is not None and v < lo) or (hi is not None and v > hi):
        raise ConfigError(f"{path}.{key}: {v} outside allowed range [{lo}, {hi}]")
    return v


def _str(d, key, path, choices=None) -> str:
    v = _get(d, key, path)
    if not isinstance(v, str) or not v:
        raise ConfigError(f"{path}.{key}: expected non-empty string, got {v!r}")
    if choices is not None and v not in choices:
        raise ConfigError(f"{path}.{key}: {v!r} not one of {sorted(choices)}")
    return v


def _bool(d, key, path) -> bool:
    v = _get(d, key, path)
    if not isinstance(v, bool):
        raise ConfigError(f"{path}.{key}: expected bool, got {v!r}")
    return v


def _int_list(d, key, path, lo=None, hi=None, unique=True) -> List[int]:
    v = _get(d, key, path)
    if not isinstance(v, list) or not v or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        raise ConfigError(f"{path}.{key}: expected non-empty list of ints, got {v!r}")
    if unique and len(set(v)) != len(v):
        raise ConfigError(f"{path}.{key}: contains duplicates: {v}")
    for x in v:
        if (lo is not None and x < lo) or (hi is not None and x > hi):
            raise ConfigError(f"{path}.{key}: value {x} outside allowed range [{lo}, {hi}]")
    return v


def _range_list(d, key, path, lo, hi) -> List[int]:
    """Accept either an explicit int list or {start: a, stop: b} (half-open)."""
    v = _get(d, key, path)
    if isinstance(v, dict):
        start, stop = _int(v, "start", f"{path}.{key}", lo, hi), _int(v, "stop", f"{path}.{key}", lo, hi + 1)
        if stop <= start:
            raise ConfigError(f"{path}.{key}: empty range [{start}, {stop})")
        return list(range(start, stop))
    return _int_list(d, key, path, lo, hi)


# ---------------------------------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class CheckpointSpec:
    openpi_config: str
    uri: str
    local_dir: str
    norm_stats_asset_id: str
    lock_file: str


@dataclasses.dataclass(frozen=True)
class PolicySpec:
    action_dim: int
    state_dim: int
    action_horizon: int
    replan_steps: int
    resize_size: int


@dataclasses.dataclass(frozen=True)
class SimulatorSpec:
    camera_resolution: int
    num_steps_wait: int
    max_steps: int
    dummy_action: List[float]
    env_seed: int


@dataclasses.dataclass(frozen=True)
class SplitSpec:
    train_demo_indices: List[int]
    val_demo_indices: List[int]
    selection_init_states: List[int]
    test_init_states: List[int]


@dataclasses.dataclass(frozen=True)
class DataSpec:
    root: str
    suite: str
    demos_per_task: int
    image_size: int
    lock_file: str


@dataclasses.dataclass(frozen=True)
class EvalSpec:
    episodes_per_task: int
    init_state_split: str
    save_videos: bool
    host: str
    port: int


@dataclasses.dataclass(frozen=True)
class ExperimentConfig:
    path: str
    experiment_name: str
    protocol_version: str
    protocol_sha256: str
    global_seed: int
    eval_seeds: List[int]
    task_manifest: str
    checkpoint: CheckpointSpec
    policy: PolicySpec
    simulator: SimulatorSpec
    data: DataSpec
    splits: SplitSpec
    evaluation: EvalSpec
    output_root: str
    raw: Dict[str, Any]

    def resolve(self, rel: str) -> Path:
        """Resolve a repo-relative path from the config against the repository root (parent of configs/)."""
        p = Path(rel)
        return p if p.is_absolute() else Path(self.path).resolve().parent.parent / p


def compute_protocol_sha256(raw: Mapping[str, Any]) -> str:
    return canonical_sha256(_get(raw, "protocol", ""))


def parse_experiment_config(
    raw: Dict[str, Any], path: str = "<memory>", env: Optional[Mapping[str, str]] = None, check_frozen: bool = True
) -> ExperimentConfig:
    raw = copy.deepcopy(raw)
    _int(raw, "schema_version", "", 1, 1)
    name = _str(raw, "experiment_name", "")
    proto = _get(raw, "protocol", "")
    pv = _str(proto, "protocol_version", "protocol")
    stored = _str(raw, "protocol_sha256", "")
    actual = compute_protocol_sha256(raw)
    if check_frozen and stored != actual:
        raise ConfigError(
            f"{path}: frozen protocol section was modified (protocol_sha256 {stored[:12]}... != actual {actual[:12]}...). "
            "Frozen settings must not change silently: revert the edit, or bump protocol.protocol_version, run "
            "scripts/freeze_config.py and document the change in docs/EXPERIMENT_PROTOCOL.md."
        )

    seeds = _get(proto, "seeds", "protocol")
    global_seed = _int(seeds, "global", "protocol.seeds", 0, 2**31 - 1)
    eval_seeds = _int_list(seeds, "eval", "protocol.seeds", 0, 2**31 - 1)

    c = _get(proto, "checkpoint", "protocol")
    checkpoint = CheckpointSpec(
        openpi_config=_str(c, "openpi_config", "protocol.checkpoint"),
        uri=_str(c, "uri", "protocol.checkpoint"),
        local_dir=expand_env(_str(c, "local_dir", "protocol.checkpoint"), env),
        norm_stats_asset_id=_str(c, "norm_stats_asset_id", "protocol.checkpoint"),
        lock_file=_str(c, "lock_file", "protocol.checkpoint"),
    )
    if checkpoint.openpi_config != "pi05_libero" or not checkpoint.uri.rstrip("/").endswith("checkpoints/pi05_libero"):
        raise ConfigError(
            "protocol.checkpoint must be the official pi05_libero checkpoint (config pi05_libero, "
            f"gs://openpi-assets/checkpoints/pi05_libero); got config={checkpoint.openpi_config!r} uri={checkpoint.uri!r}"
        )

    p = _get(proto, "policy", "protocol")
    policy = PolicySpec(
        action_dim=_int(p, "action_dim", "protocol.policy", 1, 64),
        state_dim=_int(p, "state_dim", "protocol.policy", 1, 64),
        action_horizon=_int(p, "action_horizon", "protocol.policy", 1, 1000),
        replan_steps=_int(p, "replan_steps", "protocol.policy", 1, 1000),
        resize_size=_int(p, "resize_size", "protocol.policy", 16, 2048),
    )
    if policy.replan_steps > policy.action_horizon:
        raise ConfigError(
            f"protocol.policy.replan_steps ({policy.replan_steps}) > action_horizon ({policy.action_horizon})"
        )

    s = _get(proto, "simulator", "protocol")
    dummy = _get(s, "dummy_action", "protocol.simulator")
    if not (
        isinstance(dummy, list) and len(dummy) == policy.action_dim and all(isinstance(x, (int, float)) for x in dummy)
    ):
        raise ConfigError(f"protocol.simulator.dummy_action: expected {policy.action_dim} numbers, got {dummy!r}")
    simulator = SimulatorSpec(
        camera_resolution=_int(s, "camera_resolution", "protocol.simulator", 16, 2048),
        num_steps_wait=_int(s, "num_steps_wait", "protocol.simulator", 0, 1000),
        max_steps=_int(s, "max_steps", "protocol.simulator", 1, 10000),
        dummy_action=[float(x) for x in dummy],
        env_seed=_int(s, "env_seed", "protocol.simulator", 0, 2**31 - 1),
    )

    d = _get(proto, "data", "protocol")
    data = DataSpec(
        root=expand_env(_str(d, "root", "protocol.data"), env),
        suite=_str(d, "suite", "protocol.data", choices={"libero_90"}),
        demos_per_task=_int(d, "demos_per_task", "protocol.data", 1, 1000),
        image_size=_int(d, "image_size", "protocol.data", 16, 2048),
        lock_file=_str(d, "lock_file", "protocol.data"),
    )

    sp = _get(proto, "splits", "protocol")
    n_init = _int(sp, "num_init_states", "protocol.splits", 1, 10000)
    splits = SplitSpec(
        train_demo_indices=_range_list(sp, "train_demo_indices", "protocol.splits", 0, data.demos_per_task - 1),
        val_demo_indices=_range_list(sp, "val_demo_indices", "protocol.splits", 0, data.demos_per_task - 1),
        selection_init_states=_range_list(sp, "selection_init_states", "protocol.splits", 0, n_init - 1),
        test_init_states=_range_list(sp, "test_init_states", "protocol.splits", 0, n_init - 1),
    )
    if set(splits.train_demo_indices) & set(splits.val_demo_indices):
        raise ConfigError("protocol.splits: train_demo_indices and val_demo_indices overlap")
    if set(splits.selection_init_states) & set(splits.test_init_states):
        raise ConfigError("protocol.splits: selection_init_states and test_init_states overlap")

    e = _get(raw, "evaluation", "")
    evaluation = EvalSpec(
        episodes_per_task=_int(e, "episodes_per_task", "evaluation", 1, 10000),
        init_state_split=_str(e, "init_state_split", "evaluation", choices={"selection", "test"}),
        save_videos=_bool(e, "save_videos", "evaluation"),
        host=_str(e, "host", "evaluation"),
        port=_int(e, "port", "evaluation", 1, 65535),
    )
    pool = splits.test_init_states if evaluation.init_state_split == "test" else splits.selection_init_states
    if evaluation.episodes_per_task > len(pool):
        raise ConfigError(
            f"evaluation.episodes_per_task={evaluation.episodes_per_task} exceeds the {len(pool)} init states in the "
            f"'{evaluation.init_state_split}' split"
        )

    return ExperimentConfig(
        path=str(path),
        experiment_name=name,
        protocol_version=pv,
        protocol_sha256=stored,
        global_seed=global_seed,
        eval_seeds=eval_seeds,
        task_manifest=_str(proto, "task_manifest", "protocol"),
        checkpoint=checkpoint,
        policy=policy,
        simulator=simulator,
        data=data,
        splits=splits,
        evaluation=evaluation,
        output_root=expand_env(_str(raw, "output_root", ""), env),
        raw=raw,
    )


def load_experiment_config(
    path: os.PathLike | str, env: Optional[Mapping[str, str]] = None, overrides: Optional[Dict[str, Any]] = None
) -> ExperimentConfig:
    """Load and validate. `overrides` may only touch the non-frozen `evaluation` section (e.g. episodes_per_task)."""
    raw = load_yaml(path)
    if overrides:
        for k, v in overrides.items():
            top, _, sub = k.partition(".")
            if top != "evaluation" or not sub:
                raise ConfigError(f"override {k!r} rejected: only evaluation.* settings may be overridden at run time")
            raw.setdefault("evaluation", {})[sub] = v
    return parse_experiment_config(raw, str(path), env)


# ---------------------------------------------------------------------------------------------------------------------
# Seeds


def derive_seed(global_seed: int, *parts: Any) -> int:
    """Deterministic, platform-independent sub-seed (e.g. per task / episode). Same inputs -> same seed, always."""
    h = hashlib.sha256(json.dumps([global_seed, *parts], default=str).encode()).digest()
    return int.from_bytes(h[:4], "little") & 0x7FFFFFFF


def seed_everything(seed: int) -> None:
    """Seed Python and NumPy global RNGs (the only RNGs this package uses directly)."""
    random.seed(seed)
    np.random.seed(seed)
