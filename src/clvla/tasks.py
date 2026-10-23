"""Task manifest validation and LIBERO task resolution.

The frozen task sequence lives in configs/task_manifest.yaml and is the ONLY source of task identity and order. Code
must never iterate a LIBERO suite directly or rely on its implicit ordering.

A task's identity is the pair (libero_90_index, name), where name is the official LIBERO task name (BDDL file stem).
Both must agree with the pinned LIBERO benchmark (live check: `resolve_tasks`; offline check against the committed
snapshot configs/libero90_tasks.json: `check_against_snapshot`).
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from clvla.config import ConfigError, canonical_sha256, load_yaml

NUM_TASKS = 10
LIBERO_90_SIZE = 90
_NAME_RE = re.compile(r"^(KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+_[a-z0-9_]+$")
_REQUIRED = (
    "position",
    "libero_90_index",
    "name",
    "instruction",
    "scene",
    "category",
    "bddl_file",
    "init_states_file",
    "demo_file",
)


class ManifestError(ConfigError):
    pass


@dataclasses.dataclass(frozen=True)
class TaskSpec:
    position: int
    libero_90_index: int
    name: str
    instruction: str
    scene: str
    category: str
    bddl_file: str  # relative to LIBERO bddl_files root, e.g. libero_90/<name>.bddl
    init_states_file: str  # relative to LIBERO init_states root
    demo_file: str  # relative to the LIBERO data root, e.g. libero_90/<name>_demo.hdf5
    demo_sha256: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None

    @property
    def task_id(self) -> str:
        return f"libero_90/{self.libero_90_index:02d}/{self.name}"


@dataclasses.dataclass(frozen=True)
class TaskManifest:
    path: str
    version: str
    content_sha256: str
    benchmark: str
    tasks: List[TaskSpec]
    raw: Dict[str, Any]

    def by_position(self, position: int) -> TaskSpec:
        return self.tasks[position]

    def by_index(self, libero_90_index: int) -> TaskSpec:
        for t in self.tasks:
            if t.libero_90_index == libero_90_index:
                return t
        raise KeyError(f"LIBERO-90 index {libero_90_index} is not in the manifest")


def manifest_content_sha256(raw: Dict[str, Any]) -> str:
    """Checksum over everything except the checksum field itself."""
    body = {k: v for k, v in raw.items() if k != "content_sha256"}
    return canonical_sha256(body)


def parse_manifest(raw: Dict[str, Any], path: str = "<memory>", verify_checksum: bool = True) -> TaskManifest:
    for key in ("manifest_version", "benchmark", "tasks", "content_sha256", "frozen"):
        if key not in raw:
            raise ManifestError(f"{path}: missing required key: {key}")
    if raw["benchmark"] != "libero_90":
        raise ManifestError(f"{path}: benchmark must be 'libero_90', got {raw['benchmark']!r}")
    if raw["frozen"] is not True:
        raise ManifestError(f"{path}: manifest is not frozen (frozen: {raw['frozen']!r}); refuse to run on it")
    if verify_checksum:
        actual = manifest_content_sha256(raw)
        if raw["content_sha256"] != actual:
            raise ManifestError(
                f"{path}: content checksum mismatch (stored {str(raw['content_sha256'])[:12]}..., actual {actual[:12]}...). "
                "The frozen manifest was edited; regenerate it with scripts/build_task_manifest.py and bump manifest_version."
            )
    rows = raw["tasks"]
    if not isinstance(rows, list):
        raise ManifestError(f"{path}: tasks must be a list")
    if len(rows) != NUM_TASKS:
        raise ManifestError(f"{path}: expected exactly {NUM_TASKS} tasks, found {len(rows)}")

    tasks = []
    for i, row in enumerate(rows):
        missing = [k for k in _REQUIRED if k not in row]
        if missing:
            raise ManifestError(f"{path}: tasks[{i}] missing keys {missing}")
        idx = row["libero_90_index"]
        if isinstance(idx, bool) or not isinstance(idx, int) or not 0 <= idx < LIBERO_90_SIZE:
            raise ManifestError(f"{path}: tasks[{i}].libero_90_index={idx!r} is not a LIBERO-90 index (0..89)")
        if row["position"] != i:
            raise ManifestError(
                f"{path}: tasks[{i}].position={row['position']!r}; positions must be 0..{NUM_TASKS - 1} in order"
            )
        name = row["name"]
        if not isinstance(name, str) or not _NAME_RE.match(name):
            raise ManifestError(f"{path}: tasks[{i}].name={name!r} is not a LIBERO-90 task name")
        if row["bddl_file"] != f"libero_90/{name}.bddl":
            raise ManifestError(
                f"{path}: tasks[{i}].bddl_file={row['bddl_file']!r} does not belong to libero_90/{name}"
            )
        if row["demo_file"] != f"libero_90/{name}_demo.hdf5":
            raise ManifestError(f"{path}: tasks[{i}].demo_file={row['demo_file']!r} does not match task {name}")
        if not row["instruction"] or not isinstance(row["instruction"], str):
            raise ManifestError(f"{path}: tasks[{i}].instruction must be a non-empty string")
        known = set(_REQUIRED) | {"demo_sha256"}
        tasks.append(
            TaskSpec(
                **{k: row[k] for k in _REQUIRED},
                demo_sha256=row.get("demo_sha256"),
                extra={k: v for k, v in row.items() if k not in known},
            )
        )

    for attr in ("libero_90_index", "name", "instruction"):
        values = [getattr(t, attr) for t in tasks]
        dups = sorted({v for v in values if values.count(v) > 1}, key=str)
        if dups:
            raise ManifestError(f"{path}: duplicate {attr} values in manifest: {dups}")

    return TaskManifest(
        path=str(path),
        version=str(raw["manifest_version"]),
        content_sha256=raw["content_sha256"],
        benchmark=raw["benchmark"],
        tasks=tasks,
        raw=raw,
    )


def load_manifest(path: os.PathLike | str) -> TaskManifest:
    return parse_manifest(load_yaml(path), str(path))


# ---------------------------------------------------------------------------------------------------------------------
# Resolution against LIBERO


def check_against_snapshot(manifest: TaskManifest, snapshot_path: os.PathLike | str) -> None:
    """Offline check against configs/libero90_tasks.json (generated from the pinned LIBERO by build_task_manifest.py)."""
    snap = json.loads(Path(snapshot_path).read_text())
    rows = {r["idx"]: r for r in snap["libero_90"]}
    if len(rows) != LIBERO_90_SIZE:
        raise ManifestError(f"{snapshot_path}: snapshot has {len(rows)} LIBERO-90 tasks, expected {LIBERO_90_SIZE}")
    for t in manifest.tasks:
        ref = rows.get(t.libero_90_index)
        if ref is None or ref["name"] != t.name or ref["language"] != t.instruction:
            raise ManifestError(
                f"task {t.task_id}: does not match LIBERO-90 snapshot entry {ref and (ref['name'], ref['language'])}"
            )


@dataclasses.dataclass(frozen=True)
class ResolvedTask:
    spec: TaskSpec
    libero_task: Any  # libero.libero.benchmark.Task
    bddl_path: Path
    init_states_path: Path


def resolve_tasks(
    manifest: TaskManifest, suite: Any = None, libero_paths: Optional[Dict[str, str]] = None
) -> List[ResolvedTask]:
    """Resolve every manifest entry against the official LIBERO-90 suite (sim env). Fails on any disagreement.

    `suite` / `libero_paths` can be injected for tests; by default they come from the pinned LIBERO install.
    """
    if suite is None or libero_paths is None:
        from libero.libero import benchmark, get_libero_path

        suite = suite or benchmark.get_benchmark_dict()["libero_90"]()
        libero_paths = libero_paths or {k: get_libero_path(k) for k in ("bddl_files", "init_states")}
    if suite.n_tasks != LIBERO_90_SIZE:
        raise ManifestError(f"LIBERO suite reports {suite.n_tasks} tasks, expected {LIBERO_90_SIZE} for libero_90")
    out = []
    for t in manifest.tasks:
        lt = suite.get_task(t.libero_90_index)
        problems = []
        if lt.name != t.name:
            problems.append(f"name {lt.name!r} != manifest {t.name!r}")
        if lt.language != t.instruction:
            problems.append(f"instruction {lt.language!r} != manifest {t.instruction!r}")
        bddl = Path(libero_paths["bddl_files"]) / lt.problem_folder / lt.bddl_file
        if f"{lt.problem_folder}/{lt.bddl_file}" != t.bddl_file:
            problems.append(f"bddl {lt.problem_folder}/{lt.bddl_file} != manifest {t.bddl_file}")
        if not bddl.is_file():
            problems.append(f"bddl file missing: {bddl}")
        init = Path(libero_paths["init_states"]) / lt.problem_folder / lt.init_states_file
        if f"{lt.problem_folder}/{lt.init_states_file}" != t.init_states_file:
            problems.append(f"init states {lt.problem_folder}/{lt.init_states_file} != manifest {t.init_states_file}")
        if not init.is_file():
            problems.append(f"init states file missing: {init}")
        if problems:
            raise ManifestError(f"task {t.task_id} does not resolve in official LIBERO-90: " + "; ".join(problems))
        out.append(ResolvedTask(spec=t, libero_task=lt, bddl_path=bddl, init_states_path=init))
    return out


def init_state_indices(split_indices: Sequence[int], n: int) -> List[int]:
    """The first n init-state indices of a split, in the split's fixed order."""
    if n > len(split_indices):
        raise ValueError(f"requested {n} episodes but the split only has {len(split_indices)} init states")
    return list(split_indices[:n])
