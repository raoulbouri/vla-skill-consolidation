"""Deterministic task evaluation, success recording, episode metadata and aggregate metrics.

`run_episode` reproduces the control loop of OpenPI examples/libero/main.py at the pinned commit:
  env.reset(); obs = env.set_init_state(init_states[i])
  for t < max_steps + num_steps_wait:
      t < num_steps_wait   -> step LIBERO_DUMMY_ACTION (objects settle)
      otherwise            -> if the action plan is empty, query the policy and keep the first replan_steps actions;
                              step one action; success iff env reports done
Difference from main.py (recorded, not hidden): an exception inside the loop is stored in `error` and the episode is
marked `infra_error`. main.py logs it and counts the episode as a plain failure. Success rates below are computed both
ways: over all episodes (main.py-compatible) and over episodes without infra errors.
"""

from __future__ import annotations

import collections
import csv
import dataclasses
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from clvla.policy import build_observation

EPISODE_FIELDS = (
    "run_id",
    "task_position",
    "libero_90_index",
    "task_name",
    "instruction",
    "episode_idx",
    "init_state_idx",
    "env_seed",
    "success",
    "infra_error",
    "error",
    "steps",
    "policy_calls",
    "wall_time_s",
    "mean_inference_s",
    "video_path",
)


@dataclasses.dataclass
class EpisodeResult:
    run_id: str
    task_position: int
    libero_90_index: int
    task_name: str
    instruction: str
    episode_idx: int
    init_state_idx: int
    env_seed: int
    success: bool
    infra_error: bool
    error: str
    steps: int  # policy actions executed (excludes the num_steps_wait settle steps)
    policy_calls: int
    wall_time_s: float
    mean_inference_s: float
    video_path: str = ""

    def as_row(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in EPISODE_FIELDS}


@dataclasses.dataclass
class LoopOutcome:
    success: bool
    steps: int
    policy_calls: int
    error: str
    frames: List[np.ndarray]
    inference_s: List[float]


def run_episode(
    env: Any,
    policy: Callable[[Dict[str, Any]], np.ndarray],
    init_state: Any,
    instruction: str,
    *,
    max_steps: int,
    num_steps_wait: int,
    replan_steps: int,
    resize_size: int,
    dummy_action: Sequence[float],
    record_frames: bool = False,
    build_obs: Callable[..., Dict[str, Any]] = build_observation,
) -> LoopOutcome:
    env.reset()
    obs = env.set_init_state(init_state)
    plan: collections.deque = collections.deque()
    frames: List[np.ndarray] = []
    inference_s: List[float] = []
    t, calls, executed, done, error = 0, 0, 0, False, ""
    while t < max_steps + num_steps_wait:
        try:
            if t < num_steps_wait:
                obs, _, done, _ = env.step(list(dummy_action))
                t += 1
                continue
            request = build_obs(obs, instruction, resize_size)
            if record_frames:
                frames.append(request["observation/image"])
            if not plan:
                t0 = time.perf_counter()
                chunk = np.asarray(policy(request))
                inference_s.append(time.perf_counter() - t0)
                calls += 1
                if len(chunk) < replan_steps:
                    raise RuntimeError(f"policy returned {len(chunk)} actions < replan_steps={replan_steps}")
                plan.extend(chunk[:replan_steps])
            action = plan.popleft()
            obs, _, done, _ = env.step(np.asarray(action).tolist())
            executed += 1
            if done:
                break
            t += 1
        except Exception as e:  # noqa: BLE001 - recorded as an infrastructure error, never swallowed silently
            error = f"{type(e).__name__}: {e}"
            done = False
            break
    return LoopOutcome(
        success=bool(done), steps=executed, policy_calls=calls, error=error, frames=frames, inference_s=inference_s
    )


def make_libero_env(bddl_path: str, resolution: int, seed: int):
    """Official LIBERO OffScreenRenderEnv exactly as main.py builds it (sim env only)."""
    from libero.libero.envs import OffScreenRenderEnv

    env = OffScreenRenderEnv(bddl_file_name=str(bddl_path), camera_heights=resolution, camera_widths=resolution)
    env.seed(seed)  # main.py: "seed seems to affect object positions even when using fixed initial state"
    return env


# ---------------------------------------------------------------------------------------------------------------------
# Metrics


def wilson_interval(successes: int, n: int, z: float = 1.96) -> List[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


def _rate(s: int, n: int) -> float:
    return s / n if n else float("nan")


def aggregate(results: Iterable[EpisodeResult]) -> Dict[str, Any]:
    """Per-task and overall success. Task means are unweighted (each task counts once)."""
    results = list(results)
    by_task: Dict[int, List[EpisodeResult]] = collections.OrderedDict()
    for r in sorted(results, key=lambda r: (r.task_position, r.episode_idx)):
        by_task.setdefault(r.task_position, []).append(r)
    per_task = []
    for pos, rs in by_task.items():
        n, s = len(rs), sum(r.success for r in rs)
        n_err = sum(r.infra_error for r in rs)
        clean = [r for r in rs if not r.infra_error]
        per_task.append(
            {
                "task_position": pos,
                "libero_90_index": rs[0].libero_90_index,
                "task_name": rs[0].task_name,
                "instruction": rs[0].instruction,
                "episodes": n,
                "successes": s,
                "success_rate": _rate(s, n),
                "success_rate_ci95": wilson_interval(s, n),
                "infra_errors": n_err,
                "success_rate_excl_infra": _rate(sum(r.success for r in clean), len(clean)),
                "mean_steps": float(np.mean([r.steps for r in rs])) if rs else float("nan"),
            }
        )
    n_all, s_all = len(results), sum(r.success for r in results)
    rates = [t["success_rate"] for t in per_task]
    return {
        "num_tasks": len(per_task),
        "episodes": n_all,
        "successes": s_all,
        "infra_errors": sum(r.infra_error for r in results),
        "success_rate_pooled": _rate(s_all, n_all),
        "success_rate_task_mean": float(np.mean(rates)) if rates else float("nan"),
        "per_task": per_task,
    }


# ---------------------------------------------------------------------------------------------------------------------
# Result files


def write_episode_results(out_dir: Path, results: Sequence[EpisodeResult]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EPISODE_FIELDS)
        w.writeheader()
        for r in results:
            w.writerow(r.as_row())
    with (out_dir / "episodes.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r.as_row()) + "\n")


def append_episode_result(out_dir: Path, result: EpisodeResult) -> None:
    """Incremental log so a crashed run still leaves every finished episode on disk."""
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "episodes.partial.jsonl").open("a") as f:
        f.write(json.dumps(result.as_row()) + "\n")


def write_summary(out_dir: Path, summary: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(summary)
    if extra:
        payload.update(extra)
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    cols = (
        "task_position",
        "libero_90_index",
        "task_name",
        "episodes",
        "successes",
        "success_rate",
        "infra_errors",
        "success_rate_excl_infra",
        "mean_steps",
    )
    with (out_dir / "per_task.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for row in summary["per_task"]:
            w.writerow(row)


def read_episode_results(path: Path) -> List[EpisodeResult]:
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    return [EpisodeResult(**row) for row in rows]
