"""Orchestration for independent task specialists (and the single-task smoke). Pure Python; JAX stays in the backend.

Invariants:
  * every specialist starts from the original pi05_libero checkpoint (the backend's weight loader always points at
    the base checkpoint's params; nothing here can pass another run's weights)
  * one task per run directory; runs are executed strictly one after another, guarded by an exclusive lock file so
    two runner processes cannot train concurrently; no background workers or job submission
  * executing requires an explicit task selection; there is no "all tasks" default
  * before starting, the GPU guard must report the GPU free; otherwise the run is refused (nothing is killed)
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence

from clvla.training_core import task_seed


class RunnerRefused(RuntimeError):
    pass


def parse_selection(manifest: Any, positions: Optional[str], indices: Optional[str], execute: bool) -> List[Any]:
    """Tasks (TaskSpec) in the requested order. With execute=True an explicit selection is mandatory."""
    if positions and indices:
        raise RunnerRefused("give either --positions or --indices, not both")
    if positions:
        return [manifest.by_position(int(p)) for p in positions.split(",")]
    if indices:
        return [manifest.by_index(int(i)) for i in indices.split(",")]
    if execute:
        raise RunnerRefused("--execute requires an explicit --positions/--indices selection (never all tasks by default)")
    return list(manifest.tasks)


def task_seeds(global_seed: int, tasks: Iterable[Any]) -> Dict[int, int]:
    return {t.libero_90_index: task_seed(global_seed, t.libero_90_index) for t in tasks}


def default_gpu_guard(repo_root: Path) -> Callable[[], None]:
    def guard() -> None:
        r = subprocess.run([str(repo_root / "scripts/gpu_guard.sh")], capture_output=True, text=True)
        if r.returncode != 0:
            raise RunnerRefused(f"GPU guard refused: {r.stdout.strip() or r.stderr.strip()} (no process was touched)")

    return guard


def max_concurrent_trainers() -> int:
    """$CLVLA_MAX_CONCURRENT_TRAINING (default 1 = serial). Capped at 3: each run must also fit its own memory share."""
    n = int(os.environ.get("CLVLA_MAX_CONCURRENT_TRAINING", "1"))
    if not 1 <= n <= 3:
        raise RunnerRefused(f"CLVLA_MAX_CONCURRENT_TRAINING={n}: must be 1..3")
    return n


@contextlib.contextmanager
def exclusive_training_lock(output_root: Path, max_concurrent: Optional[int] = None) -> Iterator[Path]:
    """At most `max_concurrent` trainers at a time across processes (default from max_concurrent_trainers(), i.e. 1).
    Slot 0 keeps the historical name `.training.lock`. A stale lock (crash) must be removed by hand after checking."""
    output_root.mkdir(parents=True, exist_ok=True)
    n = max_concurrent_trainers() if max_concurrent is None else max_concurrent
    names = [".training.lock"] + [f".training.slot{i}.lock" for i in range(1, n)]
    for name in names:
        lock = output_root / name
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            continue
    else:
        held = ", ".join(f"{x}: {(output_root / x).read_text().strip()}" for x in names if (output_root / x).exists())
        raise RunnerRefused(f"all {n} training slot(s) are held ({held}); runs are serial only unless "
                            "CLVLA_MAX_CONCURRENT_TRAINING allows more")
    try:
        os.write(fd, f"pid={os.getpid()}\n".encode())
        os.close(fd)
        yield lock
    finally:
        with contextlib.suppress(FileNotFoundError):
            lock.unlink()


def run_serial(tasks: Sequence[Any], run_one: Callable[[Any], Dict[str, Any]], *, guard: Callable[[], None],
               lock_root: Path, stop_on_failure: bool = True) -> List[Dict[str, Any]]:
    """Run `run_one(task)` for each task, one at a time, under one lock; the GPU guard runs before every task."""
    results = []
    with exclusive_training_lock(lock_root):
        for t in tasks:
            guard()
            try:
                results.append({"libero_90_index": t.libero_90_index, "status": "completed", **run_one(t)})
            except Exception as e:  # noqa: BLE001 - recorded; stop by default so failures are looked at
                results.append({"libero_90_index": t.libero_90_index, "status": "failed",
                                "error": f"{type(e).__name__}: {e}"})
                if stop_on_failure:
                    break
    return results


SUMMARY_FIELDS = ("libero_90_index", "task_name", "run_dir", "status", "seed", "num_steps", "final_train_loss",
                  "final_val_loss", "best_val_loss", "wall_time_s", "mean_step_time_s", "trainable_params",
                  "checkpoint_dir", "checkpoint_fingerprint_sha256", "peak_gpu_bytes", "control_task")


def summarize_runs(run_dirs: Iterable[Path], control_indices: Sequence[int] = ()) -> List[Dict[str, Any]]:
    """One row per run directory (completed or not), from run_metadata.json / summary.json / history.jsonl."""
    rows = []
    for d in sorted(Path(p) for p in run_dirs):
        meta_p, sum_p = d / "run_metadata.json", d / "summary.json"
        if not meta_p.is_file():
            continue
        meta = json.loads(meta_p.read_text())
        s = json.loads(sum_p.read_text()) if sum_p.is_file() else {"status": "incomplete"}
        vals = []
        hist = d / "history.jsonl"
        if hist.is_file():
            vals = [json.loads(line)["val_loss"] for line in hist.read_text().splitlines()
                    if line.strip() and "val_loss" in json.loads(line)]
        final_ck = (s.get("checkpoints") or [{}])[-1]
        idx = meta["task"]["libero_90_index"]
        rows.append({
            "libero_90_index": idx, "task_name": meta["task"]["name"], "run_dir": str(d), "status": s.get("status"),
            "seed": meta.get("seed"), "num_steps": s.get("num_steps"), "final_train_loss": s.get("final_train_loss"),
            "final_val_loss": s.get("final_val_loss"), "best_val_loss": min(vals) if vals else None,
            "wall_time_s": s.get("wall_time_s"), "mean_step_time_s": s.get("mean_step_time_s"),
            "trainable_params": (s.get("init") or {}).get("trainable_params"),
            "checkpoint_dir": final_ck.get("dir"), "checkpoint_fingerprint_sha256": s.get("trained_checkpoint_fingerprint"),
            "peak_gpu_bytes": (s.get("memory") or {}).get("peak_bytes_in_use"), "control_task": idx in control_indices,
        })
    return rows


def write_summary_files(rows: List[Dict[str, Any]], out_dir: Path, stem: str = "specialists_summary") -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{stem}.json").write_text(json.dumps(rows, indent=1, default=str))
    with (out_dir / f"{stem}.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def assemble_matrix(records: Iterable[Dict[str, Any]], expert_order: Sequence[int], task_order: Sequence[int]
                    ) -> Dict[str, Any]:
    """Expert-by-task success matrix from evaluation records {expert, task, successes, episodes, infra_errors}.

    Missing cells stay None (never imputed). Rows = specialists (by LIBERO-90 index), columns = evaluated tasks."""
    cells: Dict[tuple, Dict[str, Any]] = {}
    for r in records:
        key = (int(r["expert"]), int(r["task"]))
        if key in cells:
            raise ValueError(f"duplicate evaluation for expert {key[0]} on task {key[1]}")
        cells[key] = r
    matrix = [[(cells[(e, t)]["successes"] / cells[(e, t)]["episodes"]) if (e, t) in cells and cells[(e, t)]["episodes"]
               else None for t in task_order] for e in expert_order]
    counts = [[[cells[(e, t)]["successes"], cells[(e, t)]["episodes"]] if (e, t) in cells else None for t in task_order]
              for e in expert_order]
    return {"experts": list(expert_order), "tasks": list(task_order), "success_rate": matrix, "counts": counts,
            "infra_errors": sum(int(c.get("infra_errors", 0)) for c in cells.values()),
            "diagonal": [matrix[i][list(task_order).index(e)] if e in task_order else None
                         for i, e in enumerate(expert_order)]}
