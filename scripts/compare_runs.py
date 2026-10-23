"""Compare two evaluation runs episode by episode (reproducibility check). Exit 1 if any episode differs.

Usage: scripts/py-policy scripts/compare_runs.py <run_dir_a> <run_dir_b>
"""

from __future__ import annotations

import sys
from pathlib import Path

from clvla.evaluation import read_episode_results


def main() -> int:
    a, b = (Path(p) for p in sys.argv[1:3])
    key = lambda r: (r.env_seed, r.task_position, r.init_state_idx)  # noqa: E731
    ra = {key(r): (r.success, r.steps) for r in read_episode_results(a / "episodes.jsonl")}
    rb = {key(r): (r.success, r.steps) for r in read_episode_results(b / "episodes.jsonl")}
    common = sorted(set(ra) & set(rb))
    diff = [(k, ra[k], rb[k]) for k in common if ra[k] != rb[k]]
    print(f"{len(common)} common episodes; {len(diff)} differ (success, steps)")
    for k, x, y in diff:
        print(f"  seed {k[0]} task pos {k[1]} init {k[2]}: {x} vs {y}")
    return 1 if diff or not common else 0


if __name__ == "__main__":
    sys.exit(main())
