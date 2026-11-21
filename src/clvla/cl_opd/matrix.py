"""Success-rate matrix S[i][j]: success on task j of the student saved after training task i (row i), from the evaluation
jsonl rows. Diagonal = success right after learning; below it = retention; S[i][i+1] = zero-shot on the next task.

  final_avg          mean_j S[T-1][j]
  nbt_j              S[j][j] - S[T-1][j]   (negative backward transfer: forgetting; > 0 means forgotten)
  nbt_norm_j         nbt_j / S[j][j]       (share of what was learned that was lost; None if S[j][j] = 0)
  fwt_j              S[j-1][j] - base_j    (zero-shot gain on task j from training tasks 0..j-1)
"""

from __future__ import annotations

import json
from pathlib import Path


def wilson(x: int, n: int, z: float = 1.96):
    if n == 0:
        return [None, None]
    p = x / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * (p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5 / (1 + z * z / n)
    return [c - h, c + h]


def cells(rows, policy_of_row: dict, task_of_position: dict, mode: str = "ode") -> dict:
    """rows: evaluation jsonl dicts; policy_of_row: {policy_id: stage index or 'base'}. -> {(stage, task): [successes, n]}"""
    out = {}
    for r in rows:
        if r["mode"] != mode or r["policy_id"] not in policy_of_row:
            continue
        k = (policy_of_row[r["policy_id"]], task_of_position[r["position"]])
        s = out.setdefault(k, [0, 0])
        s[0] += int(bool(r["success"]))
        s[1] += 1
    return out


def summarize(task_ids: list, c: dict) -> dict:
    """task_ids in curriculum order; c from cells(). Returns the matrix (rates, None where not evaluated) and metrics."""
    T = len(task_ids)
    rate = lambda k: c[k][0] / c[k][1] if k in c and c[k][1] else None  # noqa: E731
    S = [[rate((i, t)) for t in task_ids] for i in range(T)]
    last = max((i for i in range(T) if any(S[i][j] is not None for j in range(i + 1))), default=None)
    out = {"tasks": task_ids, "S": S, "n": [[c.get((i, t), [0, 0])[1] for t in task_ids] for i in range(T)],
           "ci95": [[wilson(*c[(i, t)]) if (i, t) in c else None for t in task_ids] for i in range(T)],
           "base": {t: rate(("base", t)) for t in task_ids}, "last_stage": last}
    if last is None:
        return out
    learned = list(range(last + 1))
    final = [S[last][j] for j in learned]
    out["final_avg"] = sum(final) / len(final) if None not in final else None
    out["nbt"] = {task_ids[j]: S[j][j] - S[last][j] for j in learned[:-1] if S[j][j] is not None and S[last][j] is not None}
    out["nbt_avg"] = sum(out["nbt"].values()) / len(out["nbt"]) if out["nbt"] else None
    j_of = {t: j for j, t in enumerate(task_ids)}
    out["nbt_norm"] = {t: (v / S[j_of[t]][j_of[t]] if S[j_of[t]][j_of[t]] else None) for t, v in out["nbt"].items()}
    out["fwt"] = {task_ids[j]: S[j - 1][j] - out["base"][task_ids[j]] for j in range(1, T) if j - 1 <= last and S[j - 1][j] is not None and out["base"][task_ids[j]] is not None}
    return out


def from_files(eval_dir, task_ids: list, positions: list, mode: str = "ode") -> dict:
    rows = [json.loads(l) for f in sorted(Path(eval_dir).glob("*.jsonl")) for l in f.read_text().splitlines() if l.strip()]
    pol = {f"stage{i}": i for i in range(len(task_ids))}
    pol["base"] = "base"
    return summarize(task_ids, cells(rows, pol, dict(zip(positions, task_ids)), mode))


def heatmap(m: dict, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    T = len(m["tasks"])
    a = np.array([[np.nan if v is None else v for v in r] for r in m["S"]], float)
    fig, ax = plt.subplots(figsize=(1.1 * T + 2, 1.0 * T + 1.5))
    im = ax.imshow(a, vmin=0, vmax=1, cmap="viridis")
    for i in range(T):
        for j in range(T):
            if not np.isnan(a[i, j]):
                ax.text(j, i, f"{a[i, j]:.2f}", ha="center", va="center", color="w" if a[i, j] < 0.6 else "k", fontsize=9)
    ax.set_xticks(range(T), m["tasks"], rotation=45, ha="right")
    ax.set_yticks(range(T), [f"after {t}" for t in m["tasks"]])
    ax.set_xlabel("evaluated task")
    ax.set_title(f"success (ODE); final avg {m.get('final_avg')}, NBT avg {m.get('nbt_avg')}", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
