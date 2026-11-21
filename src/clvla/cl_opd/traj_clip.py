"""Per-trajectory gradient clipping with current-batch, source-wise thresholds (continual OPD, Stage 2). NumPy only.

The trainer differentiates one trajectory (observation x noise chain) at a time. Its batch gradient is
    g = sum_i a_i g_i,   a_i = (1 - w) / n_new  for current-task trajectories (fresh AND replay: one new-task mean),
                         a_i = w / n_old        for old-task trajectories   (a_i = 1 / n_new when there are none).
Clipping multiplies a_i by
    s_i = min(1, tau_c(i) / |g_i|),   tau_c = k * median{ |g_j| : j in the source class c of this batch },
classes fresh / replay / old. Only the threshold is class-wise: the loss weights a_i, and therefore the new/old split
and the fresh/replay proportion inside the new-task mean, are untouched. A clipped trajectory keeps norm tau_c (it is
scaled, never dropped). |g_i| is the norm over all trainable parameters (action expert and heads together).

Two passes over the batch. Pass 1 is the trainer's usual loop (unclipped sums) plus |g_i|; the thresholds then follow
from the current batch. Pass 2 recomputes only the trajectories with s_i < 1 and adds a_i (s_i - 1) g_i, which turns the
unclipped g into the clipped one exactly, provided the recomputed g_i equals the first (the trainer checks the norms).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CLASSES = ("fresh", "replay", "old")


def source_class(j: int, n_fresh: int, n_new: int) -> str:
    """Observation j of a trainer batch: [0, n_fresh) fresh rollout, [n_fresh, n_new) replay buffer, [n_new, ...) old task."""
    return "fresh" if j < n_fresh else "replay" if j < n_new else "old"


def loss_weights(classes, old_weight: float | None) -> np.ndarray:
    """a_i exactly as the trainer forms g: (1 - w) mean over new-task trajectories + w mean over old-task ones."""
    c = np.asarray(classes)
    is_old = c == "old"
    n_new, n_old = int((~is_old).sum()), int(is_old.sum())
    if n_old == 0:
        return np.where(is_old, 0.0, 1.0 / n_new)
    assert old_weight is not None and n_new > 0
    return np.where(is_old, old_weight / n_old, (1.0 - old_weight) / n_new)


def thresholds(norms, classes, k: float) -> np.ndarray:
    """tau_i = k x the median norm of trajectory i's own class in this batch (inf: no clipping)."""
    n, c = np.asarray(norms, np.float64), np.asarray(classes)
    t = np.full(n.shape, np.inf)
    if not k or not math.isfinite(k):
        return t
    for cl in dict.fromkeys(c.tolist()):
        m = c == cl
        t[m] = k * float(np.median(n[m]))
    return t


def scales(norms, taus) -> np.ndarray:
    n, t = np.asarray(norms, np.float64), np.asarray(taus, np.float64)
    return np.where(n > t, t / np.maximum(n, 1e-300), 1.0)


@dataclass
class TrajClip:
    """Bookkeeping for one batch. record() in pass 1 (in trajectory order), then plan(), then the trainer recomputes
    plan()['todo'] and adds coef(i) * g_i to its correction sum."""

    k: float
    old_weight: float | None
    meta: list = field(default_factory=list)  # (j, m, class)
    sq: list = field(default_factory=list)  # {group: squared norm}
    fp: list = field(default_factory=list)  # fingerprint of g_i (per-leaf sums and sums of squares), to check recomputation

    def record(self, j: int, m: int, cls: str, sq_by_group: dict, fingerprint=None) -> int:
        self.meta.append((j, m, cls))
        self.sq.append({g: float(v) for g, v in sq_by_group.items()})
        self.fp.append(None if fingerprint is None else np.asarray(fingerprint, np.float64))
        return len(self.meta) - 1

    def plan(self) -> dict:
        self.cls = np.array([c for _, _, c in self.meta])
        self.norm = np.sqrt(np.array([sum(d.values()) for d in self.sq], np.float64))
        self.a = loss_weights(self.cls, self.old_weight)
        self.tau = thresholds(self.norm, self.cls, self.k)
        self.s = scales(self.norm, self.tau)
        self.todo = [int(i) for i in np.flatnonzero(self.s < 1.0)]
        return {"todo": self.todo, "coef": {i: float(self.a[i] * (self.s[i] - 1.0)) for i in self.todo}}

    def coef(self, i: int) -> float:
        return float(self.a[i] * (self.s[i] - 1.0))

    def top(self, clipped: bool) -> int:
        """Trajectory with the largest a_i (s_i) |g_i|: the one with the largest Cauchy-Schwarz bound on its influence."""
        return int(np.argmax(self.a * (self.s if clipped else 1.0) * self.norm))

    def stats(self, g_norm_unclipped: float, g_norm_clipped: float) -> dict:
        """Logged every update. clip_infl_bound_*: max_i a_i s_i |g_i| / |g| >= max_i a_i <g_i, g> / |g|^2 (the influence)."""
        out = {"clip_k": self.k, "clip_n": len(self.todo), "clip_frac": float(np.mean(self.s < 1.0)),
               "clip_infl_bound_unclipped": float((self.a * self.norm).max() / g_norm_unclipped) if g_norm_unclipped > 0 else None,
               "clip_infl_bound_clipped": float((self.a * self.s * self.norm).max() / g_norm_clipped) if g_norm_clipped > 0 else None,
               "traj_norm_median": float(np.median(self.norm)), "traj_norm_max": float(self.norm.max())}
        for cl in CLASSES:
            m = self.cls == cl
            if m.any():
                med = float(np.median(self.norm[m]))
                out.update({f"clip_frac_{cl}": float(np.mean(self.s[m] < 1.0)), f"clip_n_{cl}": int((self.s[m] < 1.0).sum()), f"traj_norm_median_{cl}": med,
                            f"traj_norm_max_{cl}": float(self.norm[m].max()), f"traj_norm_max_over_median_{cl}": float(self.norm[m].max() / med) if med > 0 else None,
                            f"clip_tau_{cl}": float(self.tau[m][0]) if math.isfinite(self.tau[m][0]) else None, f"n_traj_{cl}": int(m.sum())})
        return out

    def arrays(self) -> dict:
        """Per-trajectory record for the run directory (clip/update_UUU.npz)."""
        groups = sorted({g for d in self.sq for g in d})
        return {"obs": np.array([j for j, _, _ in self.meta]), "chain": np.array([m for _, m, _ in self.meta]), "cls": self.cls, "a": self.a,
                "norm": self.norm, "tau": self.tau, "s": self.s, **{f"sq_{g}": np.array([d.get(g, 0.0) for d in self.sq]) for g in groups}}
