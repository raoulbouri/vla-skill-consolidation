"""Action-space trust region by step backtracking (continual OPD, follow-up B). NumPy only.

The optimizer (global clip, then AdamW) proposes a parameter step D = theta' - theta. Before it is applied, the trainer
measures how far it moves the student's velocity field on this update's current-task observations (fresh rollout and
replay; old-task observations are not used), at the states of each observation's first noise chain under theta:
    r_j(s) = rms(v_{theta + s D}(x_jk, t_k) - v_theta(x_jk, t_k)) / rms(v_theta(x_jk, t_k))   over chain steps k, all dims
    r(s)   = mean_j r_j(s)
The step is applied at s = 1 if r(1) <= delta, otherwise halved (s = 1/2, 1/4, ...) until r(s) <= delta, at most
`max_halvings` times (then the smallest step is applied and the update is logged as capped).
delta = k x the median of the applied r over the last `window` updates, so the bound calibrates itself as the clipping
threshold does (k x the batch median); the first `warmup` updates only record r. Only the applied parameter step is
scaled: the optimizer state keeps the full update (Adam's moments see the unscaled, globally clipped gradient).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional, Tuple

import numpy as np


def relative_change(v_old: np.ndarray, v_new: np.ndarray) -> float:
    """rms(v_new - v_old) / rms(v_old) over all entries."""
    v_old = np.asarray(v_old, np.float64)
    d = np.asarray(v_new, np.float64) - v_old
    return float(np.sqrt(np.mean(d**2)) / max(np.sqrt(np.mean(v_old**2)), 1e-30))


@dataclass
class TrustRegion:
    k: float
    window: int = 20
    warmup: int = 5
    max_halvings: int = 6
    history: list = field(default_factory=list)  # applied r of every earlier update

    def __post_init__(self):
        assert self.k >= 1.0, "trust region: k is a multiplier >= 1 of the running median"

    def threshold(self) -> Optional[float]:
        if len(self.history) < self.warmup:
            return None
        return self.k * float(np.median(self.history[-self.window:]))

    def search(self, rel: Callable[[float], Tuple[float, float]]) -> Tuple[float, dict]:
        """rel(s) -> (mean r, max r) of the step scaled by s. Returns the applied scale and the logged row."""
        thr = self.threshold()
        full = rel(1.0)
        s, h, r = 1.0, 0, full
        while thr is not None and r[0] > thr and h < self.max_halvings:
            h += 1
            s = 0.5**h
            r = rel(s)
        self.history.append(float(r[0]))
        return s, {"tr_k": self.k, "tr_threshold": thr, "tr_rel_full": full[0], "tr_rel_full_max": full[1], "tr_rel": r[0], "tr_rel_max": r[1],
                   "tr_scale": s, "tr_halvings": h, "tr_capped": bool(thr is not None and r[0] > thr)}

    def get_state(self) -> dict:
        return {"k": self.k, "window": self.window, "warmup": self.warmup, "max_halvings": self.max_halvings, "history": list(self.history)}

    def set_state(self, st: dict) -> None:
        assert (st["k"], st["window"], st["warmup"], st["max_halvings"]) == (self.k, self.window, self.warmup, self.max_halvings), "resume: trust-region settings differ"
        self.history = [float(x) for x in st["history"]]
