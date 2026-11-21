"""Task phases from events, not fixed time windows: approach takes 30-80 steps from episode to episode, so a time quarter
would mix approach and contact states.

  approach      gripper farther than `near` from the target, no contact, target not yet moved
  precontact    within `near`, no contact, not moved
  contact       robot touches the target, not moved
  manipulation  target displaced by at least `moved` from its start (joint angle or position); sticky once reached

For t35 this reproduces opd_lora_train.phase_of with opening and late merged into manipulation (door at or below -0.05 rad).
A task whose events cannot be detected falls back to equal time quarters (flagged by the caller).
"""

from __future__ import annotations

import numpy as np

PHASES = ("approach", "precontact", "contact", "manipulation")
NEAR, MOVED = 0.12, 0.05


def label(dist, contact, displacement, near: float = NEAR, moved: float = MOVED) -> list:
    """Per-step arrays (distance gripper-target in m, contact flag, |target displacement from its start|) -> phase names."""
    dist, contact, disp = np.asarray(dist, float), np.asarray(contact, bool), np.abs(np.asarray(displacement, float))
    assert dist.shape == contact.shape == disp.shape
    out, done = [], False
    for d, c, m in zip(dist, contact, disp):
        done = done or m >= moved
        out.append("manipulation" if done else "contact" if c else "precontact" if d <= near else "approach")
    return out


def label_t35(snaps, door0: float = 0.0) -> list:
    """opd_diag_rollout.Geometry snapshots (index 6 door angle, 7 grip-handle distance, 8 contact) -> phases."""
    s = np.asarray(snaps, float)
    return label(s[:, 7], s[:, 8] > 0, s[:, 6] - door0)


def label_by_time(n: int) -> list:
    """Fallback: equal time quarters of an n-step episode."""
    return [PHASES[min(3, (4 * i) // max(n, 1))] for i in range(n)]


def within_phase_time(phases) -> np.ndarray:
    """Each step's position inside its own contiguous phase segment, in [0, 1] (0.5 for a one-step segment)."""
    t = np.zeros(len(phases))
    i = 0
    while i < len(phases):
        j = i
        while j + 1 < len(phases) and phases[j + 1] == phases[i]:
            j += 1
        n = j - i + 1
        t[i : j + 1] = 0.5 if n == 1 else np.arange(n) / (n - 1)
        i = j + 1
    return t
