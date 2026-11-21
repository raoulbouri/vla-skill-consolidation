"""Old-task part of each update's batch.

n_old observations per update are split equally over the old tasks, then equally over the phases each task has in the
buffer; remainders rotate with the update index so no task or phase is favoured over time. Inside a (task, phase) cell
rows are drawn without replacement from a seeded permutation; a new permutation (next epoch) starts only when the cell
is used up, so every stored transition is used equally often. The whole state is the cursor and epoch per cell, so a
resumed run draws exactly the same rows.
"""

from __future__ import annotations

import numpy as np

from clvla.cl_opd.phases import PHASES


def split(n: int, k: int, rotate: int) -> list:
    """n items over k slots, remainder to slots rotate, rotate+1, ... (mod k)."""
    base, rem = divmod(n, k)
    out = [base] * k
    for j in range(rem):
        out[(rotate + j) % k] += 1
    return out


class OldTaskSampler:
    def __init__(self, phase_index: dict, seed: int, state: dict | None = None):
        """phase_index: {task: {phase: [row, ...]}} from FixedTotalBuffer.phase_index()."""
        self.idx, self.seed = phase_index, int(seed)
        self.tasks = list(phase_index)
        self.cells = {f"{t}|{p}": rows for t in self.tasks for p, rows in phase_index[t].items() if rows}
        self.state = state or {c: {"epoch": 0, "cursor": 0} for c in self.cells}
        assert set(self.state) == set(self.cells), "sampler state does not match the buffer"

    def _perm(self, cell: str, epoch: int) -> list:
        h = int.from_bytes(cell.encode(), "little") % (2**31)
        r = np.random.default_rng([self.seed, h, epoch])
        rows = self.cells[cell]
        return [rows[i] for i in r.permutation(len(rows))]

    def _draw(self, cell: str, n: int) -> list:
        out, st = [], self.state[cell]
        while len(out) < n:
            perm = self._perm(cell, st["epoch"])
            take = min(n - len(out), len(perm) - st["cursor"])
            out += perm[st["cursor"] : st["cursor"] + take]
            st["cursor"] += take
            if st["cursor"] == len(perm):
                st["epoch"], st["cursor"] = st["epoch"] + 1, 0
        return out

    def plan(self, n_old: int, update: int) -> list:
        """-> [(task, phase, row), ...] of length n_old (0 if there are no old tasks)."""
        if not self.tasks or n_old == 0:
            return []
        out = []
        for t, n_t in zip(self.tasks, split(n_old, len(self.tasks), update)):
            phases = [p for p in PHASES if f"{t}|{p}" in self.cells]
            for p, n_p in zip(phases, split(n_t, len(phases), update)):
                out += [(t, p, r) for r in self._draw(f"{t}|{p}", n_p)]
        return out

    def get_state(self) -> dict:
        return {c: dict(v) for c, v in self.state.items()}
