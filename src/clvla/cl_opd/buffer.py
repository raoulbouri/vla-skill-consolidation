"""Fixed-total replay buffer for old tasks, nested by construction.

Each finished task's candidates (every policy-query step of its successful teacher-driven episodes) are ranked ONCE by
`priority_order`: phases in rotation; inside a phase, episodes in rotation (seeded order), and inside an episode the step
whose within-phase time is nearest the next point of a per-episode rotated van der Corput sequence, at least `min_gap`
query steps from the episode's earlier picks. Steps that break the gap rule go last. Any prefix is therefore balanced
across phases (counts within 1 while every phase has candidates), across episodes and across time; a phase that runs out
simply drops from the rotation, so its leftover goes to the task's other phases.

`allocate` splits the total M equally across the finished tasks (remainder to the earliest tasks; a task that has fewer
ranked candidates than its share passes the rest on). Shares only shrink as tasks are added, so admitting task k and
cutting every earlier task to its new share is a truncation of each stored list: nothing is recomputed, smaller buffers
are subsets of larger ones, and the stored total never exceeds M.

On disk, one directory per task: fixed-shape arrays (`.npy`, memory-mapped on read), `manifest.jsonl` (one row per
stored transition, in rank order) and `meta.json` (task, prompt, teacher hash, count, content hash).
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path

import numpy as np

from clvla.cl_opd.phases import PHASES


def van_der_corput(i: int) -> float:
    x, d = 0.0, 0.5
    while i:
        x, i, d = x + d * (i & 1), i >> 1, d / 2
    return x


def priority_order(cands: list, seed: int, min_gap: int = 2) -> list:
    """cands: dicts with episode, step (policy-query index), phase, t (within-phase time in [0, 1]). Returns indices into cands."""
    rng = np.random.default_rng(seed)
    phase_lists = {}
    for p in PHASES:
        ids = [i for i, c in enumerate(cands) if c["phase"] == p]
        eps = sorted({cands[i]["episode"] for i in ids})
        eps = [eps[j] for j in rng.permutation(len(eps))]
        per = {e: [i for i in ids if cands[i]["episode"] == e] for e in eps}
        shift = {e: float(rng.random()) for e in eps}
        picked = {e: [] for e in eps}
        k = {e: 0 for e in eps}
        good, late = [], []
        while any(per.values()):
            for e in eps:
                if not per[e]:
                    continue
                target = (van_der_corput(k[e] + 1) + shift[e]) % 1.0
                k[e] += 1
                ok = [i for i in per[e] if all(abs(cands[i]["step"] - cands[j]["step"]) >= min_gap for j in picked[e])]
                if not ok:
                    late += per[e]
                    per[e] = []
                    continue
                i = min(ok, key=lambda i_: (abs(cands[i_]["t"] - target), cands[i_]["step"]))
                good.append(i)
                picked[e].append(i)
                per[e].remove(i)
        phase_lists[p] = good + [late[j] for j in rng.permutation(len(late))]
    order, r = [], 0
    while any(phase_lists.values()):
        for p in PHASES:
            if len(phase_lists[p]) > r:
                order.append(phase_lists[p][r])
        r += 1
        if all(len(v) <= r for v in phase_lists.values()):
            break
    return order


def allocate(total: int, available: dict) -> dict:
    """Equal shares of `total` over tasks (dict order = task order), remainder to the earliest; a task's unused share is
    passed on to the others in order."""
    tasks = list(available)
    share = {t: 0 for t in tasks}
    left = total
    while left > 0:
        open_ = [t for t in tasks if share[t] < available[t]]
        if not open_:
            break
        base, rem = divmod(left, len(open_))
        given = 0
        for i, t in enumerate(open_):
            g = min(base + (1 if i < rem else 0), available[t] - share[t])
            share[t] += g
            given += g
        left -= given
        if given == 0:
            break
    return share


FIELDS = {"image": np.uint8, "wrist_image": np.uint8, "state": np.float32, "action": np.float32}
TRAINER_TO_BUFFER = {"approach": "approach", "precontact": "precontact", "contact": "contact", "opening": "manipulation", "late": "manipulation"}


def ranked_from_records(record_dir, seed: int, min_gap: int = 2):
    """Episodes written by opd_lora_train.py --record-dir -> (rows, arrays, meta) in priority order, ready for admit()."""
    from clvla.cl_opd.phases import within_phase_time

    files = sorted(Path(record_dir).glob("*.npz"))
    assert files, f"no recorded teacher episodes in {record_dir}"
    cands, src, prompts, sources = [], [], set(), set()
    for fi, f in enumerate(files):
        z = np.load(f)
        ph = [TRAINER_TO_BUFFER[str(p)] for p in z["phase"]]
        t = within_phase_time(ph)
        prompts.add(str(z["prompt"]))
        sources.add(str(z["phase_source"]))
        for i in range(len(ph)):
            cands.append({"episode": f.stem, "step": int(z["step"][i]), "phase": ph[i], "t": float(t[i])})
            src.append((fi, i))
    assert len(prompts) == 1, f"several prompts in {record_dir}: {prompts}"
    order = priority_order(cands, seed, min_gap)
    cache, arrays = {}, {k: [] for k in ("image", "wrist_image", "state", "action")}
    for r in order:
        fi, i = src[r]
        if fi not in cache:
            z = np.load(files[fi])
            cache[fi] = {"image": z["image"], "wrist_image": z["wrist_image"], "state": z["state"], "action": z["physical"]}
        for k in arrays:
            arrays[k].append(cache[fi][k][i])
    rows = [{**cands[r], "rank": n} for n, r in enumerate(order)]
    meta = {"prompt": prompts.pop(), "phase_source": sorted(sources), "episodes": len(files), "seed": seed, "min_gap": min_gap}
    return rows, {k: np.stack(v) for k, v in arrays.items()}, meta


def content_hash(task_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(FIELDS):
        p = task_dir / f"{f}.npy"
        if p.exists():
            a = np.load(p, mmap_mode="r")
            h.update(f.encode() + str(a.dtype).encode() + str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
    h.update((task_dir / "manifest.jsonl").read_bytes())
    return h.hexdigest()


def _atomic_dir(final: Path, write):
    tmp = final.with_name(final.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    write(tmp)
    for f in tmp.iterdir():
        with open(f, "rb") as fh:
            os.fsync(fh.fileno())
    old = final.with_name(final.name + ".old")
    shutil.rmtree(old, ignore_errors=True)
    if final.exists():
        final.rename(old)
    tmp.rename(final)
    shutil.rmtree(old, ignore_errors=True)


def write_task(root: Path, task: str, rows: list, arrays: dict, meta: dict) -> dict:
    """rows (rank order, one per transition) and arrays {field: (N, ...)} -> root/task. Returns meta with count and hash."""
    n = len(rows)
    assert all(len(v) == n for v in arrays.values()), "arrays and rows differ in length"
    assert set(arrays) <= set(FIELDS), set(arrays) - set(FIELDS)

    def w(d):
        for f, v in arrays.items():
            np.save(d / f"{f}.npy", np.ascontiguousarray(np.asarray(v, FIELDS[f])))
        (d / "manifest.jsonl").write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))
        m = {**meta, "task": task, "count": n}
        m["content_sha"] = content_hash(d)
        (d / "meta.json").write_text(json.dumps(m, indent=1, sort_keys=True))

    _atomic_dir(Path(root) / task, w)
    return read_meta(root, task)


def read_meta(root: Path, task: str) -> dict:
    return json.loads((Path(root) / task / "meta.json").read_text())


def read_rows(root: Path, task: str) -> list:
    return [json.loads(l) for l in (Path(root) / task / "manifest.jsonl").read_text().splitlines()]


def load_arrays(root: Path, task: str) -> dict:
    d = Path(root) / task
    return {f: np.load(d / f"{f}.npy", mmap_mode="r") for f in FIELDS if (d / f"{f}.npy").exists()}


def truncate(root: Path, task: str, n: int) -> dict:
    """Keep the first n transitions (a prefix of the rank order)."""
    rows, arrs, meta = read_rows(root, task), load_arrays(root, task), read_meta(root, task)
    assert n <= len(rows), "a share can only shrink"
    if n == len(rows):
        return meta
    keep = {k: v for k, v in meta.items() if k not in ("count", "content_sha", "task")}
    return write_task(root, task, rows[:n], {f: np.asarray(a[:n]) for f, a in arrs.items()}, {**keep, "truncated_from": len(rows)})


class FixedTotalBuffer:
    """root/<task>/ per finished task plus root/buffer.json (total, task order, shares, hashes)."""

    def __init__(self, root, total: int):
        self.root, self.total = Path(root), int(total)
        self.root.mkdir(parents=True, exist_ok=True)
        f = self.root / "buffer.json"
        self.state = json.loads(f.read_text()) if f.exists() else {"total": self.total, "tasks": [], "shares": {}, "hashes": {}}
        assert self.state["total"] == self.total, "buffer was created with another total"

    def _save(self):
        tmp = self.root / "buffer.json.tmp"
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True))
        os.replace(tmp, self.root / "buffer.json")

    def admit(self, task: str, rows: list, arrays: dict, meta: dict) -> dict:
        """rows/arrays: the task's full ranked candidate list (rank order). Stores only its share and cuts the others."""
        assert task not in self.state["tasks"], f"{task} already admitted"
        tasks = self.state["tasks"] + [task]
        avail = {t: self.state["shares"][t] for t in self.state["tasks"]}
        avail[task] = len(rows)
        shares = allocate(self.total, avail)
        write_task(self.root, task, rows[: shares[task]], {f: np.asarray(v[: shares[task]]) for f, v in arrays.items()}, {**meta, "ranked_candidates": len(rows)})
        for t in self.state["tasks"]:
            truncate(self.root, t, shares[t])
        self.state.update(tasks=tasks, shares=shares, hashes={t: read_meta(self.root, t)["content_sha"] for t in tasks})
        self._save()
        return dict(shares)

    def verify(self) -> bool:
        return all(content_hash(self.root / t) == self.state["hashes"][t] == read_meta(self.root, t)["content_sha"] for t in self.state["tasks"])

    def stored(self) -> int:
        return sum(read_meta(self.root, t)["count"] for t in self.state["tasks"])

    def bytes_on_disk(self) -> int:
        return sum(f.stat().st_size for t in self.state["tasks"] for f in (self.root / t).iterdir())

    def phase_index(self) -> dict:
        """{task: {phase: [row index, ...]}} for the sampler."""
        out = {}
        for t in self.state["tasks"]:
            rows = read_rows(self.root, t)
            out[t] = {p: [i for i, r in enumerate(rows) if r["phase"] == p] for p in PHASES}
        return out
