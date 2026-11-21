"""Exact-resume check for the continual-learning trainer: the same short run of curriculum task K, once straight through
and once stopped and resumed, must end in the same training state (master, optimizer, replay, old-task sampler).
  scripts/py-policy scripts/cl_opd_resume_check.py --config C --run R --task 1 --out D [--warm 2 --stop 3 --updates 4] [--gpu G]
Uses R's buffer and task K-1 final master (task K-1 must be DONE). Updates 1..warm are teacher-driven, the rest student-
driven with the current-task replay buffer, so the stop lands after a replay-buffer update. Writes D/resume_check.json.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import cl_opd_run as D  # noqa: E402
from clvla.cl_opd import config as C  # noqa: E402


def setarg(cmd, flag, value):
    cmd = list(cmd)
    i = cmd.index(flag)
    cmd[i + 1] = str(value)
    return cmd


def digest(state: Path) -> dict:
    out = {"update": json.loads((state / "state.json").read_text())["update"], "master_sha": json.loads((state / "state.json").read_text())["master_sha"]}
    for f in ("opt_state.npz", "replay.pkl", "old_sampler.json"):
        if (state / f).exists():
            if f.endswith(".npz"):
                import numpy as np

                z = np.load(state / f)
                out[f] = hashlib.sha256(b"".join(z[k].tobytes() for k in sorted(z.files))).hexdigest()
            elif f.endswith(".pkl"):
                import pickle

                r = pickle.load(open(state / f, "rb"))
                h = hashlib.sha256(json.dumps({"seen": r["seen"], "rng": str(r["rng"]), "sizes": {s: {p: len(v) for p, v in d.items()} for s, d in r["pool"].items()}}, sort_keys=True).encode())
                for s in sorted(r["pool"]):
                    for p in sorted(r["pool"][s]):
                        for req in r["pool"][s][p]:
                            h.update(req["observation/state"].tobytes())
                out[f] = h.hexdigest()
            else:
                out[f] = hashlib.sha256((state / f).read_bytes()).hexdigest()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", type=int, default=1)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warm", type=int, default=2)
    ap.add_argument("--stop", type=int, default=3)
    ap.add_argument("--updates", type=int, default=4)
    ap.add_argument("--gpu", type=int, default=1)
    a = ap.parse_args()
    c = C.load(a.config)
    run, out = Path(a.run).resolve(), Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = D.trainer_cmd(c, run, a.task, 2, resume=False)
    base = setarg(setarg(base, "--updates", a.updates), "--teacher-control-updates", a.warm)
    base = setarg(base, "--state-every", 0)
    py = [str(ROOT / "scripts" / "py-policy"), "-W", "ignore"]
    env = D.child_env(a.gpu, str(c["per_task"].get("gpu_mem_fraction", 0.75)))
    res = {}
    for name, steps in (("straight", [[a.updates, False]]), ("resumed", [[a.stop, False], [a.updates, True]])):
        o = out / name
        for stop, resume in steps:
            cmd = setarg(setarg(base, "--out", o / "student"), "--record-dir", o / "teacher_episodes") + ["--stop-after", str(stop)] + (["--resume"] if resume else [])
            rc = D.run_cmd(py + cmd, o / "train.log", env, False)
            assert rc == 0, f"{name} stopped with rc {rc}; see {o / 'train.log'}"
        res[name] = digest(o / "student" / "state")
    res["identical"] = {k: res["straight"].get(k) == res["resumed"].get(k) for k in res["straight"]}
    res["all_identical"] = all(res["identical"].values())
    (out / "resume_check.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
