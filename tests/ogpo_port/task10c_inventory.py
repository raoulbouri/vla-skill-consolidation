"""Task 10C, part A: inventory of outputs/ogpo_port/task10b/ from file headers and saved JSON; reads no gradient values."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np

D = Path("outputs/ogpo_port/task10b")
OUT = Path("outputs/ogpo_port/task10c")
OUT.mkdir(parents=True, exist_ok=True)


def header(path):
    arrays = {}
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            with z.open(name) as f:
                version = np.lib.format.read_magic(f)
                shape, _, dtype = (np.lib.format.read_array_header_1_0(f) if version == (1, 0) else np.lib.format.read_array_header_2_0(f))
            arrays[name[:-4]] = {"shape": list(shape), "dtype": str(dtype)}
    return arrays


rows = {}
for p in sorted(D.glob("*.npz")):
    a = header(p)
    grads = {k: v for k, v in a.items() if k.startswith("g__")}
    row = {"size_gib": round(p.stat().st_size / 2**30, 2), "n_gradient_leaves": len(grads), "gradient_dtypes": sorted({v["dtype"] for v in grads.values()}),
           "n_gradient_params": int(sum(int(np.prod(v["shape"])) for v in grads.values())), "primal_arrays": {k: v for k, v in a.items() if not k.startswith("g__") and ("chain" in k or "objective" in k)}}
    rows[p.name] = row
meta = {}
for p in sorted(D.glob("*.json")):
    if p.name == "analysis.json":
        continue
    d = json.loads(p.read_text())
    if not isinstance(d, dict) or "phase" not in d:   # audit outputs (jump.json, fd_audit.json) describe no run configuration
        continue
    meta[p.name] = {"xla_flags": d.get("env", {}).get("xla_flags"), "input_sha256_16": {k: v[:16] for k, v in d.get("input_sha256", {}).items()}, "noise_sha256": {k: str(v)[:16] for k, v in d.get("noise_sha256", {}).items()},
                    "bf16_param_sha_16": str(d.get("bf16_param_sha"))[:16] if d.get("bf16_param_sha") else None, "bf16_kv_sha_16": str(d.get("bf16_kv_sha"))[:16] if d.get("bf16_kv_sha") else None,
                    "control": d.get("control"), "seeds": sorted(d["samples"]) if "samples" in d else None, "seed_sha": {s: {"x0": r["x0_sha"], "eps": r["eps_sha"], "grad": r["grad_sha"][:16]} for s, r in d.get("samples", {}).items()}}
inv = {"files": rows, "runs": meta, "total_gib": round(sum(p.stat().st_size for p in D.glob("*")) / 2**30, 2),
       "paths": {"p2_det_*": "seed 10000, BF16 jit(value_and_grad), deterministic flags, Triton on, canonical cache", "p2_nt_*": "seed 10000, jit, Triton off, cache rebuilt under Triton off: NOT reusable",
                 "p2_default_*": "seed 10000, no flags: NOT reusable", "p3_a_<seed>": "BF16 jit full gradient, Triton on, canonical cache", "p3_a_<seed>_eager": "BF16 eager full gradient, Triton ON: not one of the four configurations",
                 "p3_b_<seed>": "bitwise repeat of p3_a", "p5_canonical, p4_*": "primals only, no gradients"},
       "reuse": {"configuration 3 full gradient": "p2_det_1 (seed 10000), p3_a_11000-11002, after a bitwise re-check", "detached gradients": "none saved anywhere: must be computed", "configurations 1, 2, 4": "no matching saved gradient: must be computed"}}
(OUT / "inventory.json").write_text(json.dumps(inv, indent=1))
print(json.dumps({"total_gib": inv["total_gib"], "n_npz": len(rows), "paths": inv["paths"], "reuse": inv["reuse"]}, indent=0))
for name in ("p2_det_1.npz", "p2_nt_1.npz", "p3_a_11000.npz", "p3_a_11000_eager.npz", "p3_b_11000.npz"):
    r = rows[name]
    print(name, r["size_gib"], "GiB", r["n_gradient_leaves"], "leaves", r["n_gradient_params"], "params", r["gradient_dtypes"], {k: v["shape"] for k, v in r["primal_arrays"].items()})
for name in ("p2_det_1.json", "p2_nt_1.json", "p2_default_1.json", "p3_a.json", "p4_A.json", "p4_A_notriton.json", "p4_B_notriton.json"):
    m = meta[name]
    print(name, m["xla_flags"], m["input_sha256_16"], m["noise_sha256"], m["bf16_param_sha_16"], m["bf16_kv_sha_16"], m["control"])
print("p3 seeds", json.dumps({s: meta["p3_a.json"]["seed_sha"][s] for s in ("11000", "11001", "11002")}))
