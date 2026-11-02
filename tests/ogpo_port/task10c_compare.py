"""Task 10C analysis (host only). Full gradients and student-state contributions compared separately; no pass/fail on the comparison."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import numpy as np

B, D = Path("outputs/ogpo_port/task10b"), Path("outputs/ogpo_port/task10c")
SEEDS = (10000, 11000, 11001, 11002)
REUSE = {10000: "p2_det_1.npz", 11000: "p3_a_11000.npz", 11001: "p3_a_11001.npz", 11002: "p3_a_11002.npz"}
GROUPS = (("action_expert", r"^PaliGemma/llm/.*_1(/|$)"), ("action_heads", r"^(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/"))
PAIRS = (("canonical_vs_reference", "C3", "C4A"), ("precision_bf16_eager_vs_fp32_eager_triton_off", "C1", "C4A"), ("transformation_bf16_jit_vs_eager_triton_off", "C2", "C1"),
         ("backend_bf16_jit_triton_on_vs_off", "C3", "C2"), ("bf16_jit_triton_off_vs_reference", "C2", "C4A"), ("reference_cache_recomputed_vs_promoted", "C4B", "C4A"))


def load(cfg, seed):
    z = np.load(D / "tmp" / f"{cfg}_{seed}.npz")
    d = {k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("d__")}
    if cfg == "C3":
        r = np.load(B / REUSE[seed])
        f = {k[3:]: r[k].astype(np.float64) for k in r.files if k.startswith("g__")}
    else:
        f = {k[3:]: z[k].astype(np.float64) for k in z.files if k.startswith("f__")}
    return f, d, {k: f[k] - d[k] for k in f}


def stats(a, b, keys):
    """Metrics of a relative to the reference b."""
    ab = sum(float(np.sum(a[k] * b[k])) for k in keys)
    na = sum(float(np.sum(a[k] ** 2)) for k in keys) ** 0.5
    nb = sum(float(np.sum(b[k] ** 2)) for k in keys) ** 0.5
    dd = sum(float(np.sum((a[k] - b[k]) ** 2)) for k in keys) ** 0.5
    return {"cosine": ab / (na * nb) if na * nb else None, "rel_l2_diff": dd / nb if nb else None, "norm_a": na, "norm_b": nb, "norm_ratio": na / nb if nb else None}


def metrics(a, b):
    keys = sorted(a)
    out = {"overall": stats(a, b, keys)}
    for n, rx in GROUPS:
        out[n] = stats(a, b, [k for k in keys if re.search(rx, k)])
    out["per_leaf"] = {k: stats(a, b, [k]) for k in keys}
    return out


def primal(cfg, seed):
    z = np.load(D / f"primal_{cfg}_{seed}.npz")
    return {k: z[k].astype(np.float64) for k in z.files}


def primal_diff(a, b):
    da, db = a["vs"][0] - a["vt"][0], b["vs"][0] - b["vt"][0]
    return {"objective_a": float(a["objective"]), "objective_b": float(b["objective"]), "objective_diff": float(a["objective"] - b["objective"]), "objective_rel": float(abs(a["objective"] - b["objective"]) / b["objective"]),
            "state_max_abs_per_k": np.abs(a["chain"][0] - b["chain"][0]).max(axis=(1, 2)).tolist(), "state_rel_l2_per_k": (np.linalg.norm(a["chain"][0] - b["chain"][0], axis=(1, 2)) / np.linalg.norm(b["chain"][0], axis=(1, 2))).tolist(),
            "vs_rel_l2_per_k": (np.linalg.norm(a["vs"][0] - b["vs"][0], axis=(1, 2)) / np.linalg.norm(b["vs"][0], axis=(1, 2))).tolist(), "vt_rel_l2_per_k": (np.linalg.norm(a["vt"][0] - b["vt"][0], axis=(1, 2)) / np.linalg.norm(b["vt"][0], axis=(1, 2))).tolist(),
            "d_rel_l2_per_k": (np.linalg.norm(da - db, axis=(1, 2)) / np.linalg.norm(db, axis=(1, 2))).tolist(), "kl_abs_per_k": np.abs(a["kl"][0] - b["kl"][0]).tolist(), "kl_rel_per_k": (np.abs(a["kl"][0] - b["kl"][0]) / b["kl"][0]).tolist()}


def main():
    out = {"threshold": None, "pairs": {}, "seed_context": {}}
    grads = {}
    for name, a, b in PAIRS:
        out["pairs"][name] = {"a": a, "b": b, "seeds": {}}
    for seed in SEEDS:
        grads = {c: load(c, seed) for c in ("C1", "C2", "C3", "C4A", "C4B")}
        for name, a, b in PAIRS:
            fa, da, sa = grads[a]
            fb, db, sb = grads[b]
            row = {"full": metrics(fa, fb), "detached": metrics(da, db), "state": metrics(sa, sb), "primal": primal_diff(primal(a, seed), primal(b, seed))}
            out["pairs"][name]["seeds"][str(seed)] = row
            print(f"[{seed} {name}] full cos {row['full']['overall']['cosine']:.5f} rel {row['full']['overall']['rel_l2_diff']:.4f} | state cos {row['state']['overall']['cosine']:.4f} rel {row['state']['overall']['rel_l2_diff']:.4f} | "
                  f"exp full {row['full']['action_expert']['cosine']:.5f}/{row['full']['action_expert']['rel_l2_diff']:.4f} state {row['state']['action_expert']['cosine']:.4f}/{row['state']['action_expert']['rel_l2_diff']:.4f} | "
                  f"heads full {row['full']['action_heads']['cosine']:.5f}/{row['full']['action_heads']['rel_l2_diff']:.4f} state {row['state']['action_heads']['cosine']:.4f}/{row['state']['action_heads']['rel_l2_diff']:.4f} | "
                  f"norms full {row['full']['overall']['norm_a']:.1f}/{row['full']['overall']['norm_b']:.1f} state {row['state']['overall']['norm_a']:.1f}/{row['state']['overall']['norm_b']:.1f} | obj {row['primal']['objective_a']:.4f}/{row['primal']['objective_b']:.4f} | x10 {row['primal']['state_max_abs_per_k'][10]:.2e} vS0 {row['primal']['vs_rel_l2_per_k'][0]:.2e} vT0 {row['primal']['vt_rel_l2_per_k'][0]:.2e} d0 {row['primal']['d_rel_l2_per_k'][0]:.3f}", flush=True)
        fr, _, sr = grads["C4A"]
        out["seed_context"][str(seed)] = {"state_over_full_norm_reference": stats(sr, fr, sorted(fr))["norm_ratio"]}
        if seed == SEEDS[0]:
            first_ref, first_state = fr, sr
        else:
            out["seed_context"][str(seed)]["reference_full_vs_seed10000"] = metrics(fr, first_ref)["overall"]
            out["seed_context"][str(seed)]["reference_state_vs_seed10000"] = metrics(sr, first_state)["overall"]
        print(f"[{seed} context]", json.dumps(out["seed_context"][str(seed)]), flush=True)
    print("[mean over seeds]", flush=True)
    summary = {}
    for name in out["pairs"]:
        rows = out["pairs"][name]["seeds"]
        summary[name] = {g: {"cosine_mean": float(np.mean([r[g]["overall"]["cosine"] for r in rows.values()])), "cosine_min": float(np.min([r[g]["overall"]["cosine"] for r in rows.values()])),
                             "rel_mean": float(np.mean([r[g]["overall"]["rel_l2_diff"] for r in rows.values()])), "rel_max": float(np.max([r[g]["overall"]["rel_l2_diff"] for r in rows.values()]))} for g in ("full", "detached", "state")}
        summary[name]["objective_rel_mean"] = float(np.mean([r["primal"]["objective_rel"] for r in rows.values()]))
        print(name, json.dumps(summary[name]), flush=True)
    out["summary"] = summary
    (D / "comparison.json").write_text(json.dumps(out, indent=1))
    keep = D / "reference_gradients"
    keep.mkdir(exist_ok=True)
    for seed in SEEDS:                             # keep only the FP32 reference; the BF16 arrays are recomputed in seconds
        shutil.move(str(D / "tmp" / f"C4A_{seed}.npz"), str(keep / f"C4A_{seed}.npz"))
    shutil.rmtree(D / "tmp")
    print("[end] kept FP32 reference gradients; removed temporary BF16 and C4B arrays", flush=True)


if __name__ == "__main__":
    main()
