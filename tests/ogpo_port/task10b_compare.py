"""Task 10B analysis of saved outputs (host only, no GPU). Run after the task10b.py phases."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

D = Path("outputs/ogpo_port/task10b")
GROUPS = (("action_expert", r"^PaliGemma/llm/.*_1(/|$)"), ("action_heads", r"^(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/"))


def load_g(path):
    z = np.load(path)
    return {k[3:]: z[k] for k in z.files if k.startswith("g__")}


def dot(a, b, keys=None):
    return sum(float(np.sum(a[k].astype(np.float64) * b[k].astype(np.float64))) for k in (keys or a))


def stats(a, b, keys=None):
    keys = keys or sorted(a)
    na, nb = dot(a, a, keys) ** 0.5, dot(b, b, keys) ** 0.5
    d = sum(float(np.sum((a[k].astype(np.float64) - b[k].astype(np.float64)) ** 2)) for k in keys) ** 0.5
    return {"cosine": dot(a, b, keys) / (na * nb), "rel_diff": d / na, "rel_norm_diff": abs(na - nb) / na, "norm_a": na, "norm_b": nb}


def by_group(a, b):
    out = {"overall": stats(a, b)}
    for name, rx in GROUPS:
        out[name] = stats(a, b, sorted(k for k in a if re.search(rx, k)))
    return out


def p2(tags, label):
    zs = {t: np.load(D / f"p2_{t}.npz") for t in tags}
    ref = tags[0]
    out = {}
    for t in tags[1:]:
        row = {"objective": float(zs[t]["p__objective"]), "objective_ref": float(zs[ref]["p__objective"])}
        for k in ("chain", "vs", "vt", "kl"):
            d = np.abs(zs[t][f"p__{k}"].astype(np.float64) - zs[ref][f"p__{k}"].astype(np.float64))
            row[f"{k}_max_abs"] = float(d.max())
            row[f"{k}_bitwise"] = bool(d.max() == 0)
        row["state_max_abs_per_k"] = np.abs(zs[t]["p__chain"].astype(np.float64) - zs[ref]["p__chain"].astype(np.float64))[0].max(axis=(1, 2)).tolist()
        row["kl_abs_per_k"] = np.abs(zs[t]["p__kl"].astype(np.float64) - zs[ref]["p__kl"].astype(np.float64))[0].tolist()
        ga, gb = {k[3:]: zs[ref][k] for k in zs[ref].files if k.startswith("g__")}, {k[3:]: zs[t][k] for k in zs[t].files if k.startswith("g__")}
        row["gradient_bitwise"] = all(np.array_equal(ga[k], gb[k]) for k in ga)
        row["gradient"] = by_group(ga, gb)
        out[t] = row
    print(f"[{label}]", json.dumps(out), flush=True)
    return out


def p3():
    seeds = sorted(int(m.group(1)) for p in D.glob("p3_a_*.npz") if (m := re.fullmatch(r"p3_a_(\d+)\.npz", p.name)))
    res = {"seeds": seeds}
    g = {s: load_g(D / f"p3_a_{s}.npz") for s in seeds}
    keys = sorted(g[seeds[0]])
    # execution variation: same trajectory
    res["cross_process_same_trajectory"] = {}
    for s in seeds:
        b = load_g(D / f"p3_b_{s}.npz")
        res["cross_process_same_trajectory"][str(s)] = {"bitwise": all(np.array_equal(g[s][k], b[k]) for k in keys), **stats(g[s], b)}
        del b
    res["program_jit_vs_eager_same_trajectory"] = {}
    ge = {}
    for s in seeds:
        ge[s] = load_g(D / f"p3_a_{s}_eager.npz")
        res["program_jit_vs_eager_same_trajectory"][str(s)] = by_group(g[s], ge[s])
    # trajectory variation: same programme
    mean = {k: sum(g[s][k].astype(np.float64) for s in seeds) / len(seeds) for k in keys}
    res["sample_vs_aggregate"] = {str(s): by_group(g[s], mean) for s in seeds}
    res["sample_vs_leave_one_out"] = {}
    for s in seeds:
        loo = {k: (mean[k] * len(seeds) - g[s][k].astype(np.float64)) / (len(seeds) - 1) for k in keys}
        res["sample_vs_leave_one_out"][str(s)] = by_group(g[s], loo)
    pair = []
    for i, a in enumerate(seeds):
        for b in seeds[i + 1:]:
            pair.append(by_group(g[a], g[b]))
    res["pairwise"] = {n: {"cosine_min": min(p[n]["cosine"] for p in pair), "cosine_mean": float(np.mean([p[n]["cosine"] for p in pair])), "cosine_max": max(p[n]["cosine"] for p in pair),
                           "rel_diff_mean": float(np.mean([p[n]["rel_diff"] for p in pair]))} for n in ("overall", "action_expert", "action_heads")}
    mean_e = {k: sum(ge[s][k].astype(np.float64) for s in seeds) / len(seeds) for k in keys}
    res["aggregate_jit_vs_aggregate_eager"] = by_group(mean, mean_e)
    res["noise_decomposition"] = {
        "trajectory_rel_diff_mean_pairwise": res["pairwise"]["overall"]["rel_diff_mean"],
        "execution_rel_diff_mean_jit_vs_eager": float(np.mean([v["overall"]["rel_diff"] for v in res["program_jit_vs_eager_same_trajectory"].values()])),
        "execution_rel_diff_cross_process_mean": float(np.mean([v["rel_diff"] for v in res["cross_process_same_trajectory"].values()]))}
    print("[p3 analysis]", json.dumps({k: v for k, v in res.items() if k not in ("sample_vs_leave_one_out", "program_jit_vs_eager_same_trajectory", "cross_process_same_trajectory", "sample_vs_aggregate")}), flush=True)
    print("[p3 cross-process bitwise]", {s: v["bitwise"] for s, v in res["cross_process_same_trajectory"].items()}, flush=True)
    print("[p3 sample vs aggregate cos]", {s: round(v["overall"]["cosine"], 5) for s, v in res["sample_vs_aggregate"].items()}, flush=True)
    print("[p3 sample vs leave-one-out cos]", {s: round(v["overall"]["cosine"], 5) for s, v in res["sample_vs_leave_one_out"].items()}, flush=True)
    print("[p3 jit vs eager cos, rel]", {s: (round(v["overall"]["cosine"], 5), round(v["overall"]["rel_diff"], 4)) for s, v in res["program_jit_vs_eager_same_trajectory"].items()}, flush=True)
    return res


def get(z, m):
    return {k: z[f"{m}__{k}"].astype(np.float64) for k in ("chain", "vs", "vt", "kl")}


def decompose(a, r):
    """Change a relative to reference r, per step: delta d = (vS - vT)_a - (vS - vT)_r, delta K = w (2 d.delta_d + |delta_d|^2)."""
    dr, da = r["vs"][0] - r["vt"][0], a["vs"][0] - a["vt"][0]
    w = r["kl"][0] / np.sum(dr**2, axis=(1, 2))
    e = da - dr
    cross, quad = 2 * w * np.sum(dr * e, axis=(1, 2)), w * np.sum(e**2, axis=(1, 2))
    actual = a["kl"][0] - r["kl"][0]
    return {"dK_total": float(actual.sum()), "cross_total": float(cross.sum()), "quad_total": float(quad.sum()), "identity_residual": float((actual.sum() - cross.sum() - quad.sum())),
            "dvS_rel_l2_per_k": (np.linalg.norm(a["vs"][0] - r["vs"][0], axis=(1, 2)) / np.linalg.norm(r["vs"][0], axis=(1, 2))).tolist(),
            "dvT_rel_l2_per_k": (np.linalg.norm(a["vt"][0] - r["vt"][0], axis=(1, 2)) / np.linalg.norm(r["vt"][0], axis=(1, 2))).tolist(),
            "dx_max_abs_per_k": np.abs(a["chain"][0] - r["chain"][0]).max(axis=(1, 2)).tolist(),
            "dd_rel_l2_per_k": (np.linalg.norm(e, axis=(1, 2)) / np.linalg.norm(dr, axis=(1, 2))).tolist(), "dK_per_k": actual.tolist(), "cross_per_k": cross.tolist(), "quad_per_k": quad.tolist()}


def p5():
    z5 = np.load(D / "p5_canonical.npz")
    r = get(z5, "jit_vag")
    cfg = {m: get(z5, m) for m in ("jit_forward", "eager_forward", "eager_vag", "jit_forward_on_ref_chain", "eager_forward_on_ref_chain")}
    for c in "AB":
        z = np.load(D / f"p4_{c}.npz")
        for m in ("eager_forward", "eager_vag", "jit_forward", "eager_forward_on_ref_chain"):
            cfg[f"fp32{c}_{m}"] = get(z, m)
    res = {"vs_canonical": {m: decompose(a, r) for m, a in cfg.items()}}
    res["cache_precision_A_vs_B_same_mode"] = decompose(cfg["fp32B_eager_forward"], cfg["fp32A_eager_forward"])
    res["cache_precision_on_ref_chain"] = decompose(cfg["fp32B_eager_forward_on_ref_chain"], cfg["fp32A_eager_forward_on_ref_chain"])
    res["transformer_precision_bf16_vs_fp32A_eager"] = decompose(cfg["fp32A_eager_forward"], cfg["eager_forward"])
    res["transformer_precision_on_ref_chain"] = decompose(cfg["fp32A_eager_forward_on_ref_chain"], cfg["eager_forward_on_ref_chain"])
    res["bf16_execution_eager_vs_jit_forward"] = decompose(cfg["eager_forward"], cfg["jit_forward"])
    res["fp32A_execution_jit_vs_eager"] = decompose(cfg["fp32A_jit_forward"], cfg["fp32A_eager_forward"])
    res["fp32B_execution_jit_vs_eager"] = decompose(cfg["fp32B_jit_forward"], cfg["fp32B_eager_forward"])
    K = {m: float(a["kl"][0].sum()) for m, a in cfg.items()}
    K["canonical"] = float(r["kl"][0].sum())
    res["K"] = K
    res["state_vs_velocity_split"] = {name: {"velocity_at_ref_states": K[b] - K["canonical"], "state_change": K[a] - K[b]} for name, a, b in (
        ("bf16_eager", "eager_forward", "eager_forward_on_ref_chain"), ("bf16_jit_forward", "jit_forward", "jit_forward_on_ref_chain"),
        ("fp32A_eager", "fp32A_eager_forward", "fp32A_eager_forward_on_ref_chain"), ("fp32B_eager", "fp32B_eager_forward", "fp32B_eager_forward_on_ref_chain"))}
    print("[p5 K]", json.dumps(K), flush=True)
    print("[p5 split]", json.dumps(res["state_vs_velocity_split"]), flush=True)
    for n, v in list(res["vs_canonical"].items()) + [(k, res[k]) for k in res if k.startswith(("cache", "transformer", "bf16_exec", "fp32"))]:
        print(f"[p5 {n}] dK {v['dK_total']:+.5f} cross {v['cross_total']:+.5f} quad {v['quad_total']:+.5f} resid {v['identity_residual']:+.1e} | dd/d k0 {v['dd_rel_l2_per_k'][0]:.2e} k9 {v['dd_rel_l2_per_k'][9]:.2e} | dvS k0 {v['dvS_rel_l2_per_k'][0]:.2e} dvT k0 {v['dvT_rel_l2_per_k'][0]:.2e} | dx10 {v['dx_max_abs_per_k'][10]:.2e}", flush=True)
    return res


def main():
    res = {"p2_det": p2(["det_1", "det_2", "det_3", "det_4"], "p2 fresh processes, deterministic flags"), "p2_default": p2(["det_1", "default_1", "default_2"], "p2 default flags vs det_1"), "p3": p3(), "p5": p5()}
    (D / "analysis.json").write_text(json.dumps(res, indent=1))
    print("[end]", flush=True)


if __name__ == "__main__":
    main()
