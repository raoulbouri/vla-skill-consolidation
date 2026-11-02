"""Task 10D-v3 calibration of the extrapolation method on already-consumed seeds 10000 and 14000. Counts toward nothing: no preregistration, no seed 15000, no held-out seed.
  PHASE=fd CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy tests/ogpo_port/task10d_v3_cal.py
Exact C4A. M2: D0 + c2 h^2. M4: D0 + c2 h^2 + c4 h^4. OLS in the scaled variable (h / h_max)^2, coefficients reported in h. Block omission: contiguous blocks, not single points, because
neighbouring evaluations share base parameters, noise and nearly identical perturbations. No model is selected after the fact. "Provisional" flags use 0.1% of AD and decide nothing.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import task10d as D  # sets XLA flags from PHASE before JAX is imported

import jax
import jax.numpy as jnp
import numpy as np

T = D.T
SEEDS = (10000, 14000)
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ogpo_port" / "task10d" / "v3" / "calibration2"
OUT.mkdir(parents=True, exist_ok=True)
GRADIENT_GRID = np.geomspace(5e-4, 3e-3, 36)
GRIDS = {"random_heads": np.geomspace(0.05, 0.5, 12), "random_all": np.geomspace(1.0, 10.0, 12), "supplementary_kernel": np.geomspace(3e-3, 3e-2, 12)}
FLAG = 1e-3


def design(hs, order):
    z = (np.asarray(hs) / np.max(hs)) ** 2
    return np.stack([np.ones_like(z)] + [z**p for p in range(1, order + 1)], axis=1)


def ols(hs, y, order):
    return np.linalg.lstsq(design(hs, order), np.asarray(y, float), rcond=None)[0]


def model(hs, y, order, block):
    hs, y = np.asarray(hs), np.asarray(y, float)
    coef = ols(hs, y, order)
    res = y - design(hs, order) @ coef
    scale = float(np.max(hs))
    blocks = [np.arange(i, min(i + block, len(hs))) for i in range(0, len(hs), block)]
    omit = [float(ols(np.delete(hs, b), np.delete(y, b), order)[0]) for b in blocks]
    return {"D0": float(coef[0]), "c2": float(coef[1] / scale**2), "c4": float(coef[2] / scale**4) if order == 2 else None, "residuals": res.tolist(), "max_abs_residual": float(np.abs(res).max()), "residual_rms": float(np.sqrt(np.mean(res**2))),
            "residual_lag1_autocorr": float(np.corrcoef(res[:-1], res[1:])[0, 1]), "residual_block_means": [float(res[b].mean()) for b in blocks], "block_omitted_D0": omit, "block_max_deviation": float(np.max(np.abs(np.asarray(omit) - coef[0])))}


def analyse(hs, y, ad, gn, block):
    m2, m4 = model(hs, y, 1, block), model(hs, y, 2, block)
    out = {"ad": ad, "M2": m2, "M4": m4, "M2_vs_M4": abs(m2["D0"] - m4["D0"]), "M2_vs_M4_over_ad": abs(m2["D0"] - m4["D0"]) / abs(ad), "M2_rel_err": abs(m2["D0"] - ad) / abs(ad), "M4_rel_err": abs(m4["D0"] - ad) / abs(ad),
           "M2_scaled_err": abs(m2["D0"] - ad) / gn, "M4_scaled_err": abs(m4["D0"] - ad) / gn, "M2_block_over_ad": m2["block_max_deviation"] / abs(ad), "M4_block_over_ad": m4["block_max_deviation"] / abs(ad)}
    out["provisional"] = {"M2_within_0.1pct": out["M2_rel_err"] <= FLAG, "M4_within_0.1pct": out["M4_rel_err"] <= FLAG, "models_agree": out["M2_vs_M4_over_ad"] <= FLAG, "M2_block_stable": out["M2_block_over_ad"] <= FLAG, "M4_block_stable": out["M4_block_over_ad"] <= FLAG}
    return out


def main():
    H = T.helpers()
    flat = H["_flat"]
    res0 = {"xla_flags": os.environ["XLA_FLAGS"], "counted": False}
    gd_s, gd_t, theta, rest_s, rest_t, obs_data = D.setup(H, res0)
    f = T.canonical(gd_s, gd_t)
    w = T.pw.pkl.kl_coefficients()["weights"]

    def diag(aux):
        vs, vt = np.asarray(aux["vs"], np.float64), np.asarray(aux["vt"], np.float64)
        return float(np.sum(w * np.sum((vs - vt) ** 2, axis=(2, 3))))

    keys = sorted(flat(theta))
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat(theta).items()}
    names = [by_id[id(l)] for l in leaves]
    th = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in flat(theta).items()}
    n_total = sum(v.size for v in th.values())

    def build(arrs):
        return jax.tree.unflatten(treedef, [jnp.asarray(arrs[k], jnp.float32) for k in names])

    def dot(x, y):
        return sum(float(np.sum(x[k] * y[k])) for k in x)

    for seed in SEEDS:
        t_seed = time.perf_counter()
        res = {**res0, "seed": seed, "gradient_grid": GRADIENT_GRID.tolist(), "random_grids": {k: v.tolist() for k, v in GRIDS.items()}, "directions": {}}
        I, sx, se = D.make_I(H, rest_s, rest_t, obs_data["0"], seed)
        res["noise"] = {"x0_sha": sx, "eps_sha": se}
        (o, a), g = jax.value_and_grad(f, has_aux=True)(theta, I)
        gfull = {k: np.asarray(v, np.float64) for k, v in flat(g).items()}
        del g
        stored = jnp.asarray(a["chain"])
        gdet = {k: np.asarray(v, np.float64) for k, v in flat(jax.grad(lambda t: f(t, I, "detached", stored)[0])(theta)).items()}
        if seed == 10000:
            ref = np.load(D.C / "reference_gradients" / "C4A_10000.npz")
            res["reproduces_C4A"] = all(np.array_equal(gfull[k].astype(np.float32), ref[f"f__{k}"]) for k in keys)
            assert res["reproduces_C4A"]
        res["objective_fp32"], res["objective_fp64_diag"] = float(o), diag(a)
        gn = dot(gfull, gfull) ** 0.5
        r = np.random.default_rng(D.PRE["fd"]["random_seed"])      # the same independent directions as Task 10D and 10D-v2
        random_all = {k: r.standard_normal(gfull[k].shape) for k in keys}
        random_heads = {k: (r.standard_normal(gfull[k].shape) if not k.startswith("PaliGemma") else np.zeros_like(gfull[k])) for k in keys}
        r2 = np.random.default_rng(70001)
        random_kernel = {k: (r2.standard_normal(gfull[k].shape) if k == "action_out_proj/kernel" else np.zeros_like(gfull[k])) for k in keys}

        def sweep(name, d, grid, detached, block):
            t0 = time.perf_counter()
            n = dot(d, d) ** 0.5
            u = {k: d[k] / n for k in keys}
            ad_f, ad_d = dot(gfull, u), dot(gdet, u)
            rows = []
            for h in grid:
                h = float(h)
                plus = {k: (th[k] + h * u[k]).astype(np.float32) for k in keys}
                minus = {k: (th[k] - h * u[k]).astype(np.float32) for k in keys}
                changed = sum(int(np.count_nonzero(plus[k] != minus[k])) for k in keys)
                eff = {k: (plus[k].astype(np.float64) - minus[k].astype(np.float64)) / (2.0 * h) for k in keys}
                eff_norm, step_f, step_d = dot(eff, eff) ** 0.5, dot(gfull, eff), dot(gdet, eff)
                del eff
                bp, bm = build(plus), build(minus)
                del plus, minus
                op, ap = f(bp, I)
                om, am = f(bm, I)
                fp, fm = diag(ap), diag(am)
                row = {"h": h, "f_plus": fp, "f_minus": fm, "f_plus_fp32": float(op), "f_minus_fp32": float(om), "fd": (fp - fm) / (2 * h), "fd_adj": (fp - fm) / (2 * h) - (step_f - ad_f), "effective_step_norm": eff_norm, "fraction_changed": changed / n_total}
                if detached:
                    dp, dm = diag(f(bp, I, "detached", stored)[1]), diag(f(bm, I, "detached", stored)[1])
                    row.update({"det_f_plus": dp, "det_f_minus": dm, "det_fd": (dp - dm) / (2 * h), "det_fd_adj": (dp - dm) / (2 * h) - (step_d - ad_d)})
                rows.append(row)
            hs = [x["h"] for x in rows]
            out = {"cosine_with_gradient": ad_f / gn, "rows": rows, "seconds": time.perf_counter() - t0, "check": analyse(hs, [x["fd_adj"] for x in rows], ad_f, gn, block), "check_uncorrected": analyse(hs, [x["fd"] for x in rows], ad_f, gn, block)}
            if detached:
                out["detached"] = analyse(hs, [x["det_fd_adj"] for x in rows], ad_d, gn, block)
                out["state"] = analyse(hs, [x["fd_adj"] - x["det_fd_adj"] for x in rows], ad_f - ad_d, gn, block)
            res["directions"][name] = out
            (OUT / f"seed_{seed}.json").write_text(json.dumps(res, indent=1, default=str))
            print(f"[{seed} {name}] {out['seconds']:.0f} s", flush=True)
            for label in ("check", "detached", "state"):
                if label in out:
                    c = out[label]
                    print(f"[{seed} {name} {label}] AD {c['ad']:.6g} M2 {c['M2']['D0']:.6g} ({c['M2_rel_err']:.3e}) M4 {c['M4']['D0']:.6g} ({c['M4_rel_err']:.3e}) |M2-M4|/AD {c['M2_vs_M4_over_ad']:.2e} block/AD M2 {c['M2_block_over_ad']:.2e} M4 {c['M4_block_over_ad']:.2e} "
                          f"scaled {c['M2_scaled_err']:.1e}/{c['M4_scaled_err']:.1e} lag1 {c['M2']['residual_lag1_autocorr']:+.2f}/{c['M4']['residual_lag1_autocorr']:+.2f} rms/AD {c['M2']['residual_rms'] / abs(c['ad']):.2e} max/AD {c['M2']['max_abs_residual'] / abs(c['ad']):.2e}", flush=True)

        sweep("full", gfull, GRADIENT_GRID, False, 6)
        sweep("state_u", {k: gfull[k] - gdet[k] for k in keys}, GRADIENT_GRID, True, 6)
        sweep("random_heads", random_heads, GRIDS["random_heads"], False, 3)
        sweep("random_all", random_all, GRIDS["random_all"], False, 3)
        sweep("supplementary_kernel", random_kernel, GRIDS["supplementary_kernel"], False, 3)
        res["seed_seconds"] = time.perf_counter() - t_seed
        res["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}
        (OUT / f"seed_{seed}.json").write_text(json.dumps(res, indent=1, default=str))
        print(f"[{seed} end] {res['seed_seconds']:.0f} s {json.dumps(res['memory'])}", flush=True)


if __name__ == "__main__":
    main()
