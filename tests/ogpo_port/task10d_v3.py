"""Task 10D-v3: finite-difference extrapolation. D(h) = D0 + c2 h^2, OLS over a fixed grid; no point selection. Task 10D and 10D-v2 stay recorded as FAILED.
  PHASE=fd V3_SEED=<seed> V3_TAG=<calibration|development>  CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy tests/ogpo_port/task10d_v3.py
Exact C4A: FP32 eager, Triton off, canonical BF16 caches promoted to FP32. Held-out seeds are not used here.

Check A: full objective along u_state: AD g_full.u_state vs FD of the full objective.
Check B: the state contribution itself: AD (g_full - g_detached).u_state vs D_full,FD - D_detached,FD, where the detached objective holds the base chain fixed as data at both +h and -h.
The diagnostic objective is summed in FP64 from the FP32 velocities (sum_k w_k ||v_S - v_T||^2, w_k float64); the FP32 objective is recorded alongside. Production is unchanged.
Each D(h) is corrected to first order for fp32 rounding of the step: D_adj(h) = D_FD(h) - (AD on the step actually taken - AD on the nominal step). Uncorrected fits are reported too.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import task10d as D  # sets XLA flags from PHASE before JAX is imported

import jax
import jax.numpy as jnp
import numpy as np

T = D.T
SEED, TAG = int(os.environ["V3_SEED"]), os.environ["V3_TAG"]
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ogpo_port" / "task10d" / "v3"
OUT.mkdir(parents=True, exist_ok=True)
GRID = (5e-4, 7e-4, 1e-3, 1.4e-3, 2e-3, 3e-3)
assert SEED not in {s for v in D.PRE["held_out"].values() for s in v}


def fit(hs, ys):
    h2 = np.asarray(hs) ** 2
    y = np.asarray(ys, float)
    A = np.stack([np.ones_like(h2), h2], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    res = y - A @ coef
    tot = float(np.sum((y - y.mean()) ** 2))
    loo = [float(np.linalg.lstsq(np.delete(A, i, 0), np.delete(y, i), rcond=None)[0][0]) for i in range(len(y))]
    cov = np.linalg.inv(A.T @ A) * float(np.sum(res**2)) / (len(y) - 2)
    return {"D0": float(coef[0]), "c2": float(coef[1]), "residuals": res.tolist(), "residual_rms": float(np.sqrt(np.mean(res**2))), "R2": 1.0 - float(np.sum(res**2)) / tot if tot else None, "D0_se": float(cov[0, 0] ** 0.5),
            "loo_D0": loo, "loo_max_variation": float(np.max(np.abs(np.asarray(loo) - coef[0]))), "residual_sign_changes": int(np.sum(np.sign(res[1:]) != np.sign(res[:-1])))}


def summarise(hs, adj, raw, ad, gn):
    a, r = fit(hs, adj), fit(hs, raw)
    return {"ad": ad, "fit": a, "fit_uncorrected": r, "rel_err": abs(a["D0"] - ad) / abs(ad), "scaled_err": abs(a["D0"] - ad) / gn, "rel_err_uncorrected": abs(r["D0"] - ad) / abs(ad), "loo_max_over_ad": a["loo_max_variation"] / abs(ad),
            "se_over_ad": a["D0_se"] / abs(ad), "residual_rms_over_ad": a["residual_rms"] / abs(ad)}


def main():
    H = T.helpers()
    flat = H["_flat"]
    res = {"tag": TAG, "seed": SEED, "grid": GRID, "xla_flags": os.environ["XLA_FLAGS"], "counted": TAG != "calibration"}
    prereg = OUT / "preregistration_v3.json"
    if TAG != "calibration":
        import hashlib

        res["prereg_v3_sha256"] = hashlib.sha256(prereg.read_bytes()).hexdigest()
    gd_s, gd_t, theta, rest_s, rest_t, obs_data = D.setup(H, res)
    f = T.canonical(gd_s, gd_t)
    I0, _, _ = D.make_I(H, rest_s, rest_t, obs_data["0"], 10000)
    (o0, a0), g0 = jax.value_and_grad(f, has_aux=True)(theta, I0)
    g0h = T.grad_host(H, g0)
    del g0
    ref = np.load(D.C / "reference_gradients" / "C4A_10000.npz")
    res["reproduces_C4A_seed10000"] = {"objective": float(o0) == float(np.load(D.C / "primal_C4A_10000.npz")["objective"]), "gradient": all(np.array_equal(g0h[k], ref[f"f__{k}"]) for k in g0h)}
    print("[reproduce C4A]", json.dumps(res["reproduces_C4A_seed10000"]), flush=True)
    assert all(res["reproduces_C4A_seed10000"].values())
    del g0h, a0, I0

    I, sx, se = D.make_I(H, rest_s, rest_t, obs_data["0"], SEED)
    res["noise"] = {"x0_sha": sx, "eps_sha": se}
    w = T.pw.pkl.kl_coefficients()["weights"]

    def diag(aux):
        vs, vt = np.asarray(aux["vs"], np.float64), np.asarray(aux["vt"], np.float64)
        return float(np.sum(w * np.sum((vs - vt) ** 2, axis=(2, 3))))

    keys = sorted(flat(theta))
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat(theta).items()}
    names = [by_id[id(l)] for l in leaves]
    (o, a), g = jax.value_and_grad(f, has_aux=True)(theta, I)
    gfull = {k: np.asarray(v, np.float64) for k, v in flat(g).items()}
    del g
    stored = jnp.asarray(a["chain"])
    gdet = {k: np.asarray(v, np.float64) for k, v in flat(jax.grad(lambda t: f(t, I, "detached", stored)[0])(theta)).items()}
    res["objective_fp32"], res["objective_fp64_diag"] = float(o), diag(a)

    def dot(x, y):
        return sum(float(np.sum(x[k] * y[k])) for k in x)

    gn = dot(gfull, gfull) ** 0.5
    n_total = sum(gfull[k].size for k in keys)
    gstate = {k: gfull[k] - gdet[k] for k in keys}
    r = np.random.default_rng(D.PRE["fd"]["random_seed"])          # the same independent directions as Task 10D and 10D-v2
    random_all = {k: r.standard_normal(gfull[k].shape) for k in keys}
    random_heads = {k: (r.standard_normal(gfull[k].shape) if not k.startswith("PaliGemma") else np.zeros_like(gfull[k])) for k in keys}
    r2 = np.random.default_rng(70001)
    random_kernel = {k: (r2.standard_normal(gfull[k].shape) if k == "action_out_proj/kernel" else np.zeros_like(gfull[k])) for k in keys}
    th = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in flat(theta).items()}

    def build(arrs):
        return jax.tree.unflatten(treedef, [jnp.asarray(arrs[k], jnp.float32) for k in names])

    def sweep(name, d, detached_too=False):
        n = dot(d, d) ** 0.5
        u = {k: d[k] / n for k in keys}
        ad_full, ad_det = dot(gfull, u), dot(gdet, u)
        rows = []
        for h in GRID:
            plus = {k: (th[k] + h * u[k]).astype(np.float32) for k in keys}
            minus = {k: (th[k] - h * u[k]).astype(np.float32) for k in keys}
            changed = sum(int(np.count_nonzero(plus[k] != minus[k])) for k in keys)
            eff = {k: (plus[k].astype(np.float64) - minus[k].astype(np.float64)) / (2.0 * h) for k in keys}
            eff_norm = dot(eff, eff) ** 0.5
            step_ad_full, step_ad_det = dot(gfull, eff), dot(gdet, eff)
            del eff
            bp, bm = build(plus), build(minus)
            del plus, minus
            op, ap = f(bp, I)
            om, am = f(bm, I)
            fp, fm = diag(ap), diag(am)
            row = {"h": h, "f_plus_fp64": fp, "f_minus_fp64": fm, "f_plus_fp32": float(op), "f_minus_fp32": float(om), "fd": (fp - fm) / (2 * h), "fd_fp32": (float(op) - float(om)) / (2 * h), "ad_on_step": step_ad_full,
                   "effective_step_norm_over_nominal": eff_norm, "entries_changed": changed, "fraction_changed": changed / n_total, "objective_ulp_as_derivative": float(np.spacing(np.float32(max(fp, fm)))) / (2 * h)}
            row["fd_adj"] = row["fd"] - (step_ad_full - ad_full)
            if detached_too:
                dp, dm = diag(f(bp, I, "detached", stored)[1]), diag(f(bm, I, "detached", stored)[1])
                row.update({"det_f_plus_fp64": dp, "det_f_minus_fp64": dm, "det_fd": (dp - dm) / (2 * h), "det_ad_on_step": step_ad_det})
                row["det_fd_adj"] = row["det_fd"] - (step_ad_det - ad_det)
            rows.append(row)
            print(f"[fd {name}] {json.dumps({k: v for k, v in row.items() if not k.startswith('f_')})}", flush=True)
        out = {"ad_full": ad_full, "ad_detached": ad_det, "cosine_with_gradient": ad_full / gn, "rows": rows,
               "check": summarise(GRID, [x["fd_adj"] for x in rows], [x["fd"] for x in rows], ad_full, gn)}
        if detached_too:
            out["check_B_detached"] = summarise(GRID, [x["det_fd_adj"] for x in rows], [x["det_fd"] for x in rows], ad_det, gn)
            out["check_B_state"] = summarise(GRID, [x["fd_adj"] - x["det_fd_adj"] for x in rows], [x["fd"] - x["det_fd"] for x in rows], ad_full - ad_det, gn)
            print(f"[check B {name}]", json.dumps({k: {kk: v for kk, v in out[k].items() if kk != 'fit_uncorrected'} for k in ("check_B_detached", "check_B_state")}), flush=True)
        print(f"[fit {name}]", json.dumps({kk: v for kk, v in out["check"].items() if kk != "fit_uncorrected"}), flush=True)
        return out

    res["directions"] = {"full": sweep("full", gfull), "state_u": sweep("state_u", gstate, detached_too=True), "random_heads": sweep("random_heads", random_heads),
                         "random_all": sweep("random_all", random_all), "supplementary_kernel": sweep("supplementary_kernel", random_kernel)}
    res["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}
    (OUT / f"run_fd_{TAG}.json").write_text(json.dumps(res, indent=1, default=str))
    print("[end]", json.dumps(res["memory"]), flush=True)


if __name__ == "__main__":
    main()
