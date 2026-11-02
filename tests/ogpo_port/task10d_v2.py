"""Task 10D-v2: amended validation protocol. The original Task 10D, its preregistration and its failed result stay as they are.
  PHASE=fd|c3|c4a  CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy tests/ogpo_port/task10d_v2.py
Criteria come from task10d/preregistration_v2.json. Outputs go to task10d/v2/. The held-out phases reuse the Task 10D code and refuse to run unless the amended
Phase 1 result on the development seed says it passed.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import task10d as D  # sets XLA flags from PHASE before JAX is imported

import jax
import jax.numpy as jnp
import numpy as np

T = D.T
ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "outputs" / "ogpo_port" / "task10d"
PRE_FILE = BASE / "preregistration_v2.json"
PRE = json.loads(PRE_FILE.read_text())
ORIG = json.loads((BASE / "preregistration.json").read_text())
assert PRE["thresholds"] == ORIG["thresholds"] and PRE["held_out"] == ORIG["held_out"], "held-out criteria must not change"
OUT = BASE / "v2"
OUT.mkdir(parents=True, exist_ok=True)
D.OUT, D.TMP, D.PRE = OUT, OUT / "tmp", PRE
D.PAIRS = [(o, s) for o, seeds in PRE["held_out"].items() for s in seeds]
SMALL = D.SMALL


def phase_fd(H, res):
    flat = H["_flat"]
    rule = PRE["fd"]
    gd_s, gd_t, theta, rest_s, rest_t, obs_data = D.setup(H, res)
    f = T.canonical(gd_s, gd_t)
    # the configuration really is C4A: seed 10000 must reproduce Task 10C bitwise
    I0, _, _ = D.make_I(H, rest_s, rest_t, obs_data["0"], 10000)
    (o0, a0), g0 = jax.value_and_grad(f, has_aux=True)(theta, I0)
    g0h = T.grad_host(H, g0)
    del g0
    ref = np.load(D.C / "reference_gradients" / "C4A_10000.npz")
    prim = np.load(D.C / "primal_C4A_10000.npz")
    res["reproduces_C4A_seed10000"] = {"objective_bitwise": float(o0) == float(prim["objective"]), "gradient_bitwise": all(np.array_equal(g0h[k], ref[f"f__{k}"]) for k in g0h), "chain_bitwise": bool(np.array_equal(np.asarray(a0["chain"]), prim["chain"]))}
    print("[reproduce C4A]", json.dumps(res["reproduces_C4A_seed10000"]), flush=True)
    assert all(res["reproduces_C4A_seed10000"].values()), "C4A not reproduced"
    del g0h, a0, I0

    I, sx, se = D.make_I(H, rest_s, rest_t, obs_data["0"], rule["development_seed"])
    res["development"] = {"seed": rule["development_seed"], "observation": rule["observation"], "x0_sha": sx, "eps_sha": se}
    obj = lambda t: f(t, I)[0]  # noqa: E731
    keys = sorted(flat(theta))
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat(theta).items()}
    names = [by_id[id(l)] for l in leaves]
    (o, a), g = jax.value_and_grad(f, has_aux=True)(theta, I)
    gfull = {k: np.asarray(v, np.float64) for k, v in flat(g).items()}
    del g
    gdet = {k: np.asarray(v, np.float64) for k, v in flat(jax.grad(lambda t: f(t, I, "detached", jnp.asarray(a["chain"]))[0])(theta)).items()}
    res["development"]["objective"] = float(o)

    def dot(x, y):
        return sum(float(np.sum(x[k] * y[k])) for k in x)

    gn = dot(gfull, gfull) ** 0.5
    r = np.random.default_rng(rule["random_seed"])                  # same generator and order as the original Task 10D directions
    random_all = {k: r.standard_normal(gfull[k].shape) for k in keys}
    random_heads = {k: (r.standard_normal(gfull[k].shape) if not k.startswith("PaliGemma") else np.zeros_like(gfull[k])) for k in keys}
    r2 = np.random.default_rng(rule["kernel_random_seed"])
    random_kernel = {k: (r2.standard_normal(gfull[k].shape) if k == rule["supplementary_leaf"] else np.zeros_like(gfull[k])) for k in keys}
    dirs = {"full": (gfull, SMALL), "full_minus_detached": ({k: gfull[k] - gdet[k] for k in keys}, SMALL), "random_heads": (random_heads, SMALL + (0.3, 1.0)),
            "random_all": (random_all, SMALL[2:] + (0.3, 1.0, 3.0, 10.0)), "supplementary_random_action_out_proj_kernel": (random_kernel, SMALL + (0.3, 1.0))}
    th = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in flat(theta).items()}

    def build(arrs):
        return jax.tree.unflatten(treedef, [jnp.asarray(arrs[k], jnp.float32) for k in names])

    res["fd"] = {}
    for name, (d, steps) in dirs.items():
        n = dot(d, d) ** 0.5
        u = {k: d[k] / n for k in keys}
        rows = []
        for h in steps:
            plus = {k: (th[k] + h * u[k]).astype(np.float32) for k in keys}
            minus = {k: (th[k] - h * u[k]).astype(np.float32) for k in keys}
            fp, fm = float(obj(build(plus))), float(obj(build(minus)))
            ad = sum(float(np.sum(gfull[k] * (plus[k].astype(np.float64) - minus[k].astype(np.float64)))) for k in keys) / (2.0 * h)   # on the step actually taken
            del plus, minus
            fd = (fp - fm) / (2.0 * h)
            resolution = float(max(np.spacing(np.float32(fp)), np.spacing(np.float32(fm)))) / (2.0 * h)        # one fp32 unit of f, as a derivative
            rows.append({"h": h, "f_plus": fp, "f_minus": fm, "fd": fd, "ad_on_step": ad, "rel_err": abs(fd - ad) / abs(ad), "scaled_err": abs(fd - ad) / gn, "objective_ulp_as_derivative": resolution, "objective_ulp_over_ad": resolution / abs(ad)})
            print(f"[fd {name}] {json.dumps(rows[-1])}", flush=True)
        best_rel = min(rows, key=lambda x: x["rel_err"])
        best_scaled = min(rows, key=lambda x: x["scaled_err"])
        tail = rows[-3:]
        slope = float(np.polyfit(np.log10([x["h"] for x in tail]), np.log10([max(abs(x["fd"] - x["ad_on_step"]), 1e-30) for x in tail]), 1)[0])
        row = {"rows": rows, "ad_projection": dot(gfull, u), "cosine_with_gradient": dot(gfull, u) / gn, "best_rel_err": best_rel["rel_err"], "best_rel_h": best_rel["h"], "best_scaled_err": best_scaled["scaled_err"], "best_scaled_h": best_scaled["h"], "slope_largest_3_h": slope}
        if name in ("full", "random_heads"):
            row["criterion"] = f"rel <= {rule['rel_err_max']}"
            row["pass"] = best_rel["rel_err"] <= rule["rel_err_max"]
        elif name == "full_minus_detached":
            row["criterion"] = f"rel <= {rule['rel_err_max']} and slope in [{rule['slope_min']}, {rule['slope_max']}]"
            row["pass"] = best_rel["rel_err"] <= rule["rel_err_max"] and rule["slope_min"] <= slope <= rule["slope_max"]
        elif name == "random_all":
            row["criterion"] = f"scaled <= {rule['scaled_err_max_random_all']}"
            row["pass"] = best_scaled["scaled_err"] <= rule["scaled_err_max_random_all"]
        else:
            row["criterion"] = "supplementary: report only"
            row["pass"] = None
        res["fd"][name] = row
        print(f"[fd summary {name}] {json.dumps({k: v for k, v in row.items() if k != 'rows'})}", flush=True)
    res["phase1_pass"] = all(v["pass"] for v in res["fd"].values() if v["pass"] is not None)


def main():
    H = T.helpers()
    res = {"phase": D.PHASE, "xla_flags": os.environ["XLA_FLAGS"], "prereg_v2_sha256": hashlib.sha256(PRE_FILE.read_bytes()).hexdigest()}
    if D.PHASE == "fd":
        phase_fd(H, res)
    else:
        assert json.loads((OUT / "run_fd.json").read_text())["phase1_pass"] is True, "held-out seeds locked: amended Phase 1 has not passed"
        {"c3": D.phase_c3, "c4a": D.phase_c4a}[D.PHASE](H, res)
    res["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}
    (OUT / f"run_{D.PHASE}.json").write_text(json.dumps(res, indent=1, default=str))
    print("[end]", json.dumps(res["memory"]), "phase1_pass", res.get("phase1_pass"), flush=True)


if __name__ == "__main__":
    main()
