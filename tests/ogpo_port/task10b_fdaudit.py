"""Task 10B audit of the failed Phase 4 finite differences: is the harness wrong, or is the FP32 objective rough? Criteria are not changed.
Control B setup (FP32 caches). Diagnostic only."""

from __future__ import annotations

import gc
import json
import re

import task10b as T  # sets XLA flags before JAX is imported

import jax
import jax.numpy as jnp
import numpy as np

pv, pw = T.pv, T.pw


def main():
    from task10a_derivatives import promote_models

    H = T.helpers()
    nnx, flat = H["nnx"], H["_flat"]
    e = H["load_real_env"]()
    teacher = e["model"]
    student, _ = H["make_student"](teacher, H["ALPHA"], H["PERT_SEED"])
    obs = e["observation"](H["OBS_IDS"])
    obs_p = pv.build_prefix(student, obs).observation
    t32, s32 = promote_models(e["cfg"].checkpoint.openpi_config, teacher, student)
    sc, tc = pv.build_prefix(s32, obs), pv.build_prefix(t32, obs)

    def tiny(path, a):
        return jnp.zeros((18,) if "layers" in jax.tree_util.keystr(path) else (1,), a.dtype)

    gd_s, theta, vlm_s, other_s = nnx.split(s32, H["in_groups"](H["TRAINABLE"]), H["in_groups"](H["VLM"]), ...)
    gd_t, vlm_t, other_t = nnx.split(t32, H["in_groups"](H["VLM"]), ...)
    vlm_s, vlm_t = (jax.tree_util.tree_map_with_path(tiny, v) for v in (vlm_s, vlm_t))
    del teacher, student, s32, t32
    e.clear()
    jax.clear_caches()
    gc.collect()
    x0, eps = H["injected"](H["NOISE_SEED"], 1)
    I = {"rest_s": (vlm_s, other_s), "teacher": (vlm_t, other_t), "obs": obs_p, "s_cache": (sc.prefix_mask, sc.kv_cache), "t_cache": (tc.prefix_mask, tc.kv_cache), "x0": jnp.asarray(x0), "eps": jnp.asarray(eps)}
    f = T.canonical(gd_s, gd_t)
    res = {}

    # 1. any bf16 left in the traced FP32 programme?
    txt = str(jax.make_jaxpr(lambda t: f(t, I)[0])(theta))
    res["jaxpr_bfloat16_mentions"] = len(re.findall(r"bfloat16|bf16", txt))
    res["jaxpr_chars"] = len(txt)
    i = txt.find("bfloat16")
    res["jaxpr_first_bf16_context"] = txt[max(i - 200, 0): i + 200] if i >= 0 else None
    print("[bf16 in jaxpr]", res["jaxpr_bfloat16_mentions"], res["jaxpr_first_bf16_context"], flush=True)
    del txt

    keys = sorted(flat(theta))
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat(theta).items()}
    names = [by_id[id(l)] for l in leaves]
    th = {k: np.asarray(v, np.float32) for k, v in flat(theta).items()}
    res["param_stats"] = {k: {"max_abs": float(np.abs(v).max()), "rms": float(np.sqrt(np.mean(v.astype(np.float64) ** 2))), "shape": list(v.shape)} for k, v in th.items()}

    def build(arrs):
        return jax.tree.unflatten(treedef, [jnp.asarray(arrs[k], jnp.float32) for k in names])

    # 2. identity: rebuilt, unperturbed parameters
    base_o, base = f(theta, I)
    rebuilt_o, rebuilt = f(build(th), I)
    res["identity"] = {"objective": float(base_o), "rebuilt_objective": float(rebuilt_o), "bitwise": all(np.array_equal(np.asarray(base[k]), np.asarray(rebuilt[k])) for k in base)}
    print("[identity]", json.dumps(res["identity"]), flush=True)

    g = {k: np.asarray(v, np.float64) for k, v in flat(jax.grad(lambda t: f(t, I)[0])(theta)).items()}
    gn = sum(float(np.sum(v**2)) for v in g.values()) ** 0.5
    r = np.random.default_rng(60000)
    expert = [k for k in keys if k.startswith("PaliGemma")]
    heads = [k for k in keys if not k.startswith("PaliGemma")]

    def restrict(d, ks):
        return {k: (d[k] if k in ks else np.zeros_like(d[k])) for k in keys}

    dirs = {"full": g, "full_only_action_out_kernel": restrict(g, ["action_out_proj/kernel"]), "full_only_heads": restrict(g, heads), "full_only_expert": restrict(g, expert),
            "random": {k: r.standard_normal(g[k].shape) for k in keys}}
    res["dirs"] = {}
    for name, d in dirs.items():
        n = sum(float(np.sum(v**2)) for v in d.values()) ** 0.5
        u = {k: (d[k] / n).astype(np.float32) for k in keys}
        res_u = {"u_max_abs": float(max(np.abs(v).max() for v in u.values())), "proj_g": sum(float(np.sum(g[k] * u[k])) for k in keys)}
        (o0, a0), (dO, dA) = jax.jvp(lambda t: f(t, I), (theta,), (build(u),))
        res_u["jvp"] = float(dO)
        dvS0, dvT0 = np.asarray(dA["vs"][0, 0], np.float64), np.asarray(dA["kl"][0], np.float64)
        rows = []
        for h in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
            # step exact in float64, then rounded once to fp32
            p = {k: (th[k].astype(np.float64) + h * u[k].astype(np.float64)).astype(np.float32) for k in keys}
            step = sum(float(np.sum((p[k].astype(np.float64) - th[k].astype(np.float64)) ** 2)) for k in keys) ** 0.5
            changed = sum(int(np.count_nonzero(p[k] != th[k])) for k in keys)
            pred_first = sum(float(np.sum(g[k] * (p[k].astype(np.float64) - th[k].astype(np.float64)))) for k in keys)
            fp, ap = f(build(p), I)
            act = np.asarray(ap["vs"][0, 0], np.float64) - np.asarray(a0["vs"][0, 0], np.float64)
            # effect of the rounded step on vS_0 alone, predicted by a JVP along the step actually taken
            tan = {k: (p[k].astype(np.float64) - th[k].astype(np.float64)).astype(np.float32) for k in keys}
            _, dv_pred = jax.jvp(lambda t: f(t, I)[1]["vs"][0, 0], (theta,), (build(tan),))
            dv_pred = np.asarray(dv_pred, np.float64)
            rows.append({"h": h, "step_norm_actual": step, "entries_changed": changed, "df": float(fp) - float(o0), "first_order_on_actual_step": pred_first,
                         "dKL_per_step": (np.asarray(ap["kl"][0], np.float64) - np.asarray(a0["kl"][0], np.float64)).tolist(),
                         "dvS0_norm": float(np.linalg.norm(act)), "dvS0_jvp_on_actual_step_norm": float(np.linalg.norm(dv_pred)), "dvS0_residual_norm": float(np.linalg.norm(act - dv_pred))})
            print(f"[{name}] " + json.dumps({k: v for k, v in rows[-1].items() if k != "dKL_per_step"}) + " dKL_k0,k1,k9 " + json.dumps([rows[-1]["dKL_per_step"][i] for i in (0, 1, 9)]), flush=True)
        res_u["rows"] = rows
        res["dirs"][name] = res_u
        print(f"[{name} summary] {json.dumps({k: v for k, v in res_u.items() if k != 'rows'})}", flush=True)
    (H["ROOT"] / "outputs" / "ogpo_port" / "task10b" / "fd_audit.json").write_text(json.dumps(res, indent=1))
    print("[end] peak", H["_peak_gib"](), flush=True)


if __name__ == "__main__":
    main()
