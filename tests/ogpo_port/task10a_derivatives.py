"""Task 10A, Phases 3-5: derivative agreement of ONE canonical function at three precisions. Diagnostic only; production code untouched.
PREC=bf16    the production arithmetic
PREC=master  FP32 master parameters, cast to the original bf16 inside the function; forward arithmetic as bf16
PREC=fp32    full FP32 computation: the exact bf16 parameter values promoted to FP32, Pi0Config(dtype="float32"), prefix caches recomputed
Same observation, noise, schedule, score correction, covariance and objective in all three. Discrepancies: |D f[u] - <g, u>|, relative
and scale-normalised by ||g|| ||u||. Same-sample directions are built from the gradient they are compared with, so they correlate with
its own rounding noise; "xfit" directions are built from the independently compiled (jit) gradients to remove that selection bias.
"""

from __future__ import annotations

import dataclasses
import gc
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from pi05_real_env import ROOT, load_real_env
from task10a_primal import compare, diff, dtypes, ordered, sha  # noqa: F401
from test_pi05_path_kl import ALPHA, PERT_SEED, _flat, _peak_gib, injected, make_student
from test_pi05_pathwise_gradient import NOISE_SEED, OBS_IDS, TRAINABLE, VLM, in_groups, rss_gib

from clvla.ogpo import pi05_pathwise_kl as pw
from clvla.ogpo import pi05_velocity as pv

PREC = os.environ.get("PREC", "bf16")
TAG = os.environ.get("TAG", PREC)
OUT = ROOT / "outputs" / "ogpo_port" / "task10a"
N_RANDOM = 5


def promote_models(cfg_name, teacher, student):
    """FP32 copies of both models with Pi0Config(dtype=float32), preserving buffer sharing so shared leaves are stored once."""
    from clvla.policy import openpi_configs

    tc, _ = openpi_configs(cfg_name)
    mc = dataclasses.replace(tc.model, dtype="float32")
    memo = {}

    def up(a):
        if id(a) not in memo:
            memo[id(a)] = jnp.asarray(a).astype(jnp.float32)
        return memo[id(a)]

    def build(m):
        pure = jax.tree.map(up, nnx.split(m)[1].to_pure_dict())
        return mc.load(pure, remove_extra_params=False)

    return build(teacher), build(student)


def host(tree):
    return {k: np.asarray(v).astype(np.float64) for k, v in _flat(tree).items()}


def dot(a, b):
    return sum(float(np.sum(a[k] * b[k])) for k in a)


def norm(a):
    return dot(a, a) ** 0.5


def cmp(a, b):
    return {"cosine": dot(a, b) / (norm(a) * norm(b)), "rel_norm_diff": norm({k: a[k] - b[k] for k in a}) / norm(a), "norm_a": norm(a), "norm_b": norm(b)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e = load_real_env()
    teacher = e["model"]
    student, rec = make_student(teacher, ALPHA, PERT_SEED)
    obs = e["observation"](OBS_IDS)
    x0, eps = injected(NOISE_SEED, 1)
    res = {"prec": PREC, "tag": TAG, "xla_flags": os.environ.get("XLA_FLAGS", ""), "times_s": {}, "label": "numerical diagnostic" if PREC == "fp32" else "bf16 forward"}
    res["student_kernel_sha"] = sha(_flat(nnx.split(student)[1])["action_out_proj/kernel"])

    if PREC == "fp32":
        res["bf16_param_sha"] = sha(nnx.split(student)[1])
        teacher, student = promote_models(e["cfg"].checkpoint.openpi_config, teacher, student)
        e["policy"] = None
        e["model"] = None
        gc.collect()
        # exact promotion: casting back reproduces the bf16 parameters bit for bit
        back = jax.tree.map(lambda a: a.astype(jnp.bfloat16), nnx.split(student)[1])
        res["fp32_params_cast_back_sha_equal"] = sha(back) == res["bf16_param_sha"]
        del back
    sc, tc = pv.build_prefix(student, obs), pv.build_prefix(teacher, obs)
    # Once the caches exist the expert pass never reads the VLM leaves ([None, tokens]). master/fp32 replace them by placeholders to
    # fit in GPU memory: the master run checks bitwise against bf16_det that this changes nothing; the objective is untouched.
    def tiny(path, a):
        return jnp.zeros((18,) if "layers" in jax.tree_util.keystr(path) else (1,), a.dtype)

    gd_s, theta0, vlm_s, other_s = nnx.split(student, in_groups(TRAINABLE), in_groups(VLM), ...)
    gd_t, vlm_t, other_t = nnx.split(teacher, in_groups(VLM), ...)
    if PREC != "bf16":
        vlm_s, vlm_t = (jax.tree_util.tree_map_with_path(tiny, v) for v in (vlm_s, vlm_t))
        del student, teacher
        e.clear()
        jax.clear_caches()
        gc.collect()
    rest_s, t_all = (vlm_s, other_s), (vlm_t, other_t)
    I = {"rest_s": rest_s, "teacher": t_all, "obs": sc.observation, "s_cache": (sc.prefix_mask, sc.kv_cache), "t_cache": (tc.prefix_mask, tc.kv_cache),
         "x0": jnp.asarray(x0), "eps": jnp.asarray(eps)}
    if PREC == "master":
        orig = jax.tree.map(lambda a: a.dtype, theta0)
        theta = jax.tree.map(lambda a: a.astype(jnp.float32), theta0)
        to_model = lambda th: jax.tree.map(lambda a, d: a.astype(d), th, orig)  # noqa: E731
    else:
        theta = theta0
        to_model = lambda th: th  # noqa: E731

    def f(th, inp, mode="full", stored=None):
        o = pw.pathwise_kl(nnx.merge(gd_s, to_model(th), *inp["rest_s"]), pv.PrefixCache(inp["obs"], *inp["s_cache"]), nnx.merge(gd_t, *inp["teacher"]),
                           pv.PrefixCache(inp["obs"], *inp["t_cache"]), inp["x0"], inp["eps"], mode=mode, stored_chain=stored)
        return jnp.sum(o["path_kl"]), {"chain": o["chain"], "vs": o["velocities_student"], "vt": o["velocities_teacher"], "kl": o["kl"]}

    def hostp(o, a):
        out = {k: np.asarray(v) for k, v in a.items()}
        out["objective"] = float(o)
        return out

    # ---------------- primal, Phase 2 repeated at this precision
    jf = jax.jit(f)
    jg = jax.jit(jax.value_and_grad(f, has_aux=True))
    prim = {"plain": hostp(*f(theta, I))}
    (o, a), g_full_tree = jax.value_and_grad(f, has_aux=True)(theta, I)
    res["memory_after_eager_gradient"] = {"peak_gib": _peak_gib(), "rss_gib": rss_gib()}
    prim["value_and_grad"] = hostp(o, a)
    res["leaf_dtypes"] = {"theta": {k: str(v.dtype) for k, v in _flat(theta).items()}, "grad": {k: str(v.dtype) for k, v in _flat(g_full_tree).items()}}
    print("[dtypes] theta", dtypes(theta), "grad", dtypes(g_full_tree), "rest", dtypes(rest_s), "kv", dtypes(I["s_cache"]), flush=True)
    g = {"full": host(g_full_tree)}
    del g_full_tree
    prim["vjp"] = hostp(*(lambda r: (r[0], r[2]))(jax.vjp(lambda t: f(t, I), theta, has_aux=True)))
    prim["jit_forward"] = hostp(*jf(theta, I))
    live = lambda: round((jax.local_devices()[0].memory_stats() or {}).get("bytes_in_use", 0) / 2**30, 2)  # noqa: E731
    res["live_gib_before_jit_gradient"] = live()
    print("[live before jit gradient]", res["live_gib_before_jit_gradient"], flush=True)
    have_jit = True
    try:
        (o, a), gj = jg(theta, I)
        prim["jit_value_and_grad"] = hostp(o, a)
        g["full_jit"] = host(gj)
        del gj
    except jax.errors.JaxRuntimeError as ex:    # recorded as a failure; never replaced by a different objective
        have_jit = False
        res["jit_gradient_failed"] = str(ex)[:300]
        print("[jit gradient FAILED]", res["jit_gradient_failed"], flush=True)
    del jg
    res["memory_after_jit_gradient"] = {"peak_gib": _peak_gib(), "rss_gib": rss_gib()}

    stored = jnp.asarray(prim["plain"]["chain"])
    g["detached"] = host(jax.grad(lambda t: f(t, I, "detached", stored)[0])(theta))
    g["blocked"] = host(jax.grad(lambda t: f(t, I, "teacher_blocked")[0])(theta))
    res["controls"] = {"full_vs_detached": cmp(g["full"], g["detached"]), "full_vs_blocked": cmp(g["full"], g["blocked"])}
    if have_jit:
        g["detached_jit"] = host(jax.jit(jax.grad(lambda t, inp, st: f(t, inp, "detached", st)[0]))(theta, I, stored))
        g["blocked_jit"] = host(jax.jit(jax.grad(lambda t, inp: f(t, inp, "teacher_blocked")[0]))(theta, I))
        res["gradient_program_noise"] = {"full": cmp(g["full"], g["full_jit"]), "detached": cmp(g["detached"], g["detached_jit"]), "blocked": cmp(g["blocked"], g["blocked_jit"])}
        res["controls"].update({"full_vs_detached_jit": cmp(g["full_jit"], g["detached_jit"]), "full_vs_blocked_jit": cmp(g["full_jit"], g["blocked_jit"])})
        print("[noise]", json.dumps(res["gradient_program_noise"]), flush=True)
    print("[controls]", json.dumps(res["controls"]), flush=True)

    # ---------------- directions
    def sub(a, b):
        return {k: a[k] - b[k] for k in a}

    dirs = {"full": g["full"], "full_minus_detached": sub(g["full"], g["detached"]), "full_minus_blocked": sub(g["full"], g["blocked"])}
    if have_jit:
        dirs.update({"xfit_full": g["full_jit"], "xfit_full_minus_detached": sub(g["full_jit"], g["detached_jit"]), "xfit_full_minus_blocked": sub(g["full_jit"], g["blocked_jit"])})
    for i in range(N_RANDOM):
        r = np.random.default_rng(40000 + i)
        dirs[f"random_{i}"] = {k: r.standard_normal(g["full"][k].shape) for k in sorted(g["full"])}
    flat_theta = _flat(theta)
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat_theta.items()}
    leaf_names = [by_id[id(l)] for l in leaves]
    jj = jax.jit(lambda t, inp, u: jax.jvp(lambda s: f(s, inp)[0], (t,), (u,)))
    res["directions"] = {}
    tangent_dtype = None
    for name, d in dirs.items():
        n = norm(d)
        tan = {k: jnp.asarray((d[k] / n).astype(np.float32)).astype(flat_theta[k].dtype) for k in leaf_names}
        tangent_dtype = {k: str(v.dtype) for k, v in tan.items()}
        cast = {k: np.asarray(v).astype(np.float64) for k, v in tan.items()}
        un = norm(cast)
        u = jax.tree.unflatten(treedef, [tan[k] for k in leaf_names])
        t0 = time.perf_counter()
        (_, dk) = jax.jvp(lambda s: f(s, I)[0], (theta,), (u,))
        dk = float(dk)
        res["times_s"].setdefault("jvp_eager", time.perf_counter() - t0)
        dkj = float(jj(theta, I, u)[1])
        row = {"u_norm_after_cast": un, "jvp_eager": dk, "jvp_jit": dkj, "abs_jvp_eager_vs_jit": abs(dk - dkj), "ad_cast_loss": 1.0 - dot(cast, {k: d[k] / n for k in cast}) / un}
        for m, key in (("full", "full"), ("detached", "detached"), ("blocked", "blocked")) + ((("full_jit", "full_jit"),) if have_jit else ()):
            p = dot(g[key], cast)
            row[f"proj_{m}"] = p
        gn = norm(g["full"])
        row["abs_jvp_eager_vs_vjp_eager"] = abs(dk - row["proj_full"])
        row["rel_jvp_eager_vs_vjp_eager"] = abs(dk - row["proj_full"]) / abs(dk)
        row["scaled_jvp_eager_vs_vjp_eager"] = abs(dk - row["proj_full"]) / (gn * un)
        if have_jit:
            row["abs_jvp_jit_vs_vjp_jit"] = abs(dkj - row["proj_full_jit"])
            row["rel_jvp_jit_vs_vjp_jit"] = abs(dkj - row["proj_full_jit"]) / abs(dkj)
            row["scaled_jvp_jit_vs_vjp_jit"] = abs(dkj - row["proj_full_jit"]) / (norm(g["full_jit"]) * un)
        row["scaled_jvp_eager_vs_detached"] = abs(dk - row["proj_detached"]) / (gn * un)
        row["scaled_jvp_eager_vs_blocked"] = abs(dk - row["proj_blocked"]) / (gn * un)
        res["directions"][name] = row
        print(f"[dir] {name}: {json.dumps(row)}", flush=True)
    res["tangent_dtypes"] = tangent_dtype
    res["accumulator"] = "numpy float64"

    # ---------------- Phase 4: forward equality for FP32 master
    if PREC == "master":
        res["master_cast_preserves_bf16_values"] = sha(jax.tree.map(lambda a, d: a.astype(d), theta, orig)) == sha(theta0)
    ref = OUT / "derivs_primal_bf16_det.npz"
    if PREC == "master" and ref.exists():
        z = np.load(ref)
        res["master_forward_bitwise_equal_to_bf16_det"] = {m: {k: bool(np.array_equal(prim[m][k], z[f"{m}__{k}"])) for k in ("chain", "vs", "vt", "kl")} for m in ("plain", "jit_forward")}
        print("[master forward]", json.dumps(res["master_forward_bitwise_equal_to_bf16_det"]), flush=True)
    res["objectives"] = {k: v["objective"] for k, v in prim.items()}
    res["primal_vs_plain"] = {k: {kk: compare(v, prim["plain"])[kk] for kk in ("first_divergence", "first_divergence_detail", "objective_minus_ref", "x_10")} for k, v in prim.items() if k != "plain"}
    print("[objectives]", json.dumps(res["objectives"]), flush=True)
    print("[primal]", json.dumps(res["primal_vs_plain"]), flush=True)
    np.savez_compressed(OUT / f"derivs_primal_{TAG}.npz", **{f"{n}__{k}": v for n, p in prim.items() for k, v in p.items() if k != "objective"})
    res["memory_final"] = {"peak_gib": _peak_gib(), "rss_gib": rss_gib()}
    (OUT / f"derivatives_{TAG}.json").write_text(json.dumps(res, indent=1))
    print("[end]", json.dumps(res["memory_final"]), flush=True)


if __name__ == "__main__":
    main()
