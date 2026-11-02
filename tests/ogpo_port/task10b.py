"""Task 10B: validate the canonical training execution path. Diagnostic: no optimiser, no parameter update, production code untouched.
  PHASE=flags|p2|p3|p5|p4  TAG=<name>  [CACHE=A|B]  [TENB_FLAGS=...]  scripts/py-policy tests/ogpo_port/task10b.py   (CLVLA_USE_GPU=1 CLVLA_GPU=1)

Canonical path, in this transformation order: jax.jit(jax.value_and_grad(f, has_aux=True)); BF16 model, FP32 latents. f(theta, I) is pure:
every array is an explicit argument, no RNG, no mutation. jax.value_and_grad(jax.jit(f)) is a different programme and is not substituted.

Criteria fixed before the first run (not relaxed afterwards):
  P2  objective, all intermediates and gradient bitwise equal within a process and across fresh processes.
  P3  all gradients finite; expert and heads nonzero. Variability is report-only: no threshold was justified in advance.
  P4  AD vs central FD at best h: rel <= 1e-3 along full, full-minus-detached, heads-only; scaled <= 1e-5 along the random direction.
"""

from __future__ import annotations

import os
import sys

FLAGS = os.environ.get("TENB_FLAGS", "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0")
assert "jax" not in sys.modules, "XLA flags must be set before JAX is imported"
os.environ["XLA_FLAGS"] = FLAGS
os.environ["CLVLA_RUN_PI05_TESTS"] = "1"

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

PHASE, TAG, CACHE = os.environ.get("PHASE", "p2"), os.environ.get("TAG", "run"), os.environ.get("CACHE", "A")
FD_STEPS = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)
FD_REL, FD_SCALED = 1e-3, 1e-5
SEEDS = tuple(range(11000, 11008))
REPEATS = 5


def env_record():
    import jaxlib

    smi = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    ver = subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout.split("CUDA Version:")[-1].split()[0]
    return {"jax": jax.__version__, "jaxlib": getattr(jaxlib, "__version__", None) or getattr(getattr(jaxlib, "version", None), "__version__", None), "device": str(jax.devices()[0]),
            "device_kind": jax.devices()[0].device_kind, "gpus": smi, "cuda_driver_api": ver, "cuda_visible": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "xla_flags": os.environ.get("XLA_FLAGS"), "transformation_order": "jax.jit(jax.value_and_grad(f, has_aux=True))"}


def flags_check():
    """Evidence that the flags reach XLA: the GEMM backend config of a compiled dot (autotuning fills in an algorithm)."""
    a = jnp.ones((256, 256), jnp.float32)
    txt = jax.jit(lambda x: x @ x).lower(a).compile().as_text()
    lines = [l.strip()[:400] for l in txt.splitlines() if "gemm" in l.lower() or "algorithm" in l.lower()]
    print("[flags]", json.dumps({"xla_flags": os.environ["XLA_FLAGS"], "gemm_lines": lines[:3]}), flush=True)


from clvla.ogpo import pi05_pathwise_kl as pw  # noqa: E402
from clvla.ogpo import pi05_velocity as pv  # noqa: E402


def helpers():
    from flax import nnx

    from pi05_real_env import ROOT, load_real_env
    from task10a_primal import dtypes, sha
    from test_pi05_path_kl import ALPHA, PERT_SEED, _flat, _peak_gib, injected, make_student
    from test_pi05_pathwise_gradient import NOISE_SEED, OBS_IDS, TRAINABLE, VLM, in_groups, report, rss_gib

    return dict(nnx=nnx, ROOT=ROOT, load_real_env=load_real_env, dtypes=dtypes, sha=sha, ALPHA=ALPHA, PERT_SEED=PERT_SEED, _flat=_flat, _peak_gib=_peak_gib, injected=injected,
                make_student=make_student, NOISE_SEED=NOISE_SEED, OBS_IDS=OBS_IDS, TRAINABLE=TRAINABLE, VLM=VLM, in_groups=in_groups, report=report, rss_gib=rss_gib)


def canonical(gd_s, gd_t):
    from flax import nnx

    def f(th, inp, mode="full", stored=None):
        o = pw.pathwise_kl(nnx.merge(gd_s, th, *inp["rest_s"]), pv.PrefixCache(inp["obs"], *inp["s_cache"]), nnx.merge(gd_t, *inp["teacher"]),
                           pv.PrefixCache(inp["obs"], *inp["t_cache"]), inp["x0"], inp["eps"], mode=mode, stored_chain=stored)
        return jnp.sum(o["path_kl"]), {"chain": o["chain"], "vs": o["velocities_student"], "vt": o["velocities_teacher"], "kl": o["kl"]}

    return f


def primal(o, a):
    out = {k: np.asarray(v) for k, v in a.items()}
    out["objective"] = np.asarray(float(o), np.float64)
    return out


def setup_bf16(H, res):
    nnx = H["nnx"]
    e = H["load_real_env"]()
    teacher = e["model"]
    student, rec = H["make_student"](teacher, H["ALPHA"], H["PERT_SEED"])
    obs = e["observation"](H["OBS_IDS"])
    sc, tc = pv.build_prefix(student, obs), pv.build_prefix(teacher, obs)
    gd_s, theta, other_s = nnx.split(student, H["in_groups"](H["TRAINABLE"]), ...)
    gd_t, t_all = nnx.split(teacher)
    base = {"rest_s": (other_s,), "teacher": (t_all,), "obs": sc.observation, "s_cache": (sc.prefix_mask, sc.kv_cache), "t_cache": (tc.prefix_mask, tc.kv_cache)}
    from clvla.policy import openpi_configs

    res["env"] = env_record()
    res["model_compute_dtype"] = openpi_configs(e["cfg"].checkpoint.openpi_config)[0].model.dtype
    res["perturbation"] = rec
    res["dtypes"] = {"theta": H["dtypes"](theta), "rest_s": H["dtypes"](other_s), "teacher": H["dtypes"](t_all), "s_cache": H["dtypes"](base["s_cache"]), "t_cache": H["dtypes"](base["t_cache"]), "obs": H["dtypes"](base["obs"])}
    res["input_sha256"] = {"theta": H["sha"](theta), "rest_s": H["sha"](other_s), "teacher": H["sha"](t_all), "obs": H["sha"](base["obs"]), "s_cache": H["sha"](base["s_cache"]), "t_cache": H["sha"](base["t_cache"])}
    print("[env]", json.dumps(res["env"]), "compute", res["model_compute_dtype"], flush=True)
    print("[dtypes]", json.dumps(res["dtypes"]), flush=True)
    print("[sha]", json.dumps({k: v[:16] for k, v in res["input_sha256"].items()}), flush=True)
    return gd_s, gd_t, theta, base


def with_noise(H, base, seed):
    x0, eps = H["injected"](seed, 1)
    return {**base, "x0": jnp.asarray(x0), "eps": jnp.asarray(eps)}, H["sha"](x0), H["sha"](eps)


def grad_host(H, g):
    return {k: np.asarray(v).astype(np.float32) for k, v in H["_flat"](g).items()}


def digest(arrs):
    h = hashlib.sha256()
    for k in sorted(arrs):
        h.update(k.encode()); h.update(np.asarray(arrs[k]).tobytes())
    return h.hexdigest()


def p2(H, res, out):
    gd_s, gd_t, theta, base = setup_bf16(H, res)
    I, sx, se = with_noise(H, base, H["NOISE_SEED"])
    res["noise_sha256"] = {"x0": sx, "eps": se, "seed": H["NOISE_SEED"]}
    f = canonical(gd_s, gd_t)
    jg = jax.jit(jax.value_and_grad(f, has_aux=True))
    rows, first_p, first_g = [], None, None
    for i in range(REPEATS):
        t0 = time.perf_counter()
        (o, a), g = jg(theta, I)
        jax.block_until_ready(g)
        dt = time.perf_counter() - t0
        p, gh = primal(o, a), grad_host(H, g)
        del g
        if first_p is None:
            first_p, first_g = p, gh
            res["grad_leaf_dtypes"] = sorted({str(v.dtype) for v in jax.tree.leaves(jax.eval_shape(lambda: theta))})
        rows.append({"repeat": i, "objective": float(p["objective"]), "seconds": dt, "primal_sha": digest(p), "grad_sha": digest(gh),
                     "bitwise_equal_first": all(np.array_equal(p[k], first_p[k]) for k in p) and all(np.array_equal(gh[k], first_g[k]) for k in gh)})
        print("[p2 repeat]", json.dumps(rows[-1]), flush=True)
    jg2 = jax.jit(jax.value_and_grad(f, has_aux=True))           # a freshly traced and compiled copy of the same transformation
    (o, a), g = jg2(theta, I)
    p, gh = primal(o, a), grad_host(H, g)
    del g
    res["fresh_compile_same_process"] = {"objective": float(p["objective"]), "bitwise_equal_first": all(np.array_equal(p[k], first_p[k]) for k in p) and all(np.array_equal(gh[k], first_g[k]) for k in gh)}
    print("[p2 fresh compile]", json.dumps(res["fresh_compile_same_process"]), flush=True)
    res["repeats"] = rows
    res["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}
    np.savez(out / f"p2_{TAG}.npz", **{f"p__{k}": v for k, v in first_p.items()}, **{f"g__{k}": v for k, v in first_g.items()})


def p3(H, res, out):
    gd_s, gd_t, theta, base = setup_bf16(H, res)
    f = canonical(gd_s, gd_t)
    jg = jax.jit(jax.value_and_grad(f, has_aux=True))
    eager = jax.value_and_grad(f, has_aux=True)
    with_eager = os.environ.get("EAGER", "0") == "1"
    rows = {}
    for seed in SEEDS:
        I, sx, se = with_noise(H, base, seed)
        (o, a), g = jg(theta, I)
        gh, p = grad_host(H, g), primal(o, a)
        del g
        rep = H["report"](gh)
        row = {"objective": float(o), "x0_sha": sx[:16], "eps_sha": se[:16], "groups": rep["groups"], "zero_leaf_paths": rep["zero_leaf_paths"], "per_step_kl": p["kl"][0].tolist(), "grad_sha": digest(gh), "primal_sha": digest(p)}
        np.savez(out / f"p3_{TAG}_{seed}.npz", **{f"g__{k}": v for k, v in gh.items()}, **{f"p__{k}": v for k, v in p.items()})
        del gh
        if with_eager:
            (oe, ae), ge = eager(theta, I)
            gh = grad_host(H, ge)
            del ge
            row["eager_objective"] = float(oe)
            np.savez(out / f"p3_{TAG}_{seed}_eager.npz", **{f"g__{k}": v for k, v in gh.items()})
            del gh
        rows[str(seed)] = row
        print("[p3]", seed, json.dumps({k: row[k] for k in ("objective", "groups", "zero_leaf_paths", "grad_sha") if k in row} | ({"eager_objective": row["eager_objective"]} if with_eager else {})), flush=True)
    res["samples"] = rows
    res["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}


def p5(H, res, out):
    """BF16 primals for one fixed trajectory, in each execution mode, on their own rollout and on the canonical chain."""
    gd_s, gd_t, theta, base = setup_bf16(H, res)
    I, _, _ = with_noise(H, base, H["NOISE_SEED"])
    f = canonical(gd_s, gd_t)
    (o, a), _ = jax.jit(jax.value_and_grad(f, has_aux=True))(theta, I)
    ref = primal(o, a)
    stored = jnp.asarray(ref["chain"])
    prim = {"jit_vag": ref, "jit_forward": primal(*jax.jit(f)(theta, I)), "eager_forward": primal(*f(theta, I))}
    (o, a), _ = jax.value_and_grad(f, has_aux=True)(theta, I)
    prim["eager_vag"] = primal(o, a)
    fd = lambda t, inp, st: f(t, inp, "detached", st)      # noqa: E731
    prim["jit_forward_on_ref_chain"] = primal(*jax.jit(fd)(theta, I, stored))
    prim["eager_forward_on_ref_chain"] = primal(*fd(theta, I, stored))
    np.savez(out / f"p5_{TAG}.npz", **{f"{m}__{k}": v for m, p in prim.items() for k, v in p.items()})
    res["objectives"] = {m: float(p["objective"]) for m, p in prim.items()}
    print("[p5]", json.dumps(res["objectives"]), flush=True)
    res["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}


def p4(H, res, out):
    """FP32 reference. Control A: BF16 prefix cache promoted to FP32. Control B: prefix cache recomputed in FP32. Never combined."""
    from task10a_derivatives import promote_models

    nnx, flat = H["nnx"], H["_flat"]
    e = H["load_real_env"]()
    teacher, student = e["model"], None
    student, rec = H["make_student"](teacher, H["ALPHA"], H["PERT_SEED"])
    obs = e["observation"](H["OBS_IDS"])
    sc0, tc0 = pv.build_prefix(student, obs), pv.build_prefix(teacher, obs)
    obs_p = sc0.observation
    res["env"], res["perturbation"], res["control"] = env_record(), rec, {"A": "BF16 prefix cache promoted to FP32", "B": "prefix cache recomputed in FP32"}[CACHE]
    res["bf16_param_sha"] = H["sha"](nnx.split(student)[1])
    res["bf16_kv_sha"] = H["sha"](sc0.kv_cache)
    t32, s32 = promote_models(e["cfg"].checkpoint.openpi_config, teacher, student)
    back = jax.tree.map(lambda a: a.astype(jnp.bfloat16), nnx.split(s32)[1])
    res["promotion_exact"] = H["sha"](back) == res["bf16_param_sha"]
    del back
    if CACHE == "A":
        up = lambda t: jax.tree.map(lambda a: a.astype(jnp.float32), t)  # noqa: E731
        s_cache, t_cache = (sc0.prefix_mask, up(sc0.kv_cache)), (tc0.prefix_mask, up(tc0.kv_cache))
        res["cache_promotion_exact"] = H["sha"](jax.tree.map(lambda a: a.astype(jnp.bfloat16), s_cache[1])) == res["bf16_kv_sha"]
    else:
        sc1, tc1 = pv.build_prefix(s32, obs), pv.build_prefix(t32, obs)
        s_cache, t_cache = (sc1.prefix_mask, sc1.kv_cache), (tc1.prefix_mask, tc1.kv_cache)

    def tiny(path, a):      # verified bitwise harmless in Task 10A: the expert pass never reads VLM leaves
        return jnp.zeros((18,) if "layers" in jax.tree_util.keystr(path) else (1,), a.dtype)

    gd_s, theta, vlm_s, other_s = nnx.split(s32, H["in_groups"](H["TRAINABLE"]), H["in_groups"](H["VLM"]), ...)
    gd_t, vlm_t, other_t = nnx.split(t32, H["in_groups"](H["VLM"]), ...)
    vlm_s, vlm_t = (jax.tree_util.tree_map_with_path(tiny, v) for v in (vlm_s, vlm_t))
    del teacher, student, s32, t32, sc0, tc0
    e.clear()
    jax.clear_caches()
    gc.collect()
    x0, eps = H["injected"](H["NOISE_SEED"], 1)
    I = {"rest_s": (vlm_s, other_s), "teacher": (vlm_t, other_t), "obs": obs_p, "s_cache": s_cache, "t_cache": t_cache, "x0": jnp.asarray(x0), "eps": jnp.asarray(eps)}
    res["dtypes"] = {"theta": H["dtypes"](theta), "kv": H["dtypes"](s_cache), "x0": str(x0.dtype)}
    print("[p4 setup]", json.dumps({k: res.get(k) for k in ("control", "promotion_exact", "cache_promotion_exact", "dtypes")}), flush=True)
    f = canonical(gd_s, gd_t)
    obj = lambda t: f(t, I)[0]  # noqa: E731
    keys = sorted(flat(theta))
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat(theta).items()}
    names = [by_id[id(l)] for l in leaves]

    t0 = time.perf_counter()
    (o, a), g = jax.value_and_grad(f, has_aux=True)(theta, I)
    res["eager_gradient_s"] = time.perf_counter() - t0
    prim = {"eager_vag": primal(o, a), "eager_forward": primal(*f(theta, I))}
    gfull = {k: np.asarray(v, np.float64) for k, v in flat(g).items()}
    del g
    z = np.load(out / "p5_canonical.npz")
    stored = jnp.asarray(z["jit_vag__chain"])
    prim["eager_forward_on_ref_chain"] = primal(*f(theta, I, "detached", stored))
    prim["jit_forward"] = primal(*jax.jit(f)(theta, I))
    own = jnp.asarray(prim["eager_forward"]["chain"])
    gdet = {k: np.asarray(v, np.float64) for k, v in flat(jax.grad(lambda t: f(t, I, "detached", own)[0])(theta)).items()}
    np.savez(out / f"p4_{TAG}.npz", **{f"{m}__{k}": v for m, p in prim.items() for k, v in p.items()})
    res["objectives"] = {m: float(p["objective"]) for m, p in prim.items()}
    print("[p4 objectives]", json.dumps(res["objectives"]), flush=True)

    def dot(a, b):
        return sum(float(np.sum(a[k] * b[k])) for k in a)

    gn = dot(gfull, gfull) ** 0.5
    r = np.random.default_rng(50000)
    dirs = {"full": gfull, "full_minus_detached": {k: gfull[k] - gdet[k] for k in keys}, "random": {k: r.standard_normal(gfull[k].shape) for k in keys},
            "heads_only_action_out_proj": {k: (r.standard_normal(gfull[k].shape) if k == "action_out_proj/kernel" else np.zeros_like(gfull[k])) for k in keys}}
    res["gradient_norm"] = gn
    res["fd"] = {}
    th_flat = {k: np.asarray(v, np.float32) for k, v in flat(theta).items()}

    def build(arrs):
        return jax.tree.unflatten(treedef, [jnp.asarray(arrs[k], jnp.float32) for k in names])

    for name, d in dirs.items():
        n = dot(d, d) ** 0.5
        u32 = {k: (d[k] / n).astype(np.float32) for k in keys}
        _, jv = jax.jvp(obj, (theta,), (build(u32),))
        u64 = {k: u32[k].astype(np.float64) for k in keys}
        rows = []
        for h in FD_STEPS:
            plus = {k: th_flat[k] + np.float32(h) * u32[k] for k in keys}
            minus = {k: th_flat[k] - np.float32(h) * u32[k] for k in keys}
            fp, fm = float(obj(build(plus))), float(obj(build(minus)))
            eff = {k: (plus[k].astype(np.float64) - minus[k].astype(np.float64)) / (2.0 * h) for k in keys}   # the step actually taken in fp32
            ad_eff = dot(gfull, eff)
            del plus, minus, eff
            fd = (fp - fm) / (2.0 * h)
            rows.append({"h": h, "f_plus": fp, "f_minus": fm, "fd": fd, "ad_on_effective_step": ad_eff, "abs_err": abs(fd - ad_eff), "rel_err": abs(fd - ad_eff) / abs(ad_eff), "scaled_err": abs(fd - ad_eff) / gn})
            print(f"[fd {CACHE} {name}] {json.dumps(rows[-1])}", flush=True)
        best = min(rows, key=lambda x: x["abs_err"])
        tail = rows[-3:]
        slope = float(np.polyfit(np.log10([x["h"] for x in tail]), np.log10([max(x["abs_err"], 1e-30) for x in tail]), 1)[0])
        passed = best["scaled_err"] <= FD_SCALED if name == "random" else best["rel_err"] <= FD_REL
        res["fd"][name] = {"ad_gradient_projection": dot(gfull, u64), "jvp": float(jv), "rows": rows, "best_h": best["h"], "best_rel_err": best["rel_err"], "best_scaled_err": best["scaled_err"],
                           "loglog_slope_largest_3_h": slope, "criterion": f"scaled <= {FD_SCALED}" if name == "random" else f"rel <= {FD_REL}", "pass": bool(passed)}
        print(f"[fd summary {CACHE} {name}] {json.dumps({k: v for k, v in res['fd'][name].items() if k != 'rows'})}", flush=True)
    res["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}


def main():
    if PHASE == "flags":
        return flags_check()
    H = helpers()
    out = H["ROOT"] / "outputs" / "ogpo_port" / "task10b"
    out.mkdir(parents=True, exist_ok=True)
    res = {"phase": PHASE, "tag": TAG}
    {"p2": p2, "p3": p3, "p5": p5, "p4": p4}[PHASE](H, res, out)
    (out / f"{PHASE}_{TAG}.json").write_text(json.dumps(res, indent=1, default=str))
    print("[end]", json.dumps(res.get("memory")), flush=True)


if __name__ == "__main__":
    main()
