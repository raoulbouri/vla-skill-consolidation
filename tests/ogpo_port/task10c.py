"""Task 10C: direct BF16 vs FP32 pathwise-gradient comparison. No optimiser, no parameter update, production code untouched.
  CONFIG=C3|C1|C2|C4A|C4B  CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy tests/ogpo_port/task10c.py        (C3 first: it writes the common cache)

C1 BF16 eager, Triton off.   C2 BF16 jit, Triton off.   C3 BF16 jit, Triton on (canonical production).
C4A FP32 eager, Triton off, canonical BF16 prefix caches promoted to FP32 (primary reference).
C4B FP32 eager, Triton off, prefix caches recomputed in FP32 (secondary; never combined with C4A).
Parameters are the BF16 values, promoted exactly for FP32. BF16 prefix caches depend on the XLA flags, so C1, C2 and C4A LOAD the canonical cache and
verify its hash. For each noise seed a configuration generates its own student states, evaluates the teacher there and differentiates the full objective;
the detached gradient holds that configuration's OWN stored chain fixed, so g_state = g_full - g_detached is taken at the same states.
No acceptance threshold on the gradient comparison: none was justified in advance, so discrepancies are report-only. Hard integrity checks only:
input hashes match, gradients finite, reused canonical gradients reproduce bitwise.
"""

from __future__ import annotations

import os

DET = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
NT = DET + " --xla_gpu_enable_triton_gemm=false"
CONFIGS = {"C1": ("bf16", "eager", NT), "C2": ("bf16", "jit", NT), "C3": ("bf16", "jit", DET), "C4A": ("fp32", "eager", NT), "C4B": ("fp32", "eager", NT)}
CONFIG = os.environ["CONFIG"]
PREC, MODE, FLAGS = CONFIGS[CONFIG]
os.environ["TENB_FLAGS"] = FLAGS                # applied by task10b at import, before JAX is imported
os.environ["TAG"] = CONFIG

import gc  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import task10b as T  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

pv = T.pv
SEEDS = (10000, 11000, 11001, 11002)            # 10000 is the seed used by the Task 10B finite differences
REUSE = {10000: "p2_det_1.npz", 11000: "p3_a_11000.npz", 11001: "p3_a_11001.npz", 11002: "p3_a_11002.npz"}


def tiny(path, a):      # verified bitwise harmless in Task 10A; FP32 only, to fit in GPU memory
    return jnp.zeros((18,) if "layers" in jax.tree_util.keystr(path) else (1,), a.dtype)


def main():
    H = T.helpers()
    nnx, flat, sha = H["nnx"], H["_flat"], H["sha"]
    B = H["ROOT"] / "outputs" / "ogpo_port" / "task10b"
    OUT = H["ROOT"] / "outputs" / "ogpo_port" / "task10c"
    TMP = OUT / "tmp"
    TMP.mkdir(parents=True, exist_ok=True)
    res = {"config": CONFIG, "precision": PREC, "mode": MODE, "xla_flags": os.environ["XLA_FLAGS"], "env": T.env_record(), "seeds": list(SEEDS), "acceptance_threshold": None, "times_s": {}}
    assert os.environ["XLA_FLAGS"] == FLAGS

    e = H["load_real_env"]()
    teacher = e["model"]
    student, rec = H["make_student"](teacher, H["ALPHA"], H["PERT_SEED"])
    obs = e["observation"](H["OBS_IDS"])
    sc, tc = pv.build_prefix(student, obs), pv.build_prefix(teacher, obs)
    obs_p = sc.observation
    res["perturbation"] = rec
    res["input_sha256"] = {"student_all": sha(nnx.split(student)[1]), "teacher_all": sha(nnx.split(teacher)[1]), "obs": sha(obs_p)}
    res["own_flags_bf16_cache_sha"] = {"s": sha((sc.prefix_mask, sc.kv_cache)), "t": sha((tc.prefix_mask, tc.kv_cache))}
    cache_file = OUT / "bf16_cache_canonical.npz"
    s_leaves, s_def = jax.tree.flatten(sc.kv_cache)
    t_leaves, t_def = jax.tree.flatten(tc.kv_cache)
    if CONFIG == "C3":
        np.savez(cache_file, **{f"s{i}": np.asarray(a).astype(np.float32) for i, a in enumerate(s_leaves)}, **{f"t{i}": np.asarray(a).astype(np.float32) for i, a in enumerate(t_leaves)},
                 mask_s=np.asarray(sc.prefix_mask), mask_t=np.asarray(tc.prefix_mask))
    z = np.load(cache_file)
    can_s = (jnp.asarray(z["mask_s"]), jax.tree.unflatten(s_def, [jnp.asarray(z[f"s{i}"]).astype(jnp.bfloat16) for i in range(len(s_leaves))]))
    can_t = (jnp.asarray(z["mask_t"]), jax.tree.unflatten(t_def, [jnp.asarray(z[f"t{i}"]).astype(jnp.bfloat16) for i in range(len(t_leaves))]))
    res["canonical_cache_sha"] = {"s": sha(can_s), "t": sha(can_t)}
    canon = json.loads((B / "p2_det_1.json").read_text())["input_sha256"]
    assert res["canonical_cache_sha"]["s"] == canon["s_cache"] and res["canonical_cache_sha"]["t"] == canon["t_cache"], "canonical cache hash mismatch"
    assert res["input_sha256"]["obs"] == canon["obs"], "observation hash mismatch"

    if PREC == "bf16":
        gd_s, theta, other_s = nnx.split(student, H["in_groups"](H["TRAINABLE"]), ...)
        gd_t, t_all = nnx.split(teacher)
        assert sha(theta) == canon["theta"] and sha(t_all) == canon["teacher"] and sha(other_s) == canon["rest_s"], "parameter hash mismatch"
        base = {"rest_s": (other_s,), "teacher": (t_all,), "obs": obs_p, "s_cache": can_s, "t_cache": can_t}
        res["cache_used"] = "canonical BF16 (loaded)"
    else:
        from task10a_derivatives import promote_models

        t32, s32 = promote_models(e["cfg"].checkpoint.openpi_config, teacher, student)
        res["promotion_exact"] = sha(jax.tree.map(lambda a: a.astype(jnp.bfloat16), nnx.split(s32)[1])) == res["input_sha256"]["student_all"] and sha(jax.tree.map(lambda a: a.astype(jnp.bfloat16), nnx.split(t32)[1])) == res["input_sha256"]["teacher_all"]
        assert res["promotion_exact"]
        up = lambda t: jax.tree.map(lambda a: a.astype(jnp.float32) if a.dtype == jnp.bfloat16 else a, t)  # noqa: E731
        if CONFIG == "C4A":
            s_cache, t_cache = up(can_s), up(can_t)
            res["cache_promotion_exact"] = sha(jax.tree.map(lambda a: a.astype(jnp.bfloat16) if a.dtype == jnp.float32 else a, s_cache)) == res["canonical_cache_sha"]["s"]
            assert res["cache_promotion_exact"]
            res["cache_used"] = "canonical BF16 promoted to FP32"
        else:
            c1, c2 = pv.build_prefix(s32, obs), pv.build_prefix(t32, obs)
            s_cache, t_cache = (c1.prefix_mask, c1.kv_cache), (c2.prefix_mask, c2.kv_cache)
            res["cache_used"] = "recomputed in FP32 under these flags"
        gd_s, theta, vlm_s, other_s = nnx.split(s32, H["in_groups"](H["TRAINABLE"]), H["in_groups"](H["VLM"]), ...)
        gd_t, vlm_t, other_t = nnx.split(t32, H["in_groups"](H["VLM"]), ...)
        vlm_s, vlm_t = (jax.tree_util.tree_map_with_path(tiny, v) for v in (vlm_s, vlm_t))
        base = {"rest_s": (vlm_s, other_s), "teacher": (vlm_t, other_t), "obs": obs_p, "s_cache": s_cache, "t_cache": t_cache}
        del s32, t32
    del teacher, student, sc, tc, z
    e.clear()
    jax.clear_caches()
    gc.collect()
    res["dtypes"] = {"theta": H["dtypes"](theta), "s_cache": H["dtypes"](base["s_cache"]), "t_cache": H["dtypes"](base["t_cache"])}
    print("[setup]", json.dumps({k: res.get(k) for k in ("config", "precision", "mode", "xla_flags", "cache_used", "canonical_cache_sha", "own_flags_bf16_cache_sha", "promotion_exact", "cache_promotion_exact", "dtypes")}), flush=True)

    f = T.canonical(gd_s, gd_t)
    if MODE == "jit":
        full = jax.jit(jax.value_and_grad(f, has_aux=True))
        det = jax.jit(jax.grad(lambda t, inp, st: f(t, inp, "detached", st)[0]))
    else:
        full = jax.value_and_grad(f, has_aux=True)
        det = jax.grad(lambda t, inp, st: f(t, inp, "detached", st)[0])
    res["samples"] = {}
    for seed in SEEDS:
        I, sx, se = T.with_noise(H, base, seed)
        t0 = time.perf_counter()
        (o, a), g = full(theta, I)
        jax.block_until_ready(g)
        t_full = time.perf_counter() - t0
        p, gh = T.primal(o, a), T.grad_host(H, g)
        del g
        t0 = time.perf_counter()
        gd = det(theta, I, jnp.asarray(p["chain"]))
        jax.block_until_ready(gd)
        t_det = time.perf_counter() - t0
        gdh = T.grad_host(H, gd)
        del gd
        row = {"x0_sha": sx, "eps_sha": se, "objective": float(p["objective"]), "full_s": t_full, "detached_s": t_det, "full": H["report"](gh), "detached": H["report"](gdh), "grad_full_digest": T.digest(gh)}
        if CONFIG == "C3":
            ref = np.load(B / REUSE[seed])
            old = {k[3:]: ref[k] for k in ref.files if k.startswith("g__")}
            row["reused_gradient_file"] = REUSE[seed]
            row["reused_gradient_bitwise_equal"] = all(np.array_equal(gh[k], old[k]) for k in gh) and float(ref["p__objective"]) == float(p["objective"])
            assert row["reused_gradient_bitwise_equal"], "saved canonical gradient does not reproduce"
            np.savez(TMP / f"C3_{seed}.npz", **{f"d__{k}": v for k, v in gdh.items()})      # full gradient is not copied: read from REUSE
            del old, ref
        else:
            np.savez(TMP / f"{CONFIG}_{seed}.npz", **{f"f__{k}": v for k, v in gh.items()}, **{f"d__{k}": v for k, v in gdh.items()})
        np.savez(OUT / f"primal_{CONFIG}_{seed}.npz", **p)
        assert all(np.all(np.isfinite(v)) for v in gh.values()) and all(np.all(np.isfinite(v)) for v in gdh.values()), "non-finite gradient"
        res["samples"][str(seed)] = row
        print(f"[{CONFIG} {seed}] objective {row['objective']!r} full {t_full:.1f}s detached {t_det:.1f}s peak {H['_peak_gib']()} GiB " + json.dumps({g_: {n: (d['norm'], d['nonzero_leaves'], d['zero_leaves']) for n, d in row[g_]['groups'].items()} for g_ in ('full', 'detached')}) + (f" reused_bitwise {row.get('reused_gradient_bitwise_equal')}" if CONFIG == "C3" else ""), flush=True)
        del gh, gdh, p
        gc.collect()
    res["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}
    (OUT / f"run_{CONFIG}.json").write_text(json.dumps(res, indent=1, default=str))
    print("[end]", json.dumps(res["memory"]), flush=True)


if __name__ == "__main__":
    main()
