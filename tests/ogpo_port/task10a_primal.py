"""Task 10A, Phases 1-2: primal consistency of ONE canonical function across execution paths. A diagnostic: no test, no training.
  CLVLA_RUN_PI05_TESTS=1 CLVLA_USE_GPU=1 CLVLA_GPU=1 [XLA_FLAGS=...] TAG=<name> scripts/py-policy tests/ogpo_port/task10a_primal.py

f(theta, I) is pure: every array is an explicit argument, no RNG, no mutation. Every mode calls exactly this f with exactly these
inputs. XLA flags are the only thing varied between processes. Order of first possible divergence: x_0 is injected, so vS_0.
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, traverse_util

from pi05_real_env import ROOT, load_real_env
from test_pi05_path_kl import ALPHA, PERT_SEED, _flat, _peak_gib, injected, make_student
from test_pi05_pathwise_gradient import NOISE_SEED, OBS_IDS, TRAINABLE, in_groups

from clvla.ogpo import pi05_pathwise_kl as pw
from clvla.ogpo import pi05_velocity as pv

K = 10
TAG = os.environ.get("TAG", "default")
OUT = ROOT / "outputs" / "ogpo_port" / "task10a"


def sha(tree):
    h = hashlib.sha256()
    for a in jax.tree.leaves(tree):
        a = np.asarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def dtypes(tree):
    out = {}
    for a in jax.tree.leaves(tree):
        k = str(a.dtype)
        out[k] = out.get(k, 0) + 1
    return out


def ulp16(ref):
    """bf16 spacing at |ref|: 7 explicit mantissa bits."""
    m = np.maximum(np.abs(ref.astype(np.float64)), 2.0**-126)
    return 2.0 ** (np.floor(np.log2(m)) - 7)


def diff(a, ref):
    d = np.abs(a.astype(np.float64) - ref.astype(np.float64))
    return {"max_abs": float(d.max()), "rel_max": float(d.max() / max(np.abs(ref).max(), 1e-300)),
            "rel_l2": float(np.linalg.norm(d) / max(np.linalg.norm(ref.astype(np.float64)), 1e-300)),
            "frac_differ": float(np.mean(d != 0)), "max_over_bf16_ulp": float((d / ulp16(ref)).max()),
            "median_nonzero_over_bf16_ulp": float(np.median((d / ulp16(ref))[d != 0])) if np.any(d != 0) else 0.0}


def bf16_fraction(a):
    """Fraction of entries exactly representable in bf16: below 1 means FP32 precision survived past the nominal bf16 output."""
    return float(np.mean(np.asarray(jnp.asarray(a).astype(jnp.bfloat16).astype(jnp.float32)) == a))


def ordered(m):
    names, arrs = [], []
    for k in range(K + 1):
        names.append(f"x_{k}"); arrs.append(m["chain"][0, k])
        if k < K:
            names += [f"vS_{k}", f"vT_{k}", f"kl_{k}"]
            arrs += [m["vs"][0, k], m["vt"][0, k], m["kl"][0, k]]
    return names, arrs


def compare(m, ref):
    names, a = ordered(m)
    _, b = ordered(ref)
    rows = {n: diff(x, y) for n, x, y in zip(names, a, b)}
    first = next((n for n in names if rows[n]["max_abs"] != 0), None)
    return {"first_divergence": first, "first_divergence_detail": rows.get(first), "objective": m["objective"],
            "objective_minus_ref": m["objective"] - ref["objective"], "x_10": rows["x_10"], "per_tensor": rows}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    e = load_real_env()
    teacher = e["model"]
    student, rec = make_student(teacher, ALPHA, PERT_SEED)
    obs = e["observation"](OBS_IDS)
    x0, eps = injected(NOISE_SEED, len(OBS_IDS))
    sc, tc = pv.build_prefix(student, obs), pv.build_prefix(teacher, obs)
    gd_s, theta, rest_s = nnx.split(student, in_groups(TRAINABLE), ...)
    gd_t, t_all = nnx.split(teacher)

    I = {"rest_s": rest_s, "teacher": t_all, "obs": sc.observation, "s_cache": (sc.prefix_mask, sc.kv_cache),
         "t_cache": (tc.prefix_mask, tc.kv_cache), "x0": jnp.asarray(x0), "eps": jnp.asarray(eps)}

    def f(th, inp):
        o = pw.pathwise_kl(nnx.merge(gd_s, th, inp["rest_s"]), pv.PrefixCache(inp["obs"], *inp["s_cache"]), nnx.merge(gd_t, inp["teacher"]),
                           pv.PrefixCache(inp["obs"], *inp["t_cache"]), inp["x0"], inp["eps"])
        return jnp.sum(o["path_kl"]), {"chain": o["chain"], "vs": o["velocities_student"], "vt": o["velocities_teacher"], "kl": o["kl"]}

    def host(obj, aux):
        out = {k: np.asarray(v) for k, v in aux.items()}
        out["objective"] = float(obj)
        return out

    r = np.random.default_rng(30000)
    leaves, treedef = jax.tree.flatten(theta)
    raw = [r.standard_normal(np.shape(a)).astype(np.float32) for a in leaves]
    norm = np.sqrt(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in raw))
    u = jax.tree.unflatten(treedef, [jnp.asarray(a / norm).astype(l.dtype) for a, l in zip(raw, leaves)])
    del raw
    zero = jax.tree.map(jnp.zeros_like, theta)

    assert sha(sc.observation) == sha(tc.observation)
    sha0 = {"obs": sha(I["obs"]), "theta": sha(theta), "rest_s": sha(rest_s), "teacher": sha(t_all), "s_cache": sha(I["s_cache"]), "t_cache": sha(I["t_cache"]),
            "x0": sha(x0), "eps": sha(eps)}
    res = {"tag": TAG, "xla_flags": os.environ.get("XLA_FLAGS", ""), "perturbation_effective_rel": rec["effective_rel_norm"], "input_sha256": sha0,
           "dtypes": {"theta": dtypes(theta), "rest_s": dtypes(rest_s), "teacher": dtypes(t_all), "s_cache": dtypes(I["s_cache"]), "t_cache": dtypes(I["t_cache"]),
                      "x0": str(x0.dtype), "eps": str(eps.dtype)}, "times_s": {}}
    print("[inputs]", json.dumps(res["dtypes"]), flush=True)

    def plain():
        return f(theta, I)

    def vag():
        (o, a), _ = jax.value_and_grad(f, has_aux=True)(theta, I)
        return o, a

    def vjp():
        o, _, a = jax.vjp(lambda t: f(t, I), theta, has_aux=True)
        return o, a

    def jvp(tangent):
        def run():
            (o, a), _ = jax.jvp(lambda t: f(t, I), (theta,), (tangent,))
            return o, a
        return run

    jf = jax.jit(f)
    jg = jax.jit(jax.value_and_grad(f, has_aux=True))

    def jit_vag():
        (o, a), _ = jg(theta, I)
        return o, a

    modes = {"plain": plain, "plain_repeat": plain, "value_and_grad": vag, "vjp": vjp, "jvp_random": jvp(u), "jvp_zero": jvp(zero),
             "jit_forward": lambda: jf(theta, I), "jit_forward_repeat": lambda: jf(theta, I), "jit_value_and_grad": jit_vag, "jit_value_and_grad_repeat": jit_vag}
    prim = {}
    for name, fn in modes.items():
        t0 = time.perf_counter()
        prim[name] = host(*fn())
        res["times_s"][name] = time.perf_counter() - t0
        print(f"[mode] {name}: objective {prim[name]['objective']!r} ({res['times_s'][name]:.1f} s, peak {_peak_gib()} GiB)", flush=True)
    res["objectives"] = {n: p["objective"] for n, p in prim.items()}
    res["comparison_to_plain"] = {n: compare(p, prim["plain"]) for n, p in prim.items() if n != "plain"}
    res["bf16_representable_fraction"] = {n: {"vS_0": bf16_fraction(p["vs"][0, 0]), "vT_0": bf16_fraction(p["vt"][0, 0]), "vS_5": bf16_fraction(p["vs"][0, 5])} for n, p in prim.items()}
    for n, c in res["comparison_to_plain"].items():
        print(f"[cmp] {n}: first {c['first_divergence']} {json.dumps(c['first_divergence_detail'])}; dobj {c['objective_minus_ref']:+.6f}; x_10 max {c['x_10']['max_abs']:.3e}", flush=True)
    print("[bf16 repr]", json.dumps(res["bf16_representable_fraction"]), flush=True)
    np.savez_compressed(OUT / f"primal_{TAG}.npz", **{f"{n}__{k}": v for n, p in prim.items() for k, v in p.items() if k != "objective"})

    # ------------------------------------------------- inside the first velocity call: stages at k = 0, then each layer in isolation
    from openpi.models import gemma as _gemma
    from openpi.models.pi0 import make_attn_mask
    import einops

    x, s = I["x0"], 1.0

    def stages(th, inp):
        m = nnx.merge(gd_s, th, inp["rest_s"])
        c = pv.PrefixCache(inp["obs"], *inp["s_cache"])
        time_ = jnp.broadcast_to(jnp.asarray(s, jnp.float32), (x.shape[0],))
        tok, smask, ar, cond = m.embed_suffix(c.observation, x, time_)
        sa = make_attn_mask(smask, ar)
        pa = einops.repeat(c.prefix_mask, "b p -> b s p", s=tok.shape[1])
        full = jnp.concatenate([pa, sa], axis=-1)
        pos = jnp.sum(c.prefix_mask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1
        (_, out), _ = m.PaliGemma.llm([None, tok], mask=full, positions=pos, kv_cache=c.kv_cache, adarms_cond=[None, cond])
        proj = m.action_out_proj(out[:, -m.action_horizon:])
        v = proj.astype(jnp.float32)
        return jnp.sum(v), {"tokens": tok, "adarms_cond": cond, "llm_out": out, "proj": proj, "v": v, "mask": full, "positions": pos}

    sp = {"plain": stages(theta, I)[1]}
    _, _, sp["vjp"] = jax.vjp(lambda t: stages(t, I), theta, has_aux=True)
    sp["jvp_random"] = jax.jvp(lambda t: stages(t, I), (theta,), (u,))[0][1]
    sp["jit"] = jax.jit(stages)(theta, I)[1]
    keys = ("tokens", "adarms_cond", "llm_out", "proj", "v")
    res["stages_k0"] = {n: {k: diff(np.asarray(sp[n][k]).astype(np.float64), np.asarray(sp["plain"][k]).astype(np.float64)) for k in keys} for n in sp if n != "plain"}
    res["stage_dtypes"] = {k: str(sp["plain"][k].dtype) for k in keys}
    res["stage_v_equals_canonical_vS0"] = bool(np.array_equal(np.asarray(sp["plain"]["v"]), prim["plain"]["vs"][0, 0]))
    for n, st in res["stages_k0"].items():
        print(f"[stage] {n}: " + ", ".join(f"{k} {st[k]['max_abs']:.3e} ({st[k]['max_over_bf16_ulp']:.1f} ulp)" for k in keys), flush=True)
    print("[stage dtypes]", res["stage_dtypes"], "v==vS0", res["stage_v_equals_canonical_vS0"], flush=True)

    # Each layer alone, same inputs in every mode (the plain output feeds the next layer, so a layer's divergence is its own).
    full_flat = _flat(nnx.split(student)[1])
    layer_flat = {k[len("PaliGemma/llm/layers/"):]: v for k, v in full_flat.items() if k.startswith("PaliGemma/llm/layers/")}
    expert = {k: v for k, v in layer_flat.items() if re_expert(k)}
    frozen = {k: v for k, v in layer_flat.items() if not re_expert(k)}
    block = _gemma.Block(configs=(_gemma.get_config("gemma_2b"), _gemma.get_config("gemma_300m")))
    mask, pos = sp["plain"]["mask"][:, None, :, :], sp["plain"]["positions"]
    cond = sp["plain"]["adarms_cond"]
    kv = I["s_cache"][1]

    def layer(ex, fr, xs_in, kv_i):
        params = traverse_util.unflatten_dict({**ex, **fr}, sep="/")
        out, _ = block.apply({"params": params}, [None, xs_in], kv_i, pos, mask, [None, cond], True)
        return jnp.sum(out[1]), out[1]

    jl = jax.jit(layer)
    xs = sp["plain"]["tokens"].astype(jnp.bfloat16)
    depth = next(iter(expert.values())).shape[0]
    u_flat = {k: jax.random.normal(jax.random.key(1), v.shape, jnp.float32).astype(v.dtype) for k, v in expert.items()}
    rows = []
    for i in range(depth):
        ex = {k: v[i] for k, v in expert.items()}
        fr = {k: v[i] for k, v in frozen.items()}
        ui = {k: v[i] for k, v in u_flat.items()}
        kvi = jax.tree.map(lambda a: a[i], kv)
        base = np.asarray(layer(ex, fr, xs, kvi)[1]).astype(np.float64)
        row = {"layer": i}
        _, _, a = jax.vjp(lambda t: layer(t, fr, xs, kvi), ex, has_aux=True)
        row["vjp"] = diff(np.asarray(a).astype(np.float64), base)
        row["jvp"] = diff(np.asarray(jax.jvp(lambda t: layer(t, fr, xs, kvi), (ex,), (ui,))[0][1]).astype(np.float64), base)
        row["jit"] = diff(np.asarray(jl(ex, fr, xs, kvi)[1]).astype(np.float64), base)
        rows.append(row)
        print(f"[layer {i}] vjp {row['vjp']['max_abs']:.3e} jvp {row['jvp']['max_abs']:.3e} jit {row['jit']['max_abs']:.3e} (ulp {row['jit']['max_over_bf16_ulp']:.1f})", flush=True)
        xs = jnp.asarray(base).astype(jnp.bfloat16)
    res["layers_isolated"] = rows

    sha1 = {"obs": sha(I["obs"]), "theta": sha(theta), "rest_s": sha(rest_s), "teacher": sha(t_all), "s_cache": sha(I["s_cache"]), "t_cache": sha(I["t_cache"]), "x0": sha(x0), "eps": sha(eps)}
    res["inputs_unchanged"] = sha0 == sha1
    res["peak_gib"] = _peak_gib()
    (OUT / f"primal_{TAG}.json").write_text(json.dumps(res, indent=1))
    print(f"[end] inputs unchanged {res['inputs_unchanged']}; peak {res['peak_gib']} GiB; sha theta {sha0['theta'][:16]} eps {sha0['eps'][:16]}", flush=True)


def re_expert(path):
    return "_1/" in path + "/"


if __name__ == "__main__":
    main()
