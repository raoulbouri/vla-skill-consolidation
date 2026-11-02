"""Task 10A, Phase 2b: why does one eager forward differ from another? A diagnostic: no test, no training.
E1 reproducibility and call order of the k = 0 student velocity across repeated calls and separately built functions.
E2 the identical student: student vs teacher velocity under value_and_grad, per step.
E3 first diverging tensor INSIDE a layer (linen capture_intermediates), plain vs jit vs jvp vs vjp, same inputs.
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, traverse_util

from pi05_real_env import ROOT, load_real_env
from task10a_primal import diff, re_expert
from test_pi05_path_kl import ALPHA, PERT_SEED, _flat, injected, make_student
from test_pi05_pathwise_gradient import NOISE_SEED, OBS_IDS, TRAINABLE, in_groups

from clvla.ogpo import pi05_pathwise_kl as pw
from clvla.ogpo import pi05_velocity as pv

TAG = os.environ.get("TAG", "default")
OUT = ROOT / "outputs" / "ogpo_port" / "task10a"
LAYERS = (0, 5, 11)


def same(a, b):
    return bool(np.array_equal(np.asarray(a), np.asarray(b)))


def main():
    e = load_real_env()
    teacher = e["model"]
    student, _ = make_student(teacher, ALPHA, PERT_SEED)
    twin, _ = make_student(teacher, 0.0, PERT_SEED)
    obs = e["observation"](OBS_IDS)
    x0, eps = injected(NOISE_SEED, len(OBS_IDS))
    x = jnp.asarray(x0)
    sc, tc, wc = pv.build_prefix(student, obs), pv.build_prefix(teacher, obs), pv.build_prefix(twin, obs)
    gd, theta, rest = nnx.split(student, in_groups(TRAINABLE), ...)
    res = {"tag": TAG, "xla_flags": os.environ.get("XLA_FLAGS", "")}

    # ---- E1
    def v_of(th):
        return pv.velocity(nnx.merge(gd, th, rest), sc, x, 1.0)

    seq = {}
    seq["direct_1"] = pv.velocity(student, sc, x, 1.0)
    seq["direct_2"] = pv.velocity(student, sc, x, 1.0)
    seq["merged_1"] = v_of(theta)
    seq["merged_2"] = v_of(theta)
    seq["sum_then_v"] = (lambda th: (jnp.sum(v_of(th)), v_of(th))[1])(theta)
    seq["vjp_1"] = jax.vjp(lambda t: (jnp.sum(v_of(t)), v_of(t)), theta, has_aux=True)[2]
    seq["direct_after_vjp"] = pv.velocity(student, sc, x, 1.0)
    u = jax.tree.map(jnp.zeros_like, theta)
    seq["jvp_1"] = jax.jvp(v_of, (theta,), (u,))[0]
    seq["direct_after_jvp"] = pv.velocity(student, sc, x, 1.0)
    seq["jit_a"] = jax.jit(v_of)(theta)
    seq["jit_b"] = jax.jit(lambda t: v_of(t))(theta)
    seq["direct_after_jit"] = pv.velocity(student, sc, x, 1.0)
    ref = np.asarray(seq["direct_1"])
    res["E1_vs_direct_1"] = {k: diff(np.asarray(v), ref) for k, v in seq.items()}
    for k, d in res["E1_vs_direct_1"].items():
        print(f"[E1] {k:18s} max_abs {d['max_abs']:.3e} rel_l2 {d['rel_l2']:.2e} frac {d['frac_differ']:.3f}", flush=True)
    # teacher vs identical student on the same programme
    res["E1_twin_vs_teacher_direct"] = diff(np.asarray(pv.velocity(twin, wc, x, 1.0)), np.asarray(pv.velocity(teacher, tc, x, 1.0)))
    print("[E1] twin vs teacher direct", res["E1_twin_vs_teacher_direct"]["max_abs"], flush=True)

    # ---- E2
    gd_w, th_w, rest_w = nnx.split(twin, in_groups(TRAINABLE), ...)

    def f_w(th):
        o = pw.pathwise_kl(nnx.merge(gd_w, th, rest_w), wc, teacher, tc, x0, eps)
        return jnp.sum(o["path_kl"]), o

    plain_o = f_w(th_w)[1]
    (val, o), _ = jax.value_and_grad(f_w, has_aux=True)(th_w)
    (val_b, o_b), _ = jax.value_and_grad(f_w, has_aux=True)(th_w)
    res["E2_identical_student"] = {"plain_objective": float(jnp.sum(plain_o["path_kl"])), "vag_objective": float(val), "vag_repeat_objective": float(val_b),
                                   "plain_per_step": np.asarray(plain_o["kl"][0]).tolist(), "vag_per_step": np.asarray(o["kl"][0]).tolist(),
                                   "vag_vS_vs_vT_max_abs_per_step": [float(jnp.abs(o["velocities_student"][0, k] - o["velocities_teacher"][0, k]).max()) for k in range(10)]}
    print("[E2]", json.dumps(res["E2_identical_student"]), flush=True)

    # ---- E3
    from openpi.models import gemma as _gemma

    block = _gemma.Block(configs=(_gemma.get_config("gemma_2b"), _gemma.get_config("gemma_300m")))
    flat = _flat(nnx.split(student)[1])
    lf = {k[len("PaliGemma/llm/layers/"):]: v for k, v in flat.items() if k.startswith("PaliGemma/llm/layers/")}
    ex_all = {k: v for k, v in lf.items() if re_expert(k)}
    fr_all = {k: v for k, v in lf.items() if not re_expert(k)}
    time_ = jnp.ones((1,), jnp.float32)
    tok, smask, ar, cond = student.embed_suffix(sc.observation, x, time_)
    from openpi.models.pi0 import make_attn_mask
    import einops

    sa = make_attn_mask(smask, ar)
    pa = einops.repeat(sc.prefix_mask, "b p -> b s p", s=tok.shape[1])
    mask = jnp.concatenate([pa, sa], axis=-1)[:, None, :, :]
    pos = jnp.sum(sc.prefix_mask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1
    res["E3"] = {}

    def capture(ex, fr, xs_in, kvi):
        params = traverse_util.unflatten_dict({**ex, **fr}, sep="/")
        (out, _), st = block.apply({"params": params}, [None, xs_in], kvi, pos, mask, [None, cond], True, capture_intermediates=True, mutable=["intermediates"])
        inter = traverse_util.flatten_dict(st["intermediates"], sep="/")
        leaves = {}
        for name, val in inter.items():
            for i, a in enumerate(jax.tree.leaves(val)):
                leaves[f"{name}[{i}]"] = a
        return jnp.sum(out[1]), {"out": out[1], "inter": leaves}

    xs = tok.astype(jnp.bfloat16)
    jc = jax.jit(capture)
    for i in LAYERS:
        ex, fr = {k: v[i] for k, v in ex_all.items()}, {k: v[i] for k, v in fr_all.items()}
        kvi = jax.tree.map(lambda a: a[i], sc.kv_cache)
        plain = capture(ex, fr, xs, kvi)[1]
        modes = {"plain_repeat": capture(ex, fr, xs, kvi)[1], "jit": jc(ex, fr, xs, kvi)[1],
                 "jvp": jax.jvp(lambda t: capture(t, fr, xs, kvi), (ex,), (jax.tree.map(jnp.zeros_like, ex),))[0][1],
                 "vjp": jax.vjp(lambda t: capture(t, fr, xs, kvi), ex, has_aux=True)[2]}
        order = list(plain["inter"])
        row = {"order": order, "dtypes": {n: str(plain["inter"][n].dtype) for n in order}}
        for m, o in modes.items():
            d = {n: diff(np.asarray(o["inter"][n]).astype(np.float64), np.asarray(plain["inter"][n]).astype(np.float64)) for n in order}
            first = next((n for n in order if d[n]["max_abs"] != 0), None)
            row[m] = {"first": first, "first_detail": d.get(first), "first_dtype": row["dtypes"].get(first), "n_differ": sum(v["max_abs"] != 0 for v in d.values()),
                      "out": diff(np.asarray(o["out"]).astype(np.float64), np.asarray(plain["out"]).astype(np.float64)), "all": d}
            print(f"[E3 layer {i}] {m}: first {first} ({row['dtypes'].get(first)}) {json.dumps(d.get(first))}; differ {row[m]['n_differ']}/{len(order)}; out {row[m]['out']['max_abs']:.3e}", flush=True)
        res["E3"][str(i)] = row
        print(f"[E3 layer {i}] order: {order}", flush=True)
        # advance with the plain result so later layers see realistic inputs
        if i != LAYERS[-1]:
            for j in range(i, LAYERS[LAYERS.index(i) + 1]):
                exj, frj = {k: v[j] for k, v in ex_all.items()}, {k: v[j] for k, v in fr_all.items()}
                xs = capture(exj, frj, xs, jax.tree.map(lambda a: a[j], sc.kv_cache))[1]["out"]
    (OUT / f"probe_{TAG}.json").write_text(json.dumps(res, indent=1))
    print("[end]", flush=True)


if __name__ == "__main__":
    main()
