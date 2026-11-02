"""Task 10 diagnostic for the failed forward-vs-reverse check (test_5). Changes no test or threshold; computes no optimiser step.
Run with: CLVLA_RUN_PI05_TESTS=1 CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy tests/ogpo_port/task10_dot_diagnostic.py

(a) Objective and chain under each execution path: plain forward, jvp primal, vjp primal.
(b) Forward-mode directional derivative dK[u] vs <g, u> for the full, detached and teacher-blocked reverse gradients, along
    g_full, (g_full - g_detached), (g_full - g_blocked) and four random directions. A real missing derivative path would show up
    along the discriminating directions: dK[u] must then pick out g_full, not a control.
"""

from __future__ import annotations

import json

import jax
import jax.numpy as jnp
import numpy as np

from pi05_real_env import ROOT, load_real_env
from test_pi05_path_kl import ALPHA, PERT_SEED, _flat, injected, make_student
from test_pi05_pathwise_gradient import NOISE_SEED, OBS_IDS, TRAINABLE, host, split_groups

from clvla.ogpo import pi05_pathwise_kl as pw
from clvla.ogpo import pi05_sde as sde
from clvla.ogpo import pi05_velocity as pv


def main():
    e = load_real_env()
    teacher = e["model"]
    student, _ = make_student(teacher, ALPHA, PERT_SEED)
    obs = e["observation"](OBS_IDS)
    x0, eps = injected(NOISE_SEED, len(OBS_IDS))
    sc, tc = pv.build_prefix(student, obs), pv.build_prefix(teacher, obs)
    apply_s, sub = split_groups(student, TRAINABLE)
    leaves, treedef = jax.tree.flatten(sub)
    keys = sorted(_flat(sub))

    def run(s, mode="full", stored=None):
        return pw.pathwise_kl(apply_s(s), sc, teacher, tc, x0, eps, mode=mode, stored_chain=stored)

    def scalar(mode="full", stored=None):
        return lambda s: jnp.sum(run(s, mode, stored)["path_kl"])

    out = {}
    plain = run(sub)
    stored = jax.lax.stop_gradient(sde.sde_sample(student, sc, x0, eps)["chain"])
    (vjp_val, vjp_chain), _ = jax.vjp(lambda s: (lambda o: (jnp.sum(o["path_kl"]), o["chain"]))(run(s)), sub)
    zero = jax.tree.map(jnp.zeros_like, sub)
    (jvp_val, jvp_chain), _ = jax.jvp(lambda s: (lambda o: (jnp.sum(o["path_kl"]), o["chain"]))(run(s)), (sub,), (zero,))
    out["values"] = {"plain": float(plain["path_kl"][0]), "vjp_primal": float(vjp_val), "jvp_primal": float(jvp_val),
                     "chain_x10_max_abs_vjp_minus_plain": float(jnp.abs(vjp_chain[:, -1] - plain["chain"][:, -1]).max()),
                     "chain_x10_max_abs_jvp_minus_vjp": float(jnp.abs(jvp_chain[:, -1] - vjp_chain[:, -1]).max())}
    print("[a]", json.dumps(out["values"]), flush=True)
    del vjp_chain, jvp_chain, zero

    g = {"full": host(jax.grad(scalar())(sub)), "detached": host(jax.grad(scalar("detached", stored))(sub)),
         "blocked": host(jax.grad(scalar("teacher_blocked"))(sub))}

    def combo(fn):
        return {k: fn(k).astype(np.float64) for k in keys}

    dirs = {"g_full": combo(lambda k: g["full"][k]),
            "full_minus_detached": combo(lambda k: g["full"][k].astype(np.float64) - g["detached"][k]),
            "full_minus_blocked": combo(lambda k: g["full"][k].astype(np.float64) - g["blocked"][k])}
    for seed in range(4):
        r = np.random.default_rng(20000 + seed)
        dirs[f"random_{seed}"] = {k: r.standard_normal(g["full"][k].shape) for k in keys}
    dtypes = {k: v.dtype for k, v in _flat(sub).items()}
    out["directions"] = {}
    for name, d in dirs.items():
        norm = np.sqrt(sum(float(np.sum(a**2)) for a in d.values()))
        flat = {k: jnp.asarray((d[k] / norm).astype(np.float32)).astype(dtypes[k]) for k in keys}
        ref = _flat(sub)
        tang = jax.tree.unflatten(treedef, _ordered(sub, flat, ref))
        cast = {k: np.asarray(flat[k], np.float64) for k in keys}
        dots = {m: sum(float(np.sum(g[m][k].astype(np.float64) * cast[k])) for k in keys) for m in g}
        _, dk = jax.jvp(scalar(), (sub,), (tang,))
        dk = float(dk)
        row = {"forward_dK": dk, "dot_full": dots["full"], "dot_detached": dots["detached"], "dot_blocked": dots["blocked"],
               "abs_diff_full": abs(dk - dots["full"]), "abs_diff_detached": abs(dk - dots["detached"]), "abs_diff_blocked": abs(dk - dots["blocked"]),
               "rel_diff_full": abs(dk - dots["full"]) / abs(dk)}
        out["directions"][name] = row
        print("[b]", name, json.dumps(row), flush=True)
    f = ROOT / "outputs" / "ogpo_port" / "task10_dot_diagnostic.json"
    f.write_text(json.dumps(out, indent=1))
    print("results ->", f, flush=True)


def _ordered(sub, flat, ref):
    """Leaves of `flat` in the tree order of `sub` (matched by array identity of the reference leaves)."""
    by_id = {id(v): k for k, v in ref.items()}
    return [flat[by_id[id(leaf)]] for leaf in jax.tree.leaves(sub)]


if __name__ == "__main__":
    main()
