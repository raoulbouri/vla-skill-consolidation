"""Task 11 follow-up to task11_real.py: (1) fresh seeds 40000+ to test the obs 0 ratio-KL gap (paired z -2.60) on all three observations, limit |z| < 3 fixed before the run; (2) routing, one parameter group per derivative, after the two-group derivatives ran out of memory.
  CLVLA_USE_GPU=1 CLVLA_GPU=1 PYTHONPATH=tests/ogpo_port scripts/py-policy tests/ogpo_port/task11_followup.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("TAG", "task11")
import task10b as T

import jax
import jax.numpy as jnp
import numpy as np
from clvla.ogpo import pi05_path_logprob as plp
from clvla.ogpo import pi05_sde as sde
from flax import nnx

pv = T.pv
OUT = Path(__file__).resolve().parents[2] / "outputs" / "ogpo_port" / "task11"
N = 2048
cfg = sde.SDEConfig()
stds = np.asarray(cfg.stds(), np.float64)
RES = {"limit_abs_z": 3.0, "n": N, "ratio": {}, "routing": {}}


def save():
    (OUT / "followup.json").write_text(json.dumps(RES, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


H = T.helpers()
flat = H["_flat"]
e = H["load_real_env"]()
teacher = e["model"]
student, _ = H["make_student"](teacher, H["ALPHA"], H["PERT_SEED"])
in_g = H["in_groups"]
gd_s, theta, vlm_s, other_s = nnx.split(student, in_g(H["TRAINABLE"]), in_g(H["VLM"]), ...)
gd_t, theta_t, vlm_t, other_t = nnx.split(teacher, in_g(H["TRAINABLE"]), in_g(H["VLM"]), ...)
data = {}
for o in (0, 1, 2):
    sc, tc = pv.build_prefix(student, e["observation"]((o,))), pv.build_prefix(teacher, e["observation"]((o,)))
    data[o] = (sc.observation, (sc.prefix_mask, sc.kv_cache), (tc.prefix_mask, tc.kv_cache))
del student, teacher
e.clear()


def run(a, b, c, x0=None, eps=None, d=None, chain=None):
    m, mt = nnx.merge(gd_s, a[0], a[1], a[2]), nnx.merge(gd_t, b[0], b[1], b[2])
    obs, s_c, t_c = d
    if chain is None:
        chain = sde.sde_sample(m, pv.PrefixCache(obs, *s_c), x0, eps, cfg)["chain"]
    p, q = plp.path_log_prob(m, pv.PrefixCache(obs, *s_c), chain, cfg), plp.path_log_prob(mt, pv.PrefixCache(obs, *t_c), chain, cfg)
    return chain, p, q


@jax.jit
def rollout(a, b, x0, eps, d):
    chain, p, q = run(a, b, None, x0, eps, d)
    return chain, p["means"], q["means"]


A, B = (theta, vlm_s, other_s), (theta_t, vlm_t, other_t)
for o, d in data.items():
    ell, K = [], []
    for i in range(N):
        s = 40000 + 3000 * o + i
        x0, eps = H["injected"](s, 1)
        chain, ms, mt = rollout(A, B, jnp.asarray(x0), jnp.asarray(eps), d)
        y, a, b = np.asarray(chain[0, 1:], np.float64), np.asarray(ms[0], np.float64), np.asarray(mt[0], np.float64)
        ell.append(float(np.sum((np.sum((y - b) ** 2, axis=(1, 2)) - np.sum((y - a) ** 2, axis=(1, 2))) / (2 * stds**2))))
        K.append(float(np.sum(np.sum((a - b) ** 2, axis=(1, 2)) / (2 * stds**2))))
    ell, K = np.array(ell), np.array(K)
    diff = ell - K
    RES["ratio"][str(o)] = {"ell": float(ell.mean()), "ell_se": float(ell.std(ddof=1) / N**0.5), "K": float(K.mean()), "K_se": float(K.std(ddof=1) / N**0.5), "diff": float(diff.mean()), "diff_se": float(diff.std(ddof=1) / N**0.5), "z": float(diff.mean() / (diff.std(ddof=1) / N**0.5))}
    print(f"[ratio {o}]", json.dumps(RES["ratio"][str(o)]), flush=True)
    save()
RES["ratio_pass"] = all(abs(v["z"]) < 3.0 for v in RES["ratio"].values())
x0, eps = H["injected"](49999, 1)
chain0, _, _ = rollout(A, B, jnp.asarray(x0), jnp.asarray(eps), data[0])
jax.clear_caches()


def loss(a, b, d, chain):
    _, p, q = run(a, b, None, d=d, chain=chain)
    return jnp.sum(jax.lax.stop_gradient(p["total"] - q["total"]) * p["total"])


def group(name, fn):
    try:
        g = jax.jit(jax.grad(fn))(*ARGS[name])
        n2, nz, sz, fin = 0.0, 0, 0, True
        for v in flat(g).values():
            if jnp.issubdtype(v.dtype, jnp.floating):
                v = v.astype(jnp.float32)
                n2 += float(jnp.sum(v**2))
                nz += int(jnp.count_nonzero(v))
                sz += int(v.size)
                fin = fin and bool(jnp.isfinite(v).all())
        RES["routing"][name] = {"norm": n2**0.5, "nonzero": nz, "size": sz, "finite": fin}
        del g
    except Exception as err:  # noqa: BLE001 - reported, not hidden
        RES["routing"][name] = {"error": f"{type(err).__name__}: {str(err)[:200]}"}
    print(f"[routing {name}]", json.dumps(RES["routing"][name]), flush=True)
    save()
    jax.clear_caches()


ARGS = {k: (None,) for k in ()}
ARGS = {"student_trainable": (theta,), "student_vlm": (vlm_s,), "teacher_trainable": (theta_t,), "teacher_vlm": (vlm_t,)}
group("student_trainable", lambda t: loss((t, vlm_s, other_s), B, data[0], chain0))
group("student_vlm", lambda t: loss((theta, t, other_s), B, data[0], chain0))
group("teacher_trainable", lambda t: loss(A, (t, vlm_t, other_t), data[0], chain0))
group("teacher_vlm", lambda t: loss(A, (theta_t, t, other_t), data[0], chain0))
print("[end]", H["_peak_gib"](), flush=True)
