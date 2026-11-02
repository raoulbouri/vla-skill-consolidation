"""Task 10: gradient plumbing of the pathwise path KL on the real pi05_libero weights. One observation, one trajectory, batch 1.
NOT training: no optimiser, no parameter update, no rollouts, no score-function estimator, no multi-trajectory Monte Carlo.

Opt-in, run in its OWN process (one model copy per process):
  CLVLA_RUN_PI05_TESTS=1 CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy -m pytest tests/ogpo_port/test_pi05_pathwise_gradient.py -v -s
Teacher = the loaded checkpoint, never modified. The student is the frozen Task 8 `make_student` fixture (one perturbed leaf, all
others share the teacher's buffers). Execution is eager, as in Tasks 5-8. No fallback: an out-of-memory failure fails the test.
"""

from __future__ import annotations

import json
import os
import re
import resource
import time

import numpy as np
import pytest

if os.environ.get("CLVLA_RUN_PI05_TESTS") != "1":
    pytest.skip("pi0.5 pathwise gradient tests are opt-in (CLVLA_RUN_PI05_TESTS=1)", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pi05_real_env import ROOT, load_real_env  # noqa: E402
from test_pi05_path_kl import ALPHA, LEAF, PERT_SEED, _flat, _peak_gib, injected, make_student  # noqa: E402

K, SIGMA, DELTA = 10, 0.05, 0.1
OBS_IDS, NOISE_SEED, DIRECTION_SEED, JVP_STEP = (0,), 10000, 10001, 5
EPS32 = float(np.finfo(np.float32).eps)
# fixed before the first real run: bf16 activations round forward and reverse sweeps differently, so this is a plumbing bound
DOT_TEST_REL = 0.05
GROUP_RULES = (("vlm_siglip", r"^PaliGemma/img/"), ("action_expert", r"^PaliGemma/llm/.*_1(/|$)"),
               ("vlm_gemma", r"^PaliGemma/llm/"),
               ("action_heads", r"^(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/"))
TRAINABLE, VLM = ("action_expert", "action_heads"), ("vlm_siglip", "vlm_gemma")
RESULTS = {}


def group_of(path):
    return next((n for n, rx in GROUP_RULES if re.search(rx, path)), "other")


def in_groups(groups):
    from flax import nnx

    return nnx.All(nnx.Param, lambda path, _: group_of("/".join(str(p) for p in path)) in groups)


def split_groups(model, groups):
    from flax import nnx

    graphdef, sub, rest = nnx.split(model, in_groups(groups), ...)
    return (lambda s: nnx.merge(graphdef, s, rest)), sub


def host(grads):
    """Gradient State -> {path: fp32 numpy}. bf16 -> fp32 is exact."""
    return {k: np.asarray(v, np.float32) for k, v in _flat(grads).items()}


def report(g):
    """Per-subtree norms and zero / nonzero leaf counts; individual leaves are NOT required to be nonzero."""
    groups, leaves = {}, {}
    for path, a in g.items():
        d = groups.setdefault(group_of(path), {"leaves": 0, "params": 0, "nonzero_leaves": 0, "zero_leaves": 0, "nonzero_entries": 0,
                                               "finite": True, "sq": 0.0})
        nz = int(np.count_nonzero(a))
        sq = float(np.sum(a.astype(np.float64) ** 2))
        d["leaves"] += 1; d["params"] += a.size; d["nonzero_entries"] += nz; d["sq"] += sq
        d["nonzero_leaves"] += int(nz > 0); d["zero_leaves"] += int(nz == 0); d["finite"] &= bool(np.all(np.isfinite(a)))
        leaves[path] = {"norm": sq**0.5, "nonzero_entries": nz, "params": int(a.size)}
    for d in groups.values():
        d["norm"] = d.pop("sq") ** 0.5
    return {"groups": groups, "leaves": leaves, "zero_leaf_paths": sorted(p for p, v in leaves.items() if v["nonzero_entries"] == 0)}


def compare(a, b):
    """Cosine and relative norm difference of b against a, overall and per group (float64)."""
    def stats(keys):
        dot, na, nb, nd = 0.0, 0.0, 0.0, 0.0
        for k in keys:
            x, y = a[k].astype(np.float64), b[k].astype(np.float64)
            dot += float(np.sum(x * y)); na += float(np.sum(x * x)); nb += float(np.sum(y * y)); nd += float(np.sum((x - y) ** 2))
        return {"cosine": dot / (na * nb) ** 0.5, "rel_norm_diff": (nd / na) ** 0.5, "norm_reference": na**0.5, "norm_control": nb**0.5}

    out = {"overall": stats(sorted(a))}
    for name in TRAINABLE:
        out[name] = stats([k for k in sorted(a) if group_of(k) == name])
    return out


def rss_gib():
    return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20, 2)


def timed(f):
    t0 = time.perf_counter()
    out = f()
    jax.block_until_ready(out)
    return out, time.perf_counter() - t0


@pytest.fixture(scope="module")
def env():
    from flax import nnx

    from clvla.ogpo import pi05_pathwise_kl as pw
    from clvla.ogpo import pi05_path_kl as pkl
    from clvla.ogpo import pi05_sde as sde
    from clvla.ogpo import pi05_velocity as pv

    t0 = time.perf_counter()
    e = load_real_env()
    teacher = e["model"]
    t_flat0 = _flat(nnx.split(teacher)[1])          # references only: no host copy of the parameters
    student, rec = make_student(teacher, ALPHA, PERT_SEED)
    same, _ = make_student(teacher, 0.0, PERT_SEED)
    obs = e["observation"](OBS_IDS)
    x0, eps = injected(NOISE_SEED, len(OBS_IDS))
    s_cache, t_cache, same_cache = pv.build_prefix(student, obs), pv.build_prefix(teacher, obs), pv.build_prefix(same, obs)
    apply_s, sub_s = split_groups(student, TRAINABLE)

    def objective(mode="full", stored=None):
        def f(sub):
            return jnp.sum(pw.pathwise_kl(apply_s(sub), s_cache, teacher, t_cache, x0, eps, mode=mode, stored_chain=stored)["path_kl"])
        return f

    RESULTS["setup"] = {"observation_ids": list(OBS_IDS), "batch": len(OBS_IDS), "K": K, "sigma": SIGMA, "noise_seed": NOISE_SEED,
                        "x0_shape": list(x0.shape), "eps_shape": list(eps.shape), "load_s": time.perf_counter() - t0,
                        "peak_gib_after_load_and_caches": _peak_gib(), "execution": "eager"}
    RESULTS["perturbation"] = rec
    return {"teacher": teacher, "student": student, "same": same, "obs": obs, "x0": x0, "eps": eps, "s_cache": s_cache, "t_cache": t_cache,
            "same_cache": same_cache, "t_flat0": t_flat0, "apply_s": apply_s, "sub_s": sub_s, "objective": objective, "pw": pw, "pkl": pkl,
            "sde": sde, "pv": pv, "grads": {}}


def full_kl(env, mode="full", stored=None):
    return env["pw"].pathwise_kl(env["student"], env["s_cache"], env["teacher"], env["t_cache"], env["x0"], env["eps"], mode=mode, stored_chain=stored)


def stored_chain(env):
    return jax.lax.stop_gradient(env["sde"].sde_sample(env["student"], env["s_cache"], env["x0"], env["eps"])["chain"])


# ---------------------------------------------------------------- 0. setup and the objective's value
def test_0_perturbation_caches_and_objective_value(env):
    from flax import nnx

    rec = RESULTS["perturbation"]
    t_now, s_now = _flat(nnx.split(env["teacher"])[1]), _flat(nnx.split(env["student"])[1])
    differing = [k for k in s_now if s_now[k] is not t_now[k] and not np.array_equal(np.asarray(s_now[k]), np.asarray(t_now[k]))]
    rec.update({"leaves_total": len(s_now), "leaves_differing_from_teacher": differing,
                "leaves_sharing_teacher_buffer": int(sum(s_now[k] is t_now[k] for k in s_now))})
    print(f"\n  [0] perturbation: {json.dumps(rec)}")
    assert differing == [LEAF] and rec["effective_norm"] > 0
    assert all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
               zip(jax.tree.leaves(env["s_cache"].kv_cache), jax.tree.leaves(env["t_cache"].kv_cache)))
    # coefficients: the K = 10 formula
    co = env["pkl"].kl_coefficients()
    s = 1.0 - np.arange(K) / K
    formula = DELTA**2 / (2 * SIGMA**2 * s) * (1 + SIGMA**2 / 2 * (1 - s)) ** 2
    np.testing.assert_allclose(co["weights"], formula, rtol=1e-14)
    # chain = the frozen sampler's chain; value = Task 8's evaluation of the same chain
    out, t_val = timed(lambda: full_kl(env))
    frozen = env["sde"].sde_sample(env["student"], env["s_cache"], env["x0"], env["eps"])
    t8 = env["pkl"].path_kl(env["teacher"], env["t_cache"], env["student"], env["s_cache"], frozen["chain"])
    rel = abs(float(out["path_kl"][0]) - float(t8["path_kl_velocity"][0])) / float(t8["path_kl_velocity"][0])
    RESULTS["objective"] = {"path_kl": float(out["path_kl"][0]), "per_step": np.asarray(out["kl"][0]).tolist(), "weights": co["weights"].tolist(),
                            "task8_path_kl_velocity": float(t8["path_kl_velocity"][0]), "task8_path_kl_direct": float(t8["path_kl_direct"][0]),
                            "rel_diff_vs_task8": rel, "forward_only_s": t_val}
    print(f"\n  [0] objective: {json.dumps(RESULTS['objective'])}")
    assert np.array_equal(np.asarray(out["chain"]), np.asarray(frozen["chain"]))
    assert np.array_equal(np.asarray(out["velocities_student"]), np.asarray(frozen["velocities"]))
    assert rel <= 16 * EPS32 and float(out["path_kl"][0]) > 0 and np.isfinite(float(out["path_kl"][0]))


# ---------------------------------------------------------------- 1. the full gradient (first gradient: its peak is the estimator's own)
def test_1_student_gradient_reaches_expert_and_heads(env):
    (val, g), t_grad = timed(lambda: jax.value_and_grad(env["objective"]())(env["sub_s"]))
    RESULTS["memory_after_full_gradient"] = {"peak_gib_gpu": _peak_gib(), "peak_rss_gib": rss_gib()}
    env["grads"]["full"] = host(g)
    rep = report(env["grads"]["full"])
    RESULTS["student_gradient"] = {"objective": float(val), "value_and_grad_s": t_grad, **rep}
    print(f"\n  [1] value {float(val)}  time {t_grad:.1f} s  {RESULTS['memory_after_full_gradient']}")
    for name, d in rep["groups"].items():
        print(f"      {name:15s} leaves {d['leaves']:3d}  nonzero {d['nonzero_leaves']:3d}  zero {d['zero_leaves']:3d}  params {d['params']:>11d}  "
              f"nonzero entries {d['nonzero_entries']:>11d}  finite {d['finite']}  |g| {d['norm']:.4e}")
    print(f"      zero leaves: {rep['zero_leaf_paths']}")
    assert set(rep["groups"]) == set(TRAINABLE)
    for name in TRAINABLE:
        assert rep["groups"][name]["finite"] and rep["groups"][name]["norm"] > 0 and rep["groups"][name]["nonzero_leaves"] > 0
    assert np.isfinite(float(val))
    assert any(rep["leaves"][p]["norm"] > 0 for p in rep["leaves"] if p.startswith("action_out_proj/"))
    assert any(rep["leaves"][p]["norm"] > 0 for p in rep["leaves"] if p.startswith("action_in_proj/"))


# ---------------------------------------------------------------- 2. student VLM
def test_2_student_vlm_gradient_is_exactly_zero(env):
    apply_v, sub_v = split_groups(env["student"], VLM)

    def f(sub):
        return jnp.sum(env["pw"].pathwise_kl(apply_v(sub), env["s_cache"], env["teacher"], env["t_cache"], env["x0"], env["eps"])["path_kl"])

    g = host(jax.grad(f)(sub_v))
    rep = report(g)
    RESULTS["student_vlm"] = {"groups": rep["groups"], "max_abs": float(max(np.abs(a).max() for a in g.values())), "leaves_checked": len(g)}
    print(f"\n  [2] student VLM: leaves {len(g)}, max |g| {RESULTS['student_vlm']['max_abs']}, {rep['groups']}")
    assert set(rep["groups"]) == set(VLM) and RESULTS["student_vlm"]["max_abs"] == 0.0


# ---------------------------------------------------------------- 3. teacher parameters: frozen, and the freeze is what makes them zero
def test_3_teacher_parameters_frozen(env):
    from flax import nnx

    teacher = env["teacher"]
    graphdef, params, rest = nnx.split(teacher, nnx.Param, ...)

    def frozen(p):
        return jnp.sum(env["pw"].pathwise_kl(env["student"], env["s_cache"], env["pw"].frozen_view(nnx.merge(graphdef, p, rest)), env["t_cache"], env["x0"], env["eps"])["path_kl"])

    g = host(jax.grad(frozen)(params))
    max_frozen = float(max(np.abs(a).max() for a in g.values()))
    del g
    # control: the same objective WITHOUT the freeze gives the teacher's expert/heads a nonzero gradient
    apply_t, sub_t = split_groups(teacher, TRAINABLE)

    def unfrozen(p):
        return jnp.sum(env["pw"].pathwise_kl(env["student"], env["s_cache"], apply_t(p), env["t_cache"], env["x0"], env["eps"])["path_kl"])

    ctrl = report(host(jax.grad(unfrozen)(sub_t)))
    t_now = _flat(nnx.split(teacher)[1])
    unchanged = all(t_now[k] is v for k, v in env["t_flat0"].items())
    RESULTS["teacher"] = {"max_abs_grad_frozen": max_frozen, "control_unfrozen_groups": ctrl["groups"], "teacher_buffers_unchanged": unchanged}
    print(f"\n  [3] teacher: frozen max |g| {max_frozen}; unfrozen control {ctrl['groups']}; buffers unchanged {unchanged}")
    assert max_frozen == 0.0 and unchanged
    assert all(ctrl["groups"][n]["norm"] > 0 for n in TRAINABLE)


# ---------------------------------------------------------------- 4 (control C). teacher state-input derivative
def test_4_teacher_state_input_jvp(env):
    chain = np.asarray(stored_chain(env))
    x, s = jnp.asarray(chain[:, JVP_STEP]), 1.0 - JVP_STEP / K
    r = np.random.default_rng(DIRECTION_SEED)
    u = r.standard_normal(x.shape).astype(np.float32)
    u = jnp.asarray(u / np.linalg.norm(u))
    v, jv = jax.jvp(lambda y: env["pv"].velocity(env["teacher"], env["t_cache"], y, s), (x,), (u,))
    fd = {}
    for h in (1e-1, 1e-2, 1e-3):        # report only: bf16 activations round the velocity, so small h shows quantisation
        plus, minus = (env["pv"].velocity(env["teacher"], env["t_cache"], x + sg * h * u, s) for sg in (1.0, -1.0))
        d = (np.asarray(plus, np.float64) - np.asarray(minus, np.float64)) / (2 * h)
        fd[str(h)] = float(np.linalg.norm(d - np.asarray(jv, np.float64)) / np.linalg.norm(np.asarray(jv, np.float64)))
    RESULTS["teacher_input_jvp"] = {"step": JVP_STEP, "s": s, "state_norm": float(jnp.linalg.norm(x)), "direction_norm": float(jnp.linalg.norm(u)),
                                    "velocity_norm": float(jnp.linalg.norm(v)), "jvp_norm": float(jnp.linalg.norm(jv)), "jvp_max_abs": float(jnp.abs(jv).max()),
                                    "finite": bool(jnp.all(jnp.isfinite(jv))), "fd_rel_err_report_only": fd}
    print(f"\n  [4] {json.dumps(RESULTS['teacher_input_jvp'])}")
    assert RESULTS["teacher_input_jvp"]["finite"] and RESULTS["teacher_input_jvp"]["jvp_norm"] > 0


# ---------------------------------------------------------------- 5. the rollout is attached; independent forward-mode check of the reverse gradient
def test_5_states_differentiable_through_the_rollout(env):
    sub = env["sub_s"]
    leaves, treedef = jax.tree.flatten(sub)
    r = np.random.default_rng(DIRECTION_SEED + 1)
    raw = [r.standard_normal(np.shape(a)).astype(np.float32) for a in leaves]
    norm = np.sqrt(sum(float(np.sum(a.astype(np.float64) ** 2)) for a in raw))
    tang = [jnp.asarray(a / norm).astype(l.dtype) for a, l in zip(raw, leaves)]
    del raw

    def f(s):
        o = env["pw"].pathwise_kl(env["apply_s"](s), env["s_cache"], env["teacher"], env["t_cache"], env["x0"], env["eps"])
        return o["chain"], jnp.sum(o["path_kl"])

    (chain, val), (dchain, dval) = jax.jvp(f, (sub,), (jax.tree.unflatten(treedef, tang),))
    per_step = np.linalg.norm(np.asarray(dchain, np.float64).reshape(K + 1, -1), axis=1)
    flat_t = _flat(jax.tree.unflatten(treedef, tang))
    dot = sum(float(np.sum(env["grads"]["full"][k].astype(np.float64) * np.asarray(v, np.float64))) for k, v in flat_t.items())
    rel = abs(dot - float(dval)) / abs(float(dval))
    RESULTS["rollout"] = {"tangent_norm_per_state_x0_to_x10": per_step.tolist(), "finite": bool(np.all(np.isfinite(np.asarray(dchain)))),
                          "reverse_dot_direction": dot, "forward_directional": float(dval), "rel_diff": rel, "bound": DOT_TEST_REL}
    print(f"\n  [5] {json.dumps(RESULTS['rollout'])}")
    assert per_step[0] == 0.0                                   # x_0 is injected noise
    assert np.all(per_step[1:] > 0) and RESULTS["rollout"]["finite"]
    assert rel <= DOT_TEST_REL


# ---------------------------------------------------------------- 6 (control A). detached student states
def test_6_control_detached_student_states(env):
    stored = stored_chain(env)
    g, t_grad = timed(lambda: jax.grad(env["objective"]("detached", stored))(env["sub_s"]))
    env["grads"]["detached"] = host(g)
    cmp = compare(env["grads"]["full"], env["grads"]["detached"])
    RESULTS["control_detached_states"] = {**cmp, "grad_s": t_grad}
    print(f"\n  [6] detached states (control, NOT the estimator): {json.dumps(RESULTS['control_detached_states'])}")
    assert all(np.all(np.isfinite(a)) for a in env["grads"]["detached"].values())
    assert cmp["overall"]["rel_norm_diff"] > 0


# ---------------------------------------------------------------- 7 (control B). detached teacher input
def test_7_control_detached_teacher_input(env):
    g, t_grad = timed(lambda: jax.grad(env["objective"]("teacher_blocked"))(env["sub_s"]))
    env["grads"]["blocked"] = host(g)
    cmp = compare(env["grads"]["full"], env["grads"]["blocked"])
    RESULTS["control_detached_teacher_input"] = {**cmp, "grad_s": t_grad}
    print(f"\n  [7] teacher input stopped (control, NOT the estimator): {json.dumps(RESULTS['control_detached_teacher_input'])}")
    assert all(np.all(np.isfinite(a)) for a in env["grads"]["blocked"].values())
    assert cmp["overall"]["rel_norm_diff"] > 0


# ---------------------------------------------------------------- 8. determinism in eager mode, and the identical-student zero
def test_8_repeated_execution_is_deterministic(env):
    x0, eps = injected(NOISE_SEED, len(OBS_IDS))                            # the injected noise was not consumed or altered
    val, g = jax.value_and_grad(env["objective"]())(env["sub_s"])
    again = host(g)
    equal = all(np.array_equal(again[k], env["grads"]["full"][k]) for k in again)
    val_equal = float(val) == RESULTS["student_gradient"]["objective"]
    max_rel = max(float(np.abs(again[k] - env["grads"]["full"][k]).max()) for k in again) / max(float(np.abs(a).max()) for a in again.values())
    RESULTS["determinism"] = {"mode": "eager", "value_bitwise": val_equal, "gradient_bitwise": equal, "gradient_max_abs_diff_over_max_abs": max_rel}
    print(f"\n  [8] {RESULTS['determinism']}")
    assert val_equal and equal
    assert np.array_equal(x0, env["x0"]) and np.array_equal(eps, env["eps"])


def test_8b_identical_student_is_a_stationary_zero(env):
    apply_i, sub_i = split_groups(env["same"], TRAINABLE)

    def f(sub):
        return jnp.sum(env["pw"].pathwise_kl(apply_i(sub), env["same_cache"], env["teacher"], env["t_cache"], env["x0"], env["eps"])["path_kl"])

    val, g = jax.value_and_grad(f)(sub_i)
    max_abs = float(max(np.abs(a).max() for a in host(g).values()))
    RESULTS["identical_student"] = {"objective": float(val), "max_abs_grad": max_abs}
    print(f"\n  [8b] {RESULTS['identical_student']}")
    assert float(val) == 0.0 and max_abs == 0.0


def test_zz_write_results():
    RESULTS["memory_final"] = {"peak_gib_gpu": _peak_gib(), "peak_rss_gib": rss_gib()}
    out = ROOT / "outputs" / "ogpo_port"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "task10_pi05_pathwise_gradient.json"
    f.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\n  {RESULTS['memory_final']}; results -> {f}")
