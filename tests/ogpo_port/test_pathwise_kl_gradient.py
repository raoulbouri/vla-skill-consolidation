"""Task 9: synthetic verification of the reparameterised PATHWISE gradient of the expected forward path KL (CPU, float64).
No pi0.5, no optimiser, no training, no score-function estimator. Task 8's gradient-stopped `path_kl` is not used.

Objective, in the conventions of Tasks 6-8 (s_k = 1 - k/K, Delta = 1/K, sigma_k = sigma sqrt(s_k), score correction, shared covariance):

    J(theta) = E_eps[ K_theta(X_theta) ],   K_theta(X) = sum_{k=0}^{K-1} w_k || v_S(x_k, s_k; theta) - v_T(x_k, s_k) ||^2
    x_0 = eps_0,   x_{k+1} = mu_S(x_k, s_k; theta) + sigma_k eps_{k+1},   mu = x - Delta (v + sigma^2/2 (x + (1 - s) v))

and the estimator is grad J_M with J_M the average over a FIXED noise batch, differentiating through every student state and
through the teacher's state input, with the teacher's parameters frozen:

    d/dtheta K = sum_k 2 w_k dv_k^T [ d_theta v_S + (J_S,k - J_T,k) dx_k/dtheta ],   dv_k = v_S,k - v_T,k,   J = dv/dx.

Modes: "full" (the estimator); "detached" (states treated as fixed, omits the (J_S - J_T) dx/dtheta term: a control, NOT the
desired estimator); "teacher_blocked" (teacher state input stopped, omits J_T dx/dtheta: a control).
"""

from __future__ import annotations

import itertools
import json
import math
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64
from jax.flatten_util import ravel_pytree

from clvla.ogpo.pi05_path_kl import kl_coefficients
from clvla.ogpo.pi05_sde import SDEConfig

ROOT = Path(__file__).resolve().parents[2]
K, D, O, SIGMA, M = 4, 3, 2, 0.05, 64
STEPS = tuple(10.0**-e for e in range(1, 10))
RESULTS = {}
T0 = time.perf_counter()


@pytest.fixture(scope="module", autouse=True)
def float64():
    with enable_x64():
        yield


# ---------------------------------------------------------------- objective
def schedule(num_steps=K, sigma=SIGMA):
    co = kl_coefficients(SDEConfig(num_steps=num_steps, sigma=sigma))       # frozen conventions, float64
    return co["times"], co["stds"], co["weights"]


def one_path(theta, teacher, eps, velocity, num_steps, sigma, mode):
    """Per-step KL (K,) and states (K, D) of one trajectory; eps (K+1, D) is fixed data."""
    times, stds, weights = schedule(num_steps, sigma)
    teacher = jax.lax.stop_gradient(teacher)                                # frozen parameters, differentiable state input
    x, kls, states = eps[0], [], []
    for k in range(num_steps):
        s = times[k]
        xk = jax.lax.stop_gradient(x) if mode == "detached" else x
        xt = jax.lax.stop_gradient(xk) if mode == "teacher_blocked" else xk
        vs, vt = velocity(xk, s, theta), velocity(xt, s, teacher)
        kls.append(weights[k] * jnp.sum((vs - vt) ** 2))
        states.append(xk)
        mu = xk - (vs + 0.5 * sigma**2 * (xk + (1.0 - s) * vs)) / num_steps
        x = mu + stds[k] * eps[k + 1]
    return jnp.stack(kls), jnp.stack(states)


def make_objective(velocity, num_steps=K, sigma=SIGMA, mode="full"):
    def obj(theta, teacher, eps):
        kl = jax.vmap(lambda e: one_path(theta, teacher, e, velocity, num_steps, sigma, mode)[0].sum())(eps)
        return kl.mean()

    return jax.jit(obj)


# ---------------------------------------------------------------- nonlinear synthetic model
OBS = np.array([0.7, -0.4])


def nonlinear_v(x, s, p):
    """v(x, s; p) = tanh(A x + B o + c) (1 + a s^2) + d sin(3 s) x."""
    return jnp.tanh(p["A"] @ x + p["B"] @ jnp.asarray(OBS) + p["c"]) * (1.0 + p["a"] * s**2) + p["d"] * jnp.sin(3.0 * s) * x


def nonlinear_params(seed):
    r = np.random.default_rng(seed)
    return {"A": r.normal(size=(D, D)), "B": r.normal(size=(D, O)), "c": r.normal(size=D),
            "a": np.array(r.normal()), "d": np.array(r.normal())}


def perturbed(p, scale, seed):
    r = np.random.default_rng(seed)
    return {k: v + scale * r.normal(size=np.shape(v)) for k, v in p.items()}


@pytest.fixture(scope="module")
def nl():
    teacher = jax.tree.map(jnp.asarray, nonlinear_params(100))
    student = jax.tree.map(jnp.asarray, perturbed(nonlinear_params(100), 0.3, 101))
    eps = jnp.asarray(np.random.default_rng(102).standard_normal((M, K + 1, D)))
    objs = {m: make_objective(nonlinear_v, mode=m) for m in ("full", "detached", "teacher_blocked")}
    grads = {m: jax.jit(jax.grad(f)) for m, f in objs.items()}
    flat, unravel = ravel_pytree(student)
    us = [(lambda v: v / np.linalg.norm(v))(np.random.default_rng(200 + i).standard_normal(flat.shape)) for i in range(3)]
    return {"teacher": teacher, "student": student, "eps": eps, "objs": objs, "grads": grads, "flat": flat,
            "unravel": unravel, "us": us}


def directional(g, u):
    return float(np.dot(np.asarray(ravel_pytree(g)[0]), u))


def central(obj, nl, u, h):
    plus = obj(nl["unravel"](nl["flat"] + h * u), nl["teacher"], nl["eps"])
    minus = obj(nl["unravel"](nl["flat"] - h * u), nl["teacher"], nl["eps"])
    return float((plus - minus) / (2.0 * h))


# ---------------------------------------------------------------- 0. objective = the Task 8 conditional KL conventions
def test_0_objective_matches_direct_gaussian_kl_and_frozen_schedule(nl):
    times, stds, weights = schedule()
    np.testing.assert_allclose(times, [1.0, 0.75, 0.5, 0.25])
    np.testing.assert_allclose(stds, SIGMA * np.sqrt(times))
    for e in np.asarray(nl["eps"])[:8]:
        kl, xs = one_path(nl["student"], nl["teacher"], jnp.asarray(e), nonlinear_v, K, SIGMA, "full")
        for k in range(K):
            s, x = times[k], xs[k]
            vs, vt = nonlinear_v(x, s, nl["student"]), nonlinear_v(x, s, nl["teacher"])
            ms = x - (1.0 / K) * ((1 + SIGMA**2 * (1 - s) / 2) * vs + SIGMA**2 / 2 * x)         # expanded form (Task 8)
            mt = x - (1.0 / K) * ((1 + SIGMA**2 * (1 - s) / 2) * vt + SIGMA**2 / 2 * x)
            direct = float(jnp.sum((ms - mt) ** 2) / (2 * stds[k] ** 2))
            assert abs(direct - float(kl[k])) <= 1e-12 * abs(direct)
    # the state update really is mu + sigma_k eps_{k+1}
    e = nl["eps"][0]
    _, xs = one_path(nl["student"], nl["teacher"], e, nonlinear_v, K, SIGMA, "full")
    v0 = nonlinear_v(xs[0], times[0], nl["student"])
    mu0 = xs[0] - (v0 + SIGMA**2 / 2 * (xs[0] + (1 - times[0]) * v0)) / K
    np.testing.assert_allclose(xs[1], mu0 + stds[0] * e[1], rtol=1e-14)
    RESULTS["objective_value"] = float(nl["objs"]["full"](nl["student"], nl["teacher"], nl["eps"]))
    assert RESULTS["objective_value"] > 0


# ---------------------------------------------------------------- 1. directional derivative vs central differences
def test_1_directional_derivative_vs_central_difference(nl):
    g = nl["grads"]["full"](nl["student"], nl["teacher"], nl["eps"])
    table = []
    for i, u in enumerate(nl["us"]):
        ad = directional(g, u)
        rows = [{"h": h, "fd": central(nl["objs"]["full"], nl, u, h)} for h in STEPS]
        for r in rows:
            r["abs_err"] = abs(r["fd"] - ad)
            r["rel_err"] = r["abs_err"] / abs(ad)
        conv = [r for r in rows if 1e-3 - 1e-12 <= r["h"] <= 1e-1 + 1e-12]
        slope = float(np.polyfit(np.log10([r["h"] for r in conv]), np.log10([r["abs_err"] for r in conv]), 1)[0])
        best = min(rows, key=lambda r: r["rel_err"])
        table.append({"direction": i, "autodiff": ad, "rows": rows, "slope_h_1e-1_to_1e-3": slope,
                      "best_h": best["h"], "best_rel_err": best["rel_err"]})
        print(f"\n  [1] u{i}: AD {ad:.12e}, slope {slope:.3f}, best h {best['h']:.0e} rel err {best['rel_err']:.2e}")
        for r in rows:
            print(f"      h {r['h']:.0e}  FD {r['fd']:.12e}  rel err {r['rel_err']:.2e}")
        assert best["rel_err"] <= 1e-8 and 1.8 <= slope <= 2.2
    RESULTS["directional"] = table


def test_1b_every_parameter_coordinate(nl):
    """Full gradient vector, coordinate by coordinate, against central differences at h = 1e-5."""
    ad = np.asarray(ravel_pytree(nl["grads"]["full"](nl["student"], nl["teacher"], nl["eps"]))[0])
    fd = np.array([central(nl["objs"]["full"], nl, e, 1e-5) for e in np.eye(ad.size)])
    err = np.abs(ad - fd)
    RESULTS["coordinatewise"] = {"n_params": int(ad.size), "max_abs_err": float(err.max()), "grad_norm": float(np.linalg.norm(ad)),
                                 "rel_err_vector": float(np.linalg.norm(ad - fd) / np.linalg.norm(ad))}
    print(f"\n  [1b] {RESULTS['coordinatewise']}")
    assert RESULTS["coordinatewise"]["rel_err_vector"] <= 1e-7


# ---------------------------------------------------------------- 2. full vs detached
def state_term(nl, u):
    """Independent reconstruction of full - detached along u: sum_k d_x K_k . dx_k/dtheta[u], split into student and teacher Jacobians."""
    _, _, weights = schedule()
    times = schedule()[0]
    student, teacher = nl["student"], nl["teacher"]
    tot_s, tot_t = 0.0, 0.0
    for e in nl["eps"]:
        chain = lambda f: one_path(nl["unravel"](f), teacher, e, nonlinear_v, K, SIGMA, "full")[1]  # noqa: E731
        xs, dxs = jax.jvp(chain, (nl["flat"],), (jnp.asarray(u),))
        for k in range(K):
            s, x = times[k], xs[k]
            dv = nonlinear_v(x, s, student) - nonlinear_v(x, s, teacher)
            js = jax.jacfwd(nonlinear_v)(x, s, student)
            jt = jax.jacfwd(nonlinear_v)(x, s, teacher)
            tot_s += float(2 * weights[k] * dv @ js @ dxs[k])
            tot_t -= float(2 * weights[k] * dv @ jt @ dxs[k])
    return tot_s / M, tot_t / M


def test_2_full_vs_detached(nl):
    gf = np.asarray(ravel_pytree(nl["grads"]["full"](nl["student"], nl["teacher"], nl["eps"]))[0])
    gd = np.asarray(ravel_pytree(nl["grads"]["detached"](nl["student"], nl["teacher"], nl["eps"]))[0])
    cos = float(gf @ gd / (np.linalg.norm(gf) * np.linalg.norm(gd)))
    rel = float(np.linalg.norm(gf - gd) / np.linalg.norm(gf))
    rows = []
    for i, u in enumerate(nl["us"]):
        fd_full = central(nl["objs"]["full"], nl, u, 1e-5)
        ad_full, ad_det = float(gf @ u), float(gd @ u)
        fd_err = abs(fd_full - ad_full)
        term_s, term_t = state_term(nl, u)
        recon_err = abs((ad_full - ad_det) - (term_s + term_t))
        rows.append({"direction": i, "fd_full": fd_full, "ad_full": ad_full, "ad_detached": ad_det, "fd_vs_full": fd_err,
                     "fd_vs_detached": abs(fd_full - ad_det), "state_term_student_jacobian": term_s,
                     "state_term_teacher_jacobian": term_t, "state_term_reconstruction_err": recon_err})
        assert abs(fd_full - ad_det) >= 100 * max(fd_err, 1e-12)            # FD identifies full, rejects detached
        assert recon_err <= 1e-10 * max(abs(ad_full), abs(ad_det))
    # the detached gradient IS the derivative with the stored chain held fixed
    fixed = [one_path(nl["student"], nl["teacher"], e, nonlinear_v, K, SIGMA, "full")[1] for e in nl["eps"]]
    fixed = jnp.stack(fixed)
    times, _, weights = schedule()

    def fixed_obj(theta):
        tot = 0.0
        for k in range(K):
            dv = jax.vmap(lambda x, s=times[k]: nonlinear_v(x, s, theta) - nonlinear_v(x, s, nl["teacher"]))(fixed[:, k])
            tot = tot + weights[k] * jnp.mean(jnp.sum(dv**2, axis=-1))
        return tot

    u = nl["us"][0]
    h = 1e-5
    fd_fixed = float((fixed_obj(nl["unravel"](nl["flat"] + h * u)) - fixed_obj(nl["unravel"](nl["flat"] - h * u))) / (2 * h))
    RESULTS["full_vs_detached"] = {"cosine": cos, "rel_diff_norm": rel, "norm_full": float(np.linalg.norm(gf)),
                                   "norm_detached": float(np.linalg.norm(gd)), "directions": rows,
                                   "fd_fixed_chain_u0": fd_fixed, "detached_u0": float(gd @ u)}
    print(f"\n  [2] {json.dumps(RESULTS['full_vs_detached'])}")
    assert abs(fd_fixed - float(gd @ u)) <= 1e-7 * abs(float(gd @ u))
    assert rel > 0.01


# ---------------------------------------------------------------- 3. teacher
def test_3_teacher_frozen_but_input_differentiable(nl):
    student, teacher, eps = nl["student"], nl["teacher"], nl["eps"]
    before = ravel_pytree(teacher)[0]
    obj = nl["objs"]["full"]
    gt = jax.grad(obj, argnums=1)(student, teacher, eps)
    max_gt = float(jnp.abs(ravel_pytree(gt)[0]).max())
    # control: J really does depend on the teacher's values, so the zero above comes from the freeze, not from the model
    t_flat, t_unravel = ravel_pytree(teacher)
    u = nl["us"][0]
    dj_dteacher = float((obj(student, t_unravel(t_flat + 1e-5 * u), eps) - obj(student, t_unravel(t_flat - 1e-5 * u), eps)) / 2e-5)
    x, s = eps[0, 0], schedule()[0][1]
    jac = np.asarray(jax.jacfwd(nonlinear_v)(x, s, teacher))
    gf = np.asarray(ravel_pytree(nl["grads"]["full"](student, teacher, eps))[0])
    gb = np.asarray(ravel_pytree(nl["grads"]["teacher_blocked"](student, teacher, eps))[0])
    rows = []
    for i, v in enumerate(nl["us"]):
        fd = central(obj, nl, v, 1e-5)
        term_t = state_term(nl, v)[1]
        rows.append({"direction": i, "fd": fd, "ad_full": float(gf @ v), "ad_teacher_blocked": float(gb @ v),
                     "full_minus_blocked": float((gf - gb) @ v), "teacher_jacobian_term": term_t})
        assert abs(fd - float(gf @ v)) <= 1e-7 * abs(fd)
        assert abs(fd - float(gb @ v)) >= 100 * abs(fd - float(gf @ v))
        assert abs(float((gf - gb) @ v) - term_t) <= 1e-10 * max(abs(term_t), 1.0)
    RESULTS["teacher"] = {"max_abs_grad_wrt_teacher_params": max_gt, "dJ_dteacher_values_fd_u0": dj_dteacher,
                          "teacher_jacobian_frobenius": float(np.linalg.norm(jac)),
                          "rel_grad_loss_if_teacher_input_blocked": float(np.linalg.norm(gf - gb) / np.linalg.norm(gf)),
                          "directions": rows, "teacher_unchanged": bool(jnp.array_equal(before, ravel_pytree(teacher)[0]))}
    print(f"\n  [3] {json.dumps(RESULTS['teacher'])}")
    assert max_gt == 0.0 and abs(dj_dteacher) > 1e-3
    assert np.linalg.norm(jac) > 1e-3 and RESULTS["teacher"]["teacher_unchanged"]


def test_3b_identical_parameters_are_a_stationary_zero(nl):
    """Eager: exactly zero (dv = 0 bitwise). Under jit(grad) the two velocities are fused differently, so dv is zero only to
    rounding: require <= 1e-13 of the perturbed student's gradient norm."""
    te, eps = nl["teacher"], nl["eps"]

    def raw(a, b):
        return jax.vmap(lambda e: one_path(a, b, e, nonlinear_v, K, SIGMA, "full")[0].sum())(eps).mean()

    eager = float(jnp.abs(ravel_pytree(jax.grad(raw)(te, te))[0]).max())
    val = float(nl["objs"]["full"](te, te, eps))
    jitted = float(jnp.abs(ravel_pytree(nl["grads"]["full"](te, te, eps))[0]).max())
    scale = float(jnp.linalg.norm(ravel_pytree(nl["grads"]["full"](nl["student"], te, eps))[0]))
    RESULTS["identical_params"] = {"objective": val, "max_abs_grad_eager": eager, "max_abs_grad_jit": jitted, "reference_grad_norm": scale}
    print(f"\n  [3b] {RESULTS['identical_params']}")
    assert val == 0.0 and eager == 0.0 and jitted <= 1e-13 * scale


# ---------------------------------------------------------------- 4. reproducibility
def test_4_reproducibility(nl):
    obj, g, student, teacher, eps = nl["objs"]["full"], nl["grads"]["full"], nl["student"], nl["teacher"], nl["eps"]
    noise_before = np.asarray(eps).copy()
    assert len({float(obj(student, teacher, eps)) for _ in range(3)}) == 1
    a, b = ravel_pytree(g(student, teacher, eps))[0], ravel_pytree(g(student, teacher, eps))[0]
    assert jnp.array_equal(a, b)
    u = nl["us"][1]
    assert central(obj, nl, u, 1e-5) == central(obj, nl, u, 1e-5)
    # a freshly built, separately compiled objective gives the same numbers
    assert float(make_objective(nonlinear_v)(student, teacher, eps)) == float(obj(student, teacher, eps))
    assert jnp.array_equal(ravel_pytree(jax.jit(jax.grad(make_objective(nonlinear_v)))(student, teacher, eps))[0], a)
    # grad(jit) is a different compiled program from jit(grad): equal to rounding, not bitwise
    other = ravel_pytree(jax.grad(make_objective(nonlinear_v))(student, teacher, eps))[0]
    across = float(jnp.abs(other - a).max() / jnp.abs(a).max())
    assert across <= 1e-14
    assert np.array_equal(noise_before, np.asarray(eps))
    RESULTS["reproducibility"] = {"same_program": "objective, gradient, finite differences bitwise repeatable; fresh compile bitwise",
                                  "grad_of_jit_vs_jit_of_grad_max_rel": across}


# ---------------------------------------------------------------- 5. linear-Gaussian analytic case
LK, LD = 4, 2
LIN_TEACHER = {"alpha": 0.3, "beta": np.array([0.2, -0.1])}
LIN_STUDENT = {"alpha": -0.4, "beta": np.array([0.5, 0.3])}


def linear_v(x, s, p):
    return p["alpha"] * x + p["beta"]


def linear_analytic(st, te, sigma, num_steps=LK):
    """Exact J and gradients by hand (NumPy): per coordinate x_k ~ N(M_k, V_k), m_k = 1 - Delta (g_k alpha + sigma^2/2),
    M_{k+1} = m_k M_k - Delta g_k beta, V_{k+1} = m_k^2 V_k + sigma_k^2, g_k = 1 + sigma^2 (1 - s_k)/2,
    E dv^2 = da^2 (V + M^2) + 2 da db M + db^2 with da = alpha - alpha_T, db = beta - beta_T.
    `explicit` is the gradient with the state distribution held fixed (the detached gradient)."""
    times, stds, weights = schedule(num_steps, sigma)
    delta, a = 1.0 / num_steps, float(st["alpha"])
    da, db = a - float(te["alpha"]), np.asarray(st["beta"], float) - np.asarray(te["beta"], float)
    d = db.size
    Mk, Vk = np.zeros(d), np.ones(d)
    dM_a, dV_a = np.zeros(d), np.zeros(d)
    dM_b = np.zeros(d)                                                      # dM_i / dbeta_i (diagonal)
    J, g_a, g_b, ex_a, ex_b = 0.0, 0.0, np.zeros(d), 0.0, np.zeros(d)
    for k in range(num_steps):
        w = weights[k]
        J += w * np.sum(da**2 * (Vk + Mk**2) + 2 * da * db * Mk + db**2)
        ex_a += w * np.sum(2 * da * (Vk + Mk**2) + 2 * db * Mk)
        ex_b += w * (2 * da * Mk + 2 * db)
        g_a += w * np.sum(2 * da * (Vk + Mk**2) + da**2 * (dV_a + 2 * Mk * dM_a) + 2 * db * Mk + 2 * da * db * dM_a)
        g_b += w * (da**2 * 2 * Mk * dM_b + 2 * da * Mk + 2 * da * db * dM_b + 2 * db)
        gk = 1.0 + 0.5 * sigma**2 * (1.0 - times[k])
        m = 1.0 - delta * (gk * a + 0.5 * sigma**2)
        dm = -delta * gk
        dM_a, dV_a = m * dM_a + dm * Mk, m**2 * dV_a + 2 * m * dm * Vk
        dM_b = m * dM_b - delta * gk
        Mk, Vk = m * Mk - delta * gk * np.asarray(st["beta"], float), m**2 * Vk + stds[k] ** 2
    return {"J": J, "g_alpha": g_a, "g_beta": g_b, "explicit_alpha": ex_a, "explicit_beta": ex_b}


def sign_nodes():
    """All 2^(K D) sign patterns of (eps_0, ..., eps_{K-1}): their first two moments equal those of N(0, I), and K is a QUADRATIC
    polynomial of the noise in the linear model, so the plain average over the nodes is the exact expectation (no Monte Carlo)."""
    nodes = np.array(list(itertools.product((-1.0, 1.0), repeat=LK * LD))).reshape(-1, LK, LD)
    return np.concatenate([nodes, np.zeros((nodes.shape[0], 1, LD))], axis=1)      # eps_K never reaches the objective


@pytest.mark.parametrize("sigma", [0.05, 0.5])
def test_5_linear_gaussian_exact(sigma):
    st, te = jax.tree.map(jnp.asarray, LIN_STUDENT), jax.tree.map(jnp.asarray, LIN_TEACHER)
    an = linear_analytic(LIN_STUDENT, LIN_TEACHER, sigma)
    nodes = jnp.asarray(sign_nodes())
    obj = make_objective(linear_v, LK, sigma)
    val = float(obj(st, te, nodes))
    g = jax.grad(obj)(st, te, nodes)
    gd = jax.grad(make_objective(linear_v, LK, sigma, "detached"))(st, te, nodes)
    errs = {"J": abs(val - an["J"]) / abs(an["J"]), "g_alpha": abs(float(g["alpha"]) - an["g_alpha"]) / abs(an["g_alpha"]),
            "g_beta": float(np.abs(np.asarray(g["beta"]) - an["g_beta"]).max() / np.abs(an["g_beta"]).max()),
            "detached_alpha": abs(float(gd["alpha"]) - an["explicit_alpha"]) / abs(an["explicit_alpha"]),
            "detached_beta": float(np.abs(np.asarray(gd["beta"]) - an["explicit_beta"]).max() / np.abs(an["explicit_beta"]).max())}
    RESULTS[f"linear_exact_sigma{sigma}"] = {
        "J_analytic": an["J"], "J_impl": val, "g_alpha_analytic": an["g_alpha"], "g_alpha_impl": float(g["alpha"]),
        "g_beta_analytic": an["g_beta"].tolist(), "g_beta_impl": np.asarray(g["beta"]).tolist(),
        "detached_alpha_analytic": an["explicit_alpha"], "detached_beta_analytic": an["explicit_beta"].tolist(), "rel_errs": errs}
    print(f"\n  [5] sigma {sigma}: {json.dumps(RESULTS[f'linear_exact_sigma{sigma}'])}")
    assert max(errs.values()) <= 1e-11
    assert abs(an["g_alpha"] - an["explicit_alpha"]) > 1e-3 * abs(an["g_alpha"])   # the state term is really there


@pytest.mark.parametrize("sigma", [0.05, 0.5])
def test_5b_linear_gaussian_monte_carlo(sigma):
    """Fixed-batch estimator on Gaussian noise vs the analytic expectation: z scores over 4 quantities, N = 200000 (seed fixed)."""
    n = 200000
    st, te = jax.tree.map(jnp.asarray, LIN_STUDENT), jax.tree.map(jnp.asarray, LIN_TEACHER)
    an = linear_analytic(LIN_STUDENT, LIN_TEACHER, sigma)
    eps = jnp.asarray(np.random.default_rng(300).standard_normal((n, LK + 1, LD)))

    def single(theta, e):
        return one_path(theta, te, e, linear_v, LK, sigma, "full")[0].sum()

    vals, grads = jax.jit(jax.vmap(jax.value_and_grad(single), (None, 0)))(st, eps)
    stats = {}
    for name, samples, truth in (("J", vals, an["J"]), ("g_alpha", grads["alpha"], an["g_alpha"]),
                                 ("g_beta0", grads["beta"][:, 0], an["g_beta"][0]), ("g_beta1", grads["beta"][:, 1], an["g_beta"][1])):
        a = np.asarray(samples)
        se = float(a.std(ddof=1) / math.sqrt(n))
        stats[name] = {"mc": float(a.mean()), "se": se, "analytic": float(truth), "z": float((a.mean() - truth) / se)}
    RESULTS[f"linear_mc_sigma{sigma}"] = stats
    print(f"\n  [5b] sigma {sigma}: {json.dumps(stats)}")
    assert all(abs(v["z"]) <= 3.5 for v in stats.values())                  # fixed in advance; two-sided 0.05 / 8 quantities (2 sigmas x 4) is about 2.7


# ---------------------------------------------------------------- 5c/5d. audit of the exact linear-Gaussian expectation
def all_sign_nodes():
    """All 2^((K+1) D) = 1024 sign patterns of every noise coordinate (eps_0, ..., eps_K), shape (1024, K+1, D)."""
    return np.array(list(itertools.product((-1.0, 1.0), repeat=(LK + 1) * LD))).reshape(-1, LK + 1, LD)


def _flat_lin(g):
    return np.concatenate([np.atleast_1d(np.asarray(g["alpha"])), np.asarray(g["beta"])])


@pytest.mark.parametrize("sigma", [0.05, 0.5])
def test_5c_full_enumeration_over_all_ten_noise_coordinates(sigma):
    st, te = jax.tree.map(jnp.asarray, LIN_STUDENT), jax.tree.map(jnp.asarray, LIN_TEACHER)
    an = linear_analytic(LIN_STUDENT, LIN_TEACHER, sigma)
    an_g = np.concatenate([[an["g_alpha"]], an["g_beta"]])
    an_d = np.concatenate([[an["explicit_alpha"]], an["explicit_beta"]])
    obj, det = make_objective(linear_v, LK, sigma), make_objective(linear_v, LK, sigma, "detached")
    full, old = all_sign_nodes(), sign_nodes()
    # what the existing 256-pattern test actually enumerates
    inputs = {"noise_shape_per_trajectory": list(full.shape[1:]), "scalar_noise_coordinates": int(full[0].size),
              "patterns_full": int(full.shape[0]), "patterns_existing": int(old.shape[0]),
              "existing_eps0_x0_sampled": bool(np.all(np.abs(old[:, 0]) == 1.0)), "existing_eps1_to_3_sampled": bool(np.all(np.abs(old[:, 1:LK]) == 1.0)),
              "existing_eps4_held_at_zero": bool(np.all(old[:, LK] == 0.0)), "distinct_x0_values_existing": int(np.unique(old[:, 0], axis=0).shape[0])}
    assert inputs["patterns_full"] == 1024 == 2 ** ((LK + 1) * LD) and inputs["patterns_existing"] == 256
    # eps_K moves only x_K, which is not in the objective: per-trajectory KL must be bitwise blind to it
    single = jax.jit(jax.vmap(lambda e: one_path(st, te, e, linear_v, LK, sigma, "full")[0].sum()))
    a = np.asarray(single(jnp.asarray(full)))
    flipped = full.copy()
    flipped[:, LK] *= -1.0
    assert np.array_equal(a, np.asarray(single(jnp.asarray(flipped))))
    val = float(obj(st, te, jnp.asarray(full)))
    g = _flat_lin(jax.grad(obj)(st, te, jnp.asarray(full)))
    gd = _flat_lin(jax.grad(det)(st, te, jnp.asarray(full)))
    val_old = float(obj(st, te, jnp.asarray(old)))
    g_old = _flat_lin(jax.grad(obj)(st, te, jnp.asarray(old)))
    # contrast: a genuinely x_0-conditional enumeration (eps_0 = 0, the four transition noises enumerated: also 256 patterns)
    cond = np.array(list(itertools.product((-1.0, 1.0), repeat=LK * LD))).reshape(-1, LK, LD)
    cond = np.concatenate([np.zeros((cond.shape[0], 1, LD)), cond], axis=1)
    val_cond = float(obj(st, te, jnp.asarray(cond)))
    g_cond = _flat_lin(jax.grad(obj)(st, te, jnp.asarray(cond)))
    errs = {"J_vs_analytic": abs(val - an["J"]) / abs(an["J"]), "grad_vs_analytic": float(np.abs(g - an_g).max() / np.abs(an_g).max()),
            "detached_vs_analytic": float(np.abs(gd - an_d).max() / np.abs(an_d).max()),
            "J_1024_vs_256": abs(val - val_old) / abs(val), "grad_1024_vs_256": float(np.abs(g - g_old).max() / np.abs(g).max())}
    RESULTS[f"linear_full_1024_sigma{sigma}"] = {
        "inputs": inputs, "J_analytic": an["J"], "J_1024": val, "J_256_existing": val_old, "J_conditional_on_x0_zero_NOT_full": val_cond,
        "grad_analytic": an_g.tolist(), "grad_1024": g.tolist(), "grad_256_existing": g_old.tolist(),
        "grad_conditional_on_x0_zero_NOT_full": g_cond.tolist(), "detached_analytic": an_d.tolist(), "detached_1024": gd.tolist(), "rel_errs": errs}
    print(f"\n  [5c] sigma {sigma}: {json.dumps(RESULTS[f'linear_full_1024_sigma{sigma}'])}")
    assert max(errs.values()) <= 1e-11
    assert abs(val_cond - an["J"]) > 1e-3 * abs(an["J"])                    # the check can tell conditional from full


def test_5d_k4_coefficients_from_the_implementation():
    sigma, delta = 0.05, 0.25
    s = np.array([1.0, 0.75, 0.5, 0.25])
    times, stds, weights = schedule(LK, sigma)
    formula = delta**2 / (2.0 * sigma**2 * s) * (1.0 + sigma**2 / 2.0 * (1.0 - s)) ** 2
    # the weights the rollout actually applies: per-step KL / ||dv||^2 with dv independent of the state (alpha_S = alpha_T)
    st = {"alpha": jnp.asarray(0.3), "beta": jnp.asarray([1.0, 0.0])}
    te = {"alpha": jnp.asarray(0.3), "beta": jnp.asarray([0.0, 0.0])}
    applied = np.asarray(one_path(st, te, jnp.zeros((LK + 1, LD)), linear_v, LK, sigma, "full")[0])
    k10 = kl_coefficients()["weights"]                                       # the frozen K = 10 vector, must NOT be what is used here
    RESULTS["k4_coefficients"] = {"times": times.tolist(), "stds": stds.tolist(), "schedule_weights": weights.tolist(), "applied_in_rollout": applied.tolist(),
                                  "formula": formula.tolist(), "k10_weights_first4_for_contrast": k10[:4].tolist()}
    print(f"\n  [5d] {json.dumps(RESULTS['k4_coefficients'])}")
    np.testing.assert_array_equal(times, s)
    np.testing.assert_allclose(weights, formula, rtol=1e-14)
    np.testing.assert_allclose(applied, formula, rtol=1e-14)
    np.testing.assert_allclose(weights, [12.5000, 16.6771, 25.0313, 50.0938], atol=5e-5)
    assert not np.allclose(weights, k10[:4])


def test_zz_write_results():
    RESULTS["runtime_s"] = time.perf_counter() - T0
    out = ROOT / "outputs" / "ogpo_port"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "task9_pathwise_kl_gradient.json"
    f.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\n  runtime {RESULTS['runtime_s']:.1f} s; results -> {f}")
