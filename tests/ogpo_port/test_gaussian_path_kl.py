"""Teacher-student Gaussian path KL under the FROZEN configuration (tests/ogpo_port/README.md). CPU, float64, NumPy only,
synthetic nonlinear observation-conditioned velocity fields. No pi0.5, no optimisation.

Frozen: K = 10; OpenPI time s_k = 1 - k/K, Delta = 1/K; `ogpo_discrete` noise sigma_k = sigma*sqrt(s_k) with base sigma = 0.05
(the value used for the ledger's 10.0225 coefficient ratio and by OGPO's tapered scripts); score correction on; no clipping of
any kind; no normalisation; full chain; common N(0, I_320) start; all 320 coordinates stochastic; identical covariance.

Corrected transition mean (verified in Tasks 1-2):  mu_theta,k = x_k - Delta [ v_theta + sigma^2/2 (x_k + (1 - s_k) v_theta) ].
SCORING RULE: trajectories are sampled ONLY from the student; the teacher is evaluated at the student's states x_k^S.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

K, SIGMA = 10, 0.05
DELTA = 1.0 / K
H, D, A, OBS_DIM = 10, 32, 320, 6
N_TRAJ, BATCH = 40_000, 2_000
LOG2PI = math.log(2.0 * math.pi)
REL = 1e-12

# coordinate groups on the (10 rows x 32 dims) chunk
GROUP_EXEC = np.zeros((H, D), bool); GROUP_EXEC[0:5, 0:7] = True
GROUP_UNEXEC = np.zeros((H, D), bool); GROUP_UNEXEC[5:10, 0:7] = True
GROUP_PAD = np.zeros((H, D), bool); GROUP_PAD[:, 7:32] = True
GROUPS = {"executed": GROUP_EXEC, "unexecuted_real": GROUP_UNEXEC, "padded": GROUP_PAD}


def s_of(k):
    return 1.0 - k / K


def sigma_k(k):
    return SIGMA * math.sqrt(s_of(k))


def w_of(k):
    s = s_of(k)
    return DELTA**2 / (2.0 * SIGMA**2 * s) * (1.0 + 0.5 * SIGMA**2 * (1.0 - s)) ** 2


def make_params(seed, base=None, scale=0.1):
    r = np.random.default_rng(seed)
    p = {"W": r.normal(size=(D, D)) / math.sqrt(D), "b": r.normal(size=D), "P": r.normal(size=(OBS_DIM, D)),
         "a": r.uniform(0.5, 1.5), "c": r.uniform(-0.5, 0.5)}
    if base is None:
        return p
    return {k: base[k] + scale * (p[k] if np.ndim(p[k]) else p[k] - 1.0) for k in base}


STUDENT = make_params(1)
TEACHER = make_params(2, base=STUDENT, scale=0.1)       # different, nearby parameters


def velocity(p, x, s, o):
    """Nonlinear, observation-conditioned v(x, s, o; theta), OpenPI convention. x (N,10,32), o (N,6) -> (N,10,32)."""
    return np.tanh(x @ p["W"] + p["b"] + (o @ p["P"])[:, None, :]) * (1.0 + p["a"] * s**2) + p["c"] * np.sin(3.0 * s) * x


def corrected_mean(v, x, s):
    return x - DELTA * (v + 0.5 * SIGMA**2 * (x + (1.0 - s) * v))


class RecordingTeacher:
    """Teacher velocity that records every state it is evaluated at (to prove it never makes its own trajectory)."""

    def __init__(self, p):
        self.p, self.seen = p, []

    def __call__(self, x, s, o):
        self.seen.append(x.copy())
        return velocity(self.p, x, s, o)


def simulate(student, teacher_fn, n, seed):
    """Student-only rollouts. Returns per-trajectory arrays and per-step, per-coordinate quantities."""
    r = np.random.default_rng(seed)
    o = r.normal(size=(n, OBS_DIM))
    x = r.normal(size=(n, H, D))                                   # x_0 ~ N(0, I_320), shared start
    out = {"o": o, "x": [x], "mu_s": [], "mu_t": [], "v_s": [], "v_t": [], "eps": []}
    for k in range(K):
        s = s_of(k)
        v_s = velocity(student, x, s, o)
        v_t = teacher_fn(x, s, o)                                  # teacher at the STUDENT state x_k^S
        mu_s, mu_t = corrected_mean(v_s, x, s), corrected_mean(v_t, x, s)
        eps = r.normal(size=x.shape)
        x = mu_s + sigma_k(k) * eps                                # next state from the STUDENT transition only
        for key, val in (("mu_s", mu_s), ("mu_t", mu_t), ("v_s", v_s), ("v_t", v_t), ("eps", eps), ("x", x)):
            out[key].append(val)
    return out


# ---------------------------------------------------------------- independent reference formulas (never shared)
def kl_diag_gauss_general(mu_p, sd_p, mu_q, sd_q):
    """KL(N(mu_p, diag sd_p^2) || N(mu_q, diag sd_q^2)) per coordinate, general unequal-variance formula."""
    return np.log(sd_q / sd_p) + (sd_p**2 + (mu_p - mu_q) ** 2) / (2.0 * sd_q**2) - 0.5


def kl_shared_cov_closed_form(mu_s, mu_t, sd):
    return ((mu_s - mu_t) ** 2).sum(axis=(-2, -1)) / (2.0 * sd**2)


def gauss_logpdf_elementwise(x, mu, sd):
    return -0.5 * LOG2PI - math.log(sd) - (x - mu) ** 2 / (2.0 * sd**2)


@lru_cache(maxsize=None)
def monte_carlo():
    """N_TRAJ student trajectories in batches: per-trajectory log-ratio, analytic path KL, per-step and per-group KL."""
    lr, ana, per_step, per_group, init_s, init_t = [], [], [], {g: [] for g in GROUPS}, [], []
    for b in range(N_TRAJ // BATCH):
        sim = simulate(STUDENT, lambda x, s, o: velocity(TEACHER, x, s, o), BATCH, seed=1000 + b)
        x0 = sim["x"][0]
        ls = lt = gauss_logpdf_elementwise(x0, 0.0, 1.0).sum(axis=(1, 2))    # same N(0,I) term for both
        init_s.append(ls.copy()); init_t.append(lt.copy())
        ls, lt = ls.copy(), lt.copy()
        steps, groups = np.zeros((BATCH, K)), {g: np.zeros(BATCH) for g in GROUPS}
        for k in range(K):
            x_next, sd = sim["x"][k + 1], sigma_k(k)
            ls += gauss_logpdf_elementwise(x_next, sim["mu_s"][k], sd).sum(axis=(1, 2))
            lt += gauss_logpdf_elementwise(x_next, sim["mu_t"][k], sd).sum(axis=(1, 2))
            coord_kl = (sim["mu_s"][k] - sim["mu_t"][k]) ** 2 / (2.0 * sd**2)      # (BATCH, 10, 32)
            steps[:, k] = coord_kl.sum(axis=(1, 2))
            for g, m in GROUPS.items():
                groups[g] += coord_kl[:, m].sum(axis=1)
        lr.append(ls - lt); ana.append(steps.sum(1)); per_step.append(steps)
        for g in GROUPS:
            per_group[g].append(groups[g])
    cat = np.concatenate
    return (cat(lr), cat(ana), cat(per_step), {g: cat(v) for g, v in per_group.items()}, cat(init_s), cat(init_t))


def mean_se_ci(a):
    m, se = float(np.mean(a)), float(np.std(a, ddof=1) / math.sqrt(len(a)))
    return m, se, (m - 1.96 * se, m + 1.96 * se)


# ================================================================ tests
def test_1_direct_gaussian_kl_equals_shared_covariance_formula():
    sim = simulate(STUDENT, lambda x, s, o: velocity(TEACHER, x, s, o), 256, seed=7)
    errs = []
    for k in range(K):
        sd = sigma_k(k)
        direct = kl_diag_gauss_general(sim["mu_s"][k], np.full_like(sim["mu_s"][k], sd),
                                       sim["mu_t"][k], np.full_like(sim["mu_t"][k], sd)).sum(axis=(1, 2))
        closed = kl_shared_cov_closed_form(sim["mu_s"][k], sim["mu_t"][k], sd)
        np.testing.assert_allclose(direct, closed, rtol=1e-11, atol=1e-11)
        errs.append(np.max(np.abs(direct - closed) / np.maximum(closed, 1e-300)))
    print(f"\n  [1] max relative |direct - closed form| over all steps: {max(errs):.2e}")


def test_2_transition_mean_kl_equals_weighted_velocity_regression():
    sim = simulate(STUDENT, lambda x, s, o: velocity(TEACHER, x, s, o), 256, seed=8)
    errs = []
    for k in range(K):
        closed = kl_shared_cov_closed_form(sim["mu_s"][k], sim["mu_t"][k], sigma_k(k))
        weighted = w_of(k) * ((sim["v_s"][k] - sim["v_t"][k]) ** 2).sum(axis=(1, 2))     # sum over coordinates
        np.testing.assert_allclose(closed, weighted, rtol=REL * 100, atol=0)
        errs.append(np.max(np.abs(closed - weighted) / closed))
        # the x_k terms cancel: mu_S - mu_T = -Delta (1 + sigma^2 (1-s)/2) (v_S - v_T)
        np.testing.assert_allclose(sim["mu_s"][k] - sim["mu_t"][k],
                                   -DELTA * (1 + 0.5 * SIGMA**2 * (1 - s_of(k))) * (sim["v_s"][k] - sim["v_t"][k]),
                                   rtol=1e-12, atol=1e-15)
    print(f"  [2] max relative |closed form - w_k ||v_S - v_T||^2|: {max(errs):.2e}")


def test_3_teacher_is_evaluated_only_at_student_states():
    teacher = RecordingTeacher(TEACHER)
    sim = simulate(STUDENT, teacher, 64, seed=9)
    assert len(teacher.seen) == K
    for k in range(K):
        assert np.array_equal(teacher.seen[k], sim["x"][k])         # bitwise: x_k^S, not a teacher-propagated state
    # negative control: a teacher rollout of its own (WRONG for reverse KL) visits different states
    r = np.random.default_rng(9)
    o = r.normal(size=(64, OBS_DIM)); xt = r.normal(size=(64, H, D))
    for k in range(K):
        xt = corrected_mean(velocity(TEACHER, xt, s_of(k), o), xt, s_of(k)) + sigma_k(k) * r.normal(size=xt.shape)
    assert np.abs(xt - sim["x"][K]).max() > 1e-3


def test_4_monte_carlo_log_ratio_agrees_with_analytic_path_kl():
    lr, ana, _, _, _, _ = monte_carlo()
    m_lr, se_lr, ci_lr = mean_se_ci(lr)
    m_an, se_an, ci_an = mean_se_ci(ana)
    m_d, se_d, ci_d = mean_se_ci(lr - ana)                     # paired difference, mean 0 in expectation
    print(f"\n  [4] N = {N_TRAJ} student trajectories ({N_TRAJ // BATCH} batches of {BATCH})")
    print(f"      MC  log p_S - log p_T : mean {m_lr:.4f}  SE {se_lr:.4f}  95% CI [{ci_lr[0]:.4f}, {ci_lr[1]:.4f}]")
    print(f"      analytic sum_k KL_k   : mean {m_an:.4f}  SE {se_an:.4f}  95% CI [{ci_an[0]:.4f}, {ci_an[1]:.4f}]")
    print(f"      paired difference      : mean {m_d:.4f}  SE {se_d:.4f}  95% CI [{ci_d[0]:.4f}, {ci_d[1]:.4f}]  "
          f"z = {m_d / se_d:.2f}")
    assert abs(m_d) < 4.0 * se_d                               # consistent with zero (fixed seeds: reproducible)
    assert ci_lr[0] <= m_an <= ci_lr[1]                        # analytic mean inside the MC estimator's 95% CI
    assert se_an < se_lr                                       # Rao-Blackwellised estimator has lower variance


def test_5_identical_parameters_give_zero_kl():
    sim = simulate(STUDENT, lambda x, s, o: velocity(STUDENT, x, s, o), 512, seed=10)
    total = np.zeros(512); lr = np.zeros(512)
    for k in range(K):
        sd = sigma_k(k)
        total += kl_shared_cov_closed_form(sim["mu_s"][k], sim["mu_t"][k], sd)
        lr += (gauss_logpdf_elementwise(sim["x"][k + 1], sim["mu_s"][k], sd)
               - gauss_logpdf_elementwise(sim["x"][k + 1], sim["mu_t"][k], sd)).sum(axis=(1, 2))
    assert np.all(total == 0.0) and np.all(lr == 0.0)


def test_6_shared_initial_noise_log_probability_cancels():
    lr, _, _, _, init_s, init_t = monte_carlo()
    assert np.all(init_s == init_t) and np.all(np.abs(init_s) > 100)   # nonzero, identical for both models
    # recompute the log-ratio WITHOUT the init term on a fresh small set: identical to the version with it
    sim = simulate(STUDENT, lambda x, s, o: velocity(TEACHER, x, s, o), 128, seed=11)
    with_init = gauss_logpdf_elementwise(sim["x"][0], 0.0, 1.0).sum(axis=(1, 2))
    a = with_init.copy(); b = with_init.copy(); a0 = np.zeros(128); b0 = np.zeros(128)
    for k in range(K):
        ps = gauss_logpdf_elementwise(sim["x"][k + 1], sim["mu_s"][k], sigma_k(k)).sum(axis=(1, 2))
        pt = gauss_logpdf_elementwise(sim["x"][k + 1], sim["mu_t"][k], sigma_k(k)).sum(axis=(1, 2))
        a += ps; b += pt; a0 += ps; b0 += pt
    np.testing.assert_allclose(a - b, a0 - b0, rtol=1e-12, atol=1e-9)


def test_7_group_contributions_sum_to_full_kl():
    masks = np.stack(list(GROUPS.values()))
    assert [int(m.sum()) for m in masks] == [35, 35, 250] and np.all(masks.sum(0) == 1)   # a partition of 320
    _, ana, _, groups, _, _ = monte_carlo()
    np.testing.assert_allclose(sum(groups.values()), ana, rtol=1e-12)
    tot = float(np.mean(ana))
    print("\n  [7] group contributions (mean over trajectories; additive parts of the full KL, NOT marginal KLs):")
    for g, v in groups.items():
        m, se, ci = mean_se_ci(v)
        print(f"      {g:16s} {m:9.4f}  SE {se:.4f}  ({100 * m / tot:5.1f}% of full KL)")
    print(f"      {'sum':16s} {sum(float(np.mean(v)) for v in groups.values()):9.4f}   full KL {tot:.4f}")


def test_8_coefficient_ratio_and_measured_per_step_kl():
    ratio = w_of(K - 1) / w_of(0)
    assert abs(ratio - 10.0225) < 5e-5, ratio
    np.testing.assert_allclose(ratio, 10 * (1 + 0.05**2 * 0.9 / 2) ** 2, rtol=1e-12)
    _, _, per_step, _, _, _ = monte_carlo()
    m = per_step.mean(0)
    se = per_step.std(0, ddof=1) / math.sqrt(per_step.shape[0])
    print("\n  [8]  k   s_k   sigma_k    w_k (theory)   measured E[KL_k]  (SE)       E||v_S-v_T||^2 = KL_k / w_k")
    for k in range(K):
        print(f"      {k:2d}  {s_of(k):4.1f}  {sigma_k(k):.5f}  {w_of(k):12.6f}   {m[k]:12.4f}  ({se[k]:.4f})   {m[k] / w_of(k):10.4f}")
    print(f"      coefficient ratio w_9/w_0 = {ratio:.4f};  measured KL ratio E[KL_9]/E[KL_0] = {m[-1] / m[0]:.4f}")
