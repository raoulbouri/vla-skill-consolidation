"""OGPO vs OpenPI stochastic flow transitions (tapered noise + score correction), CPU, float64, synthetic velocity.

OGPO side: the ACTUAL `sample_flow_actions_sde` + `sde_drift_correction` (ast-extracted from the OGPO clone's
pg_helper.py, same loader as test_euler_convention.py). Its random draws are made injectable by binding two names in
its namespace: `jax.random` (split passes keys through, `normal` returns the injected x0) and `distrax.Normal`
(`sample` returns loc + scale * eps_k from the injected noise list and records (loc, scale) per step; `log_prob`
returns zeros and is NOT tested: log-probabilities are out of scope here). All arithmetic (drift, correction, mean
clip, sigma schedule, noise clip, final clip) is OGPO's own code with real jax.numpy.
Assumption (not executed here, distrax is not installed): distrax.Normal.sample(seed) = loc + scale * N(0, 1) draw.

OpenPI side: reference written in the OpenPI convention (s = 1 noise), with the synthetic velocity v_pi from
test_euler_convention.py:  mu_pi = x - D [v_pi + sigma^2/2 (x + (1 - s) v_pi)],  x' = mu_pi + sigma_k eps.
Two NAMED noise variants:  OGPO_DISCRETE  sigma_k = sigma * sqrt(s_k)          (what OGPO's code does, pg_helper.py:214)
                           EULER_MARUYAMA sigma_k = sigma * sqrt(s_k * Delta)  (EM discretisation of the cited SDE)
"""

from __future__ import annotations

import ast
import importlib.util
import typing
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import enable_x64

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("euler_conv", HERE / "test_euler_convention.py")
EC = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(EC)          # reuses its OGPO clone discovery, synthetic v_pi, shapes and Flow loader
B, H, D, A = EC.B, EC.H, EC.D, EC.H * EC.D
KS = (1, 2, 10)
SEEDS = (0, 1, 2)
SIGMAS = (0.05, 0.2)
TOL = dict(rtol=1e-12, atol=1e-12)   # float64 throughout; identical operation order up to commutativity
OGPO_DISCRETE, EULER_MARUYAMA = "ogpo_discrete", "euler_maruyama"


def _load_sde():
    tree = ast.parse(EC.PG_HELPER.read_text())
    names = ("sde_drift_correction", "sample_flow_actions_sde")
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert sorted(f.name for f in funcs) == sorted(names)
    return funcs


SDE_FUNCS = _load_sde()


class Injected:
    """Binds OGPO's `jax.random` and `distrax.Normal` to injected arrays; records each Normal's (loc, scale)."""

    def __init__(self, x0, eps):
        self.x0, self.eps, self.calls = x0, list(eps), []
        self.jax = SimpleNamespace(random=SimpleNamespace(split=lambda key: (key, key),
                                                          normal=lambda key, shape: self._x0(shape)))
        outer = self

        class Normal:
            def __init__(self, loc, scale):
                self.loc, self.scale = jnp.asarray(loc), jnp.broadcast_to(jnp.asarray(scale), jnp.shape(loc))
                outer.calls.append(self)

            def sample(self, seed=None):
                e = outer.eps.pop(0)
                return self.loc + self.scale * e

            def log_prob(self, value):  # not under test
                return jnp.zeros_like(jnp.asarray(value))

        self.distrax = SimpleNamespace(Normal=Normal)

    def _x0(self, shape):
        assert tuple(shape) == tuple(self.x0.shape)
        return self.x0

    def sampler(self):
        ns = {"jax": self.jax, "jnp": jnp, "distrax": self.distrax,
              **{k: getattr(typing, k) for k in ("Tuple", "Callable", "Optional", "Dict", "Any")}}
        exec(compile(ast.Module(body=SDE_FUNCS, type_ignores=[]), str(EC.PG_HELPER), "exec"), ns)  # noqa: S102
        return ns["sample_flow_actions_sde"]

    def step_dists(self):
        return self.calls[1:]  # calls[0] is the N(0, I) used only for the initial log-prob


def noise(seed, K):
    r = np.random.default_rng(10_000 + seed)
    x0 = jnp.asarray(r.normal(size=(B, A)))
    eps = [jnp.asarray(r.normal(size=(B, A))) for _ in range(K)]
    return x0, eps


def v_pi64(x, s):
    """The SAME synthetic v_pi as test_euler_convention.py, evaluated in float64 (its weights are float32 values)."""
    s = jnp.asarray(s, jnp.float64)
    W, b, obs, P = (jnp.asarray(a, jnp.float64) for a in (EC.W, EC.BIAS, EC.OBS, EC.P))
    return jnp.tanh(x @ W + b) * (1.0 + s**2) + jnp.sin(3.0 * s) * x + 0.3 * (obs @ P)[:, None, :] * (1.0 - s)


def run_ogpo(seed, K, sigma, clip_intermediate=False, clip_value=1.0, randn_clip=1e12, x0=None, eps=None):
    """OGPO's own sampler, tapered noise + score correction. Clipping disabled unless asked; final action clip
    bounds are +-inf (environment action clipping is not part of the latent transition)."""
    if x0 is None:
        x0, eps = noise(seed, K)
    inj = Injected(x0, eps)

    def actor_fn(obs, actions, t, is_encoded=False):
        return (-v_pi64(actions.reshape(B, H, D), 1.0 - t.reshape(B, 1, 1))).reshape(B, A)   # v_O = -v_pi

    out, chains, _, sigmas = inj.sampler()(
        actor_fn, None, EC.OBS, jnp.zeros(2, jnp.uint32), K, A,
        use_constant_noise=False, use_tapered_noise=True, error_correct_sde_to_ode=True, constant_noise_std=sigma,
        randn_clip_value=randn_clip, act_min=-jnp.inf, act_max=jnp.inf,
        clip_intermediate=clip_intermediate, clip_value=clip_value, ft_flow_steps=0)
    means = [np.asarray(d.loc) for d in inj.step_dists()]
    stds = [np.asarray(d.scale) for d in inj.step_dists()]
    return np.asarray(chains), means, stds, np.asarray(sigmas)


def run_openpi(seed, K, sigma, variant):
    """Reference in the OpenPI convention; returns chain, means and stds per step."""
    x0, eps = noise(seed, K)
    Dt = 1.0 / K
    x = x0.reshape(B, H, D)
    chain, means, stds = [np.asarray(x0)], [], []
    for k in range(K):
        s = 1.0 - k / K
        vp = v_pi64(x, s)
        c_pi = 0.5 * sigma**2 * (x + (1.0 - s) * vp)
        mu = x - Dt * (vp + c_pi)
        sd = sigma * np.sqrt(s) if variant == OGPO_DISCRETE else sigma * np.sqrt(s * Dt)
        x = mu + sd * eps[k].reshape(B, H, D)
        means.append(np.asarray(mu).reshape(B, A))
        stds.append(np.full((B, A), sd))
        chain.append(np.asarray(x).reshape(B, A))
    return np.stack(chain, axis=1), means, stds


GRID = [(seed, K, sig) for seed in SEEDS for K in KS for sig in SIGMAS]


@pytest.mark.parametrize("seed,K,sigma", GRID)
def test_transition_means_agree(seed, K, sigma):
    with enable_x64():
        _, m_o, _, _ = run_ogpo(seed, K, sigma)
        _, m_p, _ = run_openpi(seed, K, sigma, OGPO_DISCRETE)
    assert len(m_o) == len(m_p) == K
    for k in range(K):
        np.testing.assert_allclose(m_o[k], m_p[k], **TOL)


@pytest.mark.parametrize("seed,K,sigma", GRID)
def test_per_step_stds_agree(seed, K, sigma):
    with enable_x64():
        _, _, s_o, sig_ret = run_ogpo(seed, K, sigma)
        _, _, s_p = run_openpi(seed, K, sigma, OGPO_DISCRETE)
    for k in range(K):
        np.testing.assert_allclose(s_o[k], s_p[k], **TOL)                     # scale passed to the Gaussian
        np.testing.assert_allclose(sig_ret[:, k], s_p[k], **TOL)               # `sigmas` returned by OGPO
        np.testing.assert_allclose(s_o[k], sigma * np.sqrt(1.0 - k / K), **TOL)  # sigma * sqrt(1 - t_k) = sigma * sqrt(s_k)


@pytest.mark.parametrize("seed,K,sigma", GRID)
def test_sampled_next_states_agree(seed, K, sigma):
    with enable_x64():
        c_o, m_o, s_o, _ = run_ogpo(seed, K, sigma)
        c_p, _, _ = run_openpi(seed, K, sigma, OGPO_DISCRETE)
        _, eps = noise(seed, K)
    for k in range(K):
        np.testing.assert_allclose(c_o[:, k + 1], m_o[k] + s_o[k] * np.asarray(eps[k]), **TOL)  # x' = mu + sigma_k eps
        np.testing.assert_allclose(c_o[:, k + 1], c_p[:, k + 1], **TOL)


@pytest.mark.parametrize("seed,K,sigma", GRID)
def test_complete_stochastic_trajectories_agree(seed, K, sigma):
    with enable_x64():
        c_o, _, _, _ = run_ogpo(seed, K, sigma)
        c_p, _, _ = run_openpi(seed, K, sigma, OGPO_DISCRETE)
    assert c_o.shape == c_p.shape == (B, K + 1, A)
    np.testing.assert_allclose(c_o, c_p, **TOL)


@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("seed", SEEDS)
def test_sigma_zero_reduces_to_the_verified_ode(seed, K):
    """sigma = 0: correction vanishes, sigma_k = 0, so every SDE mean/state equals the Task-1 Euler step x - D v_pi."""
    with enable_x64():
        c_o, m_o, s_o, _ = run_ogpo(seed, K, 0.0)
        x0, _ = noise(seed, K)
        x = x0.reshape(B, H, D)
        for k in range(K):
            x_next = x - (1.0 / K) * v_pi64(x, 1.0 - k / K)
            np.testing.assert_allclose(m_o[k], np.asarray(x_next).reshape(B, A), **TOL)
            assert np.all(s_o[k] == 0.0)
            np.testing.assert_allclose(c_o[:, k + 1], np.asarray(x_next).reshape(B, A), **TOL)
            x = x_next
    # and in float32 the same chain matches the Task-1 ODE run of our actual Flow.sample (same x0)
    f = object.__new__(EC.Flow)
    f.jnp, f.velocity = jnp, (lambda xx, t, rec=None: EC.v_pi(xx, t))
    xs, _, _ = f.sample(jnp.asarray(np.asarray(x0).reshape(B, H, D), jnp.float32), num_steps=K)
    np.testing.assert_allclose(c_o[:, -1].reshape(B, H, D), np.asarray(xs[-1]), rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("K", KS)
def test_final_tapered_step_is_stochastic(K):
    sigma = 0.05
    with enable_x64():
        c_o, m_o, s_o, _ = run_ogpo(0, K, sigma)
    s_last = 1.0 / K                                            # last grid point: t = (K-1)/K, s = 1/K
    np.testing.assert_allclose(s_o[-1], sigma * np.sqrt(s_last), **TOL)
    assert np.all(s_o[-1] > 0)
    assert np.abs(c_o[:, -1] - m_o[-1]).max() > 0              # x_K differs from its mean: noise was injected


def test_two_noise_conventions_are_different_processes(capsys):
    """OGPO_DISCRETE sigma*sqrt(s) vs EULER_MARUYAMA sigma*sqrt(s*Delta), K = 10, sigma = 0.05 (same eps)."""
    K, sigma = 10, 0.05
    with enable_x64():
        _, _, s_ogpo = run_openpi(0, K, sigma, OGPO_DISCRETE)
        _, _, s_em = run_openpi(0, K, sigma, EULER_MARUYAMA)
        c_ogpo, _, _ = run_openpi(0, K, sigma, OGPO_DISCRETE)
        c_em, _, _ = run_openpi(0, K, sigma, EULER_MARUYAMA)
        c_real, _, s_real, _ = run_ogpo(0, K, sigma)
    rows = []
    for k in range(K):
        a, b = s_ogpo[k][0, 0], s_em[k][0, 0]
        np.testing.assert_allclose(b, a * np.sqrt(1.0 / K), **TOL)           # EM = OGPO * sqrt(Delta)
        np.testing.assert_allclose(s_real[k][0, 0], a, **TOL)                # OGPO's code is the OGPO_DISCRETE variant
        rows.append((k, 1.0 - k / K, a, b, a / b))
    np.testing.assert_allclose(c_real, c_ogpo, **TOL)
    with pytest.raises(AssertionError):                                      # not the same discrete process
        np.testing.assert_allclose(c_real, c_em, **TOL)
    with capsys.disabled():
        print("\n   k   s_k   sigma_OGPO=sigma*sqrt(s)  sigma_EM=sigma*sqrt(s*D)  ratio")
        for k, s, a, b, r in rows:
            print(f"  {k:2d}  {s:4.1f}   {a:.6f}                {b:.6f}                {r:.4f}")
        print(f"   final-state max |OGPO_DISCRETE - EULER_MARUYAMA| (same eps): {np.abs(c_ogpo[:, -1] - c_em[:, -1]).max():.4e}")


def test_mean_clipping_and_noise_clipping_are_distinct_operations():
    """OGPO clips the MEAN to [-clip_value, clip_value] (pg_helper.py:210-211) BEFORE sampling, and the SAMPLE to
    mean +- randn_clip_value * sigma (239-241) AFTER sampling. Final action clip (246-247) is excluded (bounds +-inf)."""
    K, sigma = 1, 0.05
    with enable_x64():
        x0 = jnp.full((B, A), 5.0)                    # mean after one step is far outside [-1, 1]
        small = [jnp.full((B, A), 0.5)]               # |eps| < 3: noise clip inactive
        _, m_clip, s, _ = run_ogpo(0, K, sigma, clip_intermediate=True, randn_clip=3.0, x0=x0, eps=small)
        _, m_free, _, _ = run_ogpo(0, K, sigma, clip_intermediate=False, randn_clip=3.0, x0=x0, eps=small)
        assert np.abs(m_free[0]).max() > 1.0 and np.abs(m_clip[0]).max() <= 1.0     # only the mean clip acted
        x0b = jnp.asarray(noise(0, 1)[0])
        big = [jnp.full((B, A), 5.0)]                 # eps = 5 > 3: noise clip binds, mean untouched
        c_nc, m_nc, s_nc, _ = run_ogpo(0, K, sigma, clip_intermediate=False, randn_clip=3.0, x0=x0b, eps=big)
        c_no, m_no, _, _ = run_ogpo(0, K, sigma, clip_intermediate=False, randn_clip=1e12, x0=x0b, eps=big)
    np.testing.assert_allclose(m_nc[0], m_no[0], **TOL)                                       # same mean
    np.testing.assert_allclose(c_nc[:, 1], m_nc[0] + 3.0 * s_nc[0], **TOL)                     # sample clipped to mu + 3 sigma
    np.testing.assert_allclose(c_no[:, 1], m_no[0] + 5.0 * s_nc[0], **TOL)                     # unclipped sample at mu + 5 sigma
