"""OGPO flow path log-probabilities (CPU, float64, synthetic velocity, fixed injected noise).

Executed OGPO code (ast-extracted at the pinned commit, run with real jax.numpy):
  pg_helper.py  sde_drift_correction, sample_flow_actions_sde (123-254), compute_flow_log_prob (257-414)
  ogpo.py       OGPOAgent.compute_log_probs_and_entropy (148-216), OGPOAgent.sample_actions_with_noise (2248-2333),
                run as plain functions on a minimal fake agent (config + actor returning the synthetic velocity).
Rebound names (see README caveat 1): `jax.random` (injected x0) and `distrax.Normal`
(`sample` = loc + scale * injected eps; `log_prob` = jax.scipy.stats.norm.logpdf; `entropy` = 0.5 log(2 pi e scale^2)).
Independent reference: the Gaussian path log-density written out by hand in the OpenPI convention (Task 1-2 mapping).
"""

from __future__ import annotations

import ast
import importlib.util
import math
import typing
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import jax.scipy.stats as jss
import numpy as np
import pytest
from jax.experimental import enable_x64

HERE = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


EC = _load("test_euler_convention")
ST = _load("test_sde_transition")
v_pi64 = ST.v_pi64
B, H, D = EC.B, EC.H, EC.D
A = H * D
LOG2PI = math.log(2.0 * math.pi)
RTOL = 1e-12          # float64; path log-probs are sums of ~K*A terms of O(1)


def _extract():
    pg = ast.parse(EC.PG_HELPER.read_text())
    pg_funcs = [n for n in pg.body if isinstance(n, ast.FunctionDef)
                and n.name in ("sde_drift_correction", "sample_flow_actions_sde", "compute_flow_log_prob")]
    agent_src = EC.PG_HELPER.parents[1] / "ogpo.py"
    ag = ast.parse(agent_src.read_text())
    cls = next(n for n in ag.body if isinstance(n, ast.ClassDef) and n.name == "OGPOAgent")
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ("compute_log_probs_and_entropy", "sample_actions_with_noise")]
    assert len(pg_funcs) == 3 and len(methods) == 2
    for m in methods:
        assert not m.decorator_list, f"{m.name} is decorated; plain-function extraction would be wrong"
    return pg_funcs, methods, agent_src


PG_FUNCS, METHODS, AGENT_SRC = _extract()


class Harness:
    """Executes the extracted OGPO code with injected x0 / eps and a real Gaussian log_prob."""

    def __init__(self, x0=None, eps=None):
        self.x0, self.eps = x0, list(eps or [])
        outer = self

        class Normal:
            def __init__(self, loc, scale):
                self.loc = jnp.asarray(loc)
                self.scale = jnp.broadcast_to(jnp.asarray(scale, self.loc.dtype), self.loc.shape)

            def sample(self, seed=None):
                return self.loc + self.scale * outer.eps.pop(0)

            def log_prob(self, value):
                return jss.norm.logpdf(jnp.asarray(value), self.loc, self.scale)

            def entropy(self):
                return 0.5 * jnp.log(2.0 * jnp.pi * jnp.e * self.scale**2)

        rnd = SimpleNamespace(split=lambda key: (key, key), normal=lambda key, shape: outer.x0)
        typing_names = {k: getattr(typing, k) for k in ("Tuple", "Callable", "Optional", "Dict", "Any", "List")}
        self.ns = {"jax": SimpleNamespace(random=rnd), "jnp": jnp, "distrax": SimpleNamespace(Normal=Normal), **typing_names}
        exec(compile(ast.Module(body=PG_FUNCS, type_ignores=[]), str(EC.PG_HELPER), "exec"), self.ns)  # noqa: S102
        ans = {**self.ns, "flax": SimpleNamespace(core=SimpleNamespace(FrozenDict=dict)),
               "compute_mip_log_prob": None, "sample_mip_actions_sde": None}
        exec(compile(ast.Module(body=METHODS, type_ignores=[]), str(AGENT_SRC), "exec"), ans)  # noqa: S102
        self.agent_ns = ans


class Cfg(dict):
    __getattr__ = dict.__getitem__


def make_cfg(schedule="tapered", sigma=0.05, K=10, ft=0, norm=(False, False), clip_mean=False, randn_clip=1e12,
             act_bounds=(-np.inf, np.inf)):
    return Cfg(policy_type="flow", action_dim=D, horizon_length=H, action_chunking=True, flow_steps=K,
               use_constant_noise=(schedule == "constant"), use_tapered_noise=(schedule == "tapered"),
               error_correct_sde_to_ode=True, constant_noise_std=sigma, min_noise_std=sigma,
               randn_clip_value=randn_clip, act_min=act_bounds[0], act_max=act_bounds[1],
               clip_intermediate_actions=clip_mean, denoised_clip_value=1.0, ft_flow_steps=ft,
               normalize_denoising_horizon=norm[0], normalize_act_space_dimension=norm[1])


def actor_fn(obs, actions, t, params=None, is_encoded=True):
    return (-v_pi64(actions.reshape(B, H, D), 1.0 - t.reshape(B, 1, 1))).reshape(B, A)   # v_O = -v_pi


def noise(seed, K, scale0=1.0):
    r = np.random.default_rng(20_000 + seed)
    return scale0 * jnp.asarray(r.normal(size=(B, A))), [jnp.asarray(r.normal(size=(B, A))) for _ in range(K)]


def sample(cfg, seed=0, x0=None, eps=None):
    """pg_helper.sample_flow_actions_sde -> (actions, chains, lp_sampling (unnormalised), sigmas)."""
    if x0 is None:
        x0, eps = noise(seed, cfg.flow_steps)
    h = Harness(x0, eps)
    return h.ns["sample_flow_actions_sde"](
        actor_fn, None, EC.OBS, None, cfg.flow_steps, A, use_constant_noise=cfg.use_constant_noise,
        use_tapered_noise=cfg.use_tapered_noise, error_correct_sde_to_ode=True, constant_noise_std=cfg.constant_noise_std,
        min_noise_std=cfg.min_noise_std, randn_clip_value=cfg.randn_clip_value, act_min=cfg.act_min, act_max=cfg.act_max,
        clip_intermediate=cfg.clip_intermediate_actions, clip_value=cfg.denoised_clip_value, ft_flow_steps=cfg.ft_flow_steps)


def recompute(cfg, chains):
    """pg_helper.compute_flow_log_prob -> (logprob, info)."""
    lp, _, info = Harness().ns["compute_flow_log_prob"](
        actor_fn, None, EC.OBS, jnp.asarray(chains), cfg.flow_steps, use_constant_noise=cfg.use_constant_noise,
        use_tapered_noise=cfg.use_tapered_noise, error_correct_sde_to_ode=True, constant_noise_std=cfg.constant_noise_std,
        min_noise_std=cfg.min_noise_std, normalize_horizon=cfg.normalize_denoising_horizon,
        normalize_dim=cfg.normalize_act_space_dimension, clip_intermediate=cfg.clip_intermediate_actions,
        clip_value=cfg.denoised_clip_value, ft_flow_steps=cfg.ft_flow_steps)
    return np.asarray(lp), info


class FakeAgent:
    def __init__(self, cfg):
        self.config = cfg
        self.actor_network = SimpleNamespace(params={"actor": "same"},
                                             select=lambda name: (actor_fn if "actor" in name else None))


def agent_sample(cfg, seed=0):
    """OGPOAgent.sample_actions_with_noise (target actor; agent-level normalisation, ogpo.py:2318-2331)."""
    x0, eps = noise(seed, cfg.flow_steps)
    h = Harness(x0, eps)
    return h.agent_ns["sample_actions_with_noise"](FakeAgent(cfg), EC.OBS, rng=None, is_encoded=True)


def agent_recompute(cfg, chains):
    """OGPOAgent.compute_log_probs_and_entropy with the SAME parameters (here: the same synthetic velocity)."""
    h = Harness()
    lp, _ = h.agent_ns["compute_log_probs_and_entropy"](FakeAgent(cfg), EC.OBS, jnp.asarray(chains), {"actor": "same"},
                                                        get_entropy=False, is_encoded=True)
    return np.asarray(lp)


def independent(cfg, chains):
    """Hand-written Gaussian path log-density in the OpenPI convention, using the stored chain.
    log N(x';mu,s^2 I) = -d/2 log 2pi - d log s - |x'-mu|^2/(2 s^2), d = A (isotropic per step)."""
    K, sig, Dt = cfg.flow_steps, cfg.constant_noise_std, 1.0 / cfg.flow_steps
    eff = cfg.ft_flow_steps if 0 < cfg.ft_flow_steps < K else K
    ft_start = K - eff
    c = np.asarray(chains)
    lp = np.zeros(B)
    if ft_start == 0:
        lp += -0.5 * A * LOG2PI - 0.5 * (c[:, 0] ** 2).sum(-1)                  # log N(x0; 0, I)
    for k in range(ft_start, K):
        s, last = 1.0 - k / K, k == K - 1
        x = jnp.asarray(c[:, k]).reshape(B, H, D)
        vp = v_pi64(x, s)
        if cfg.use_tapered_noise:
            mu, sd = x - Dt * (vp + 0.5 * sig**2 * (x + (1 - s) * vp)), sig * math.sqrt(s)
        elif last:                                                               # constant: deterministic last step
            continue
        else:
            mu, sd = x - Dt * (vp + 0.5 * sig**2 * (x + (1 - s) * vp) / s), sig
        mu = np.asarray(mu).reshape(B, A)
        if cfg.clip_intermediate_actions:
            mu = np.clip(mu, -cfg.denoised_clip_value, cfg.denoised_clip_value)
        lp += -0.5 * A * LOG2PI - A * math.log(sd) - ((c[:, k + 1] - mu) ** 2).sum(-1) / (2 * sd**2)
    return lp


SCHEDULES = ("tapered", "constant")


# ------------------------------------------------------------------ 1-3, 6: exact agreement (unclipped, unnormalised)
@pytest.mark.parametrize("seed", (0, 1, 2))
@pytest.mark.parametrize("K", (1, 2, 10))
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_sampling_recompute_and_independent_sum_agree(schedule, K, seed):
    with enable_x64():
        cfg = make_cfg(schedule, K=K)
        _, chains, lp_s, _ = sample(cfg, seed)
        lp_r, info = recompute(cfg, chains)
        lp_i = independent(cfg, chains)
    np.testing.assert_allclose(np.asarray(lp_s), lp_i, rtol=RTOL)        # sampling-time == independent
    np.testing.assert_allclose(lp_r, lp_i, rtol=RTOL)                    # compute_flow_log_prob == independent
    np.testing.assert_allclose(lp_r, np.asarray(lp_s), rtol=RTOL)        # recompute == sampling-time
    np.testing.assert_allclose(lp_r - np.asarray(lp_s), 0.0, atol=1e-8)  # identical params: log-ratio 0 ...
    np.testing.assert_allclose(np.exp(lp_r - np.asarray(lp_s)), 1.0, atol=1e-8)  # ... ratio 1


# ------------------------------------------------------------------ 4, 5: ft_flow_steps and the initial-noise term
@pytest.mark.parametrize("schedule", SCHEDULES)
def test_partial_chain_scores_only_last_steps_without_init(schedule):
    with enable_x64():
        full, part = make_cfg(schedule, K=10), make_cfg(schedule, K=10, ft=3)
        _, chains, lp_full_s, _ = sample(full, 0)
        _, chains_p, lp_part_s, _ = sample(part, 0)
        np.testing.assert_allclose(np.asarray(chains_p), np.asarray(chains), rtol=0, atol=0)   # same chain generated
        lp_part_r, info_p = recompute(part, chains)
        lp_part_i = independent(part, chains)
    np.testing.assert_allclose(np.asarray(lp_part_s), lp_part_i, rtol=RTOL)
    np.testing.assert_allclose(lp_part_r, lp_part_i, rtol=RTOL)
    assert np.all(np.asarray(lp_part_s) != np.asarray(lp_full_s))
    expected_steps = 3 if schedule == "tapered" else 2          # constant: the masked last step is one of the 3
    assert int(info_p["logprob_steps"]) == expected_steps


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_initial_noise_term_included_only_for_full_chain(schedule):
    with enable_x64():
        cfg = make_cfg(schedule, K=10)
        _, chains, lp_s, _ = sample(cfg, 1)
        init = -0.5 * A * LOG2PI - 0.5 * (np.asarray(chains)[:, 0] ** 2).sum(-1)
        lp_r, info = recompute(cfg, chains)
        # the full-chain value minus the init term equals the sum of transitions computed independently
        trans_only = independent(cfg, chains) - init
        cfg10 = make_cfg(schedule, K=10, ft=9)                  # ft = K-1 < K: partial -> init omitted
        lp_9, _ = recompute(cfg10, chains)
    np.testing.assert_allclose(lp_r - init, trans_only, rtol=RTOL)
    assert int(info["logprob_steps"]) == 1 + (10 if schedule == "tapered" else 9)   # init counted as one step
    assert np.all(np.abs(lp_9 - (lp_r - init)) > 0)            # partial chain also drops transition 0


# ------------------------------------------------------------------ 7: normalisation on/off (pg_helper level)
@pytest.mark.parametrize("schedule", SCHEDULES)
@pytest.mark.parametrize("norm", ((True, False), (False, True), (True, True)))
def test_normalisation_divides_by_steps_and_dims(schedule, norm):
    with enable_x64():
        raw_cfg, n_cfg = make_cfg(schedule), make_cfg(schedule, norm=norm)
        _, chains, _, _ = sample(raw_cfg, 2)
        lp_raw, info = recompute(raw_cfg, chains)
        lp_n, _ = recompute(n_cfg, chains)
    steps = int(info["logprob_steps"])
    expected = lp_raw / (steps if norm[0] else 1) / (A if norm[1] else 1)
    np.testing.assert_allclose(lp_n, expected, rtol=RTOL)


# ------------------------------------------------------------------ agent level: identical params, normalised
def test_agent_level_tapered_normalised_ratio_is_one():
    with enable_x64():
        cfg = make_cfg("tapered", norm=(True, True))
        _, chains, old_lp, _ = agent_sample(cfg)
        new_lp = agent_recompute(cfg, chains)
    np.testing.assert_allclose(new_lp - np.asarray(old_lp), 0.0, atol=1e-10)


# ================================================================== likelihood-correctness investigations
def test_three_sigma_noise_clip_is_scored_as_an_ordinary_gaussian():
    """eps forced to 5 in one coordinate: the sample is clipped to mu + 3 sigma, and both sampler and recompute score
    log N(mu + 3 sigma; mu, sigma^2) = -log sigma - log(2 pi)/2 - 4.5 there. Under the clipped process that value is an
    ATOM with probability P(eps >= 3) = 1.35e-3, not a density point: B (Gaussian approximation)."""
    with enable_x64():
        cfg = make_cfg("tapered", K=10, randn_clip=3.0)
        x0, eps = noise(3, 10)
        eps = [e.at[0, 0].set(5.0) if k == 4 else e for k, e in enumerate(eps)]
        _, chains, lp_s, sig = sample(cfg, x0=x0, eps=eps)
        lp_r, _ = recompute(cfg, chains)
        lp_i = independent(cfg, chains)
        s4 = float(np.asarray(sig)[0, 4, 0])
        step4_x, step5_x = np.asarray(chains)[0, 4], np.asarray(chains)[0, 5]
        mu4 = independent_mean("tapered", cfg, step4_x, 4)
    assert abs((step5_x[0] - mu4[0]) / s4 - 3.0) < 1e-9          # stored value sits exactly on the 3-sigma boundary
    np.testing.assert_allclose(np.asarray(lp_s), lp_r, rtol=RTOL)  # sampler and recompute agree with EACH OTHER
    np.testing.assert_allclose(lp_r, lp_i, rtol=RTOL)              # ... and with the unclipped-Gaussian formula
    log_density_at_boundary = -math.log(s4) - 0.5 * LOG2PI - 4.5
    log_true_atom_mass = math.log(0.5 * math.erfc(3.0 / math.sqrt(2.0)))
    assert abs(log_true_atom_mass - math.log(1.3499e-3)) < 1e-3
    print(f"\n  boundary element: Gaussian log-density {log_density_at_boundary:.4f} (a density, units 1/length) vs "
          f"log P(clipped atom) {log_true_atom_mass:.4f} (a probability); per-element clip rate at |eps|>3: 2.70e-3")


def independent_mean(schedule, cfg, x_flat, k):
    K, sig, Dt = cfg.flow_steps, cfg.constant_noise_std, 1.0 / cfg.flow_steps
    s = 1.0 - k / K
    x = jnp.asarray(np.broadcast_to(x_flat, (B, A))).reshape(B, H, D)
    vp = v_pi64(x, s)
    mu = x - Dt * (vp + 0.5 * sig**2 * (x + (1 - s) * vp)) if schedule == "tapered" else \
        x - Dt * (vp + 0.5 * sig**2 * (x + (1 - s) * vp) / s)
    return np.asarray(mu).reshape(B, A)[0]


def test_final_action_clip_after_sampling_makes_sampler_and_recompute_disagree():
    """Tapered: the last step is scored BEFORE the [act_min, act_max] clip (pg_helper.py:243-247) but the chain stores
    the clipped value, which compute_flow_log_prob scores. Whenever the clip binds, the two numbers differ: B."""
    with enable_x64():
        cfg = make_cfg("tapered", K=10, act_bounds=(-1.0, 1.0))
        x0, eps = noise(4, 10, scale0=3.0)
        _, chains, lp_s, _ = sample(cfg, x0=x0, eps=eps)
        lp_r, _ = recompute(cfg, chains)
        free = make_cfg("tapered", K=10)
        _, chains_free, lp_free, _ = sample(free, x0=x0, eps=list(eps))
    binds = np.abs(np.asarray(chains_free)[:, -1]) > 1.0
    assert binds.any(), "test setup must make the final clip bind"
    np.testing.assert_allclose(np.asarray(lp_s), np.asarray(lp_free), rtol=RTOL)  # sampler scored the PRE-clip value
    diff = lp_r - np.asarray(lp_s)
    assert np.all(np.abs(diff[binds.any(axis=1)]) > 1.0)                          # recompute scored the clipped value
    print(f"\n  final clip binds on {int(binds.sum())} of {binds.size} elements; recompute - sampling log-prob per row: "
          f"{np.round(diff, 2)}")


def test_intermediate_mean_clip_is_consistent_and_exact_for_the_clipped_process():
    """Mean clip is a deterministic function of x_k applied identically in sampling and recompute: both equal the
    independent Gaussian sum with clipped means. Exact density of the (modified) clipped-mean process: A."""
    with enable_x64():
        cfg = make_cfg("tapered", K=10, clip_mean=True)
        x0, eps = noise(5, 10, scale0=3.0)
        _, chains, lp_s, _ = sample(cfg, x0=x0, eps=eps)
        lp_r, _ = recompute(cfg, chains)
        lp_i = independent(cfg, chains)
        noclip = make_cfg("tapered", K=10)
        _, chains_nc, _, _ = sample(noclip, x0=x0, eps=list(eps))
    np.testing.assert_allclose(np.asarray(lp_s), lp_i, rtol=RTOL)
    np.testing.assert_allclose(lp_r, lp_i, rtol=RTOL)
    assert np.abs(np.asarray(chains) - np.asarray(chains_nc)).max() > 1e-3          # the clip changed the process


def test_constant_noise_deterministic_final_step():
    """sigma_{K-1} = 0: x_K = mu_{K-1} exactly; the sampler adds nothing for that step and the recompute masks it
    (pg_helper.py:233-234, 364-375). The value is the exact density of the stochastic PREFIX (x_0..x_{K-1}); the full
    (x_0..x_K) has no density w.r.t. Lebesgue measure in (K+1)*A dims (x_K is a deterministic function): A for the prefix."""
    with enable_x64():
        cfg = make_cfg("constant", K=10)
        _, chains, lp_s, sig = sample(cfg, 6)
        lp_r, info = recompute(cfg, chains)
        c = np.asarray(chains)
        x9 = jnp.asarray(c[:, 9]).reshape(B, H, D)
        det = np.asarray(x9 - 0.1 * v_pi64(x9, 0.1)).reshape(B, A)        # last step: no correction, no noise
    assert np.all(np.asarray(sig)[:, -1] == 0.0)
    np.testing.assert_allclose(c[:, 10], det, rtol=RTOL)
    np.testing.assert_allclose(np.asarray(lp_s), lp_r, rtol=RTOL)
    assert int(info["logprob_steps"]) == 1 + 9


@pytest.mark.parametrize("schedule", SCHEDULES)
def test_agent_level_normalisation_divisors(schedule):
    """Sampler (ogpo.py:2324-2331) divides by effective_ft + 1 = 11 for a full chain; compute_flow_log_prob divides by
    1 + contributing steps = 11 (tapered) or 10 (constant). Identical parameters -> log-ratio 0 only for tapered."""
    with enable_x64():
        cfg = make_cfg(schedule, K=10, norm=(True, True))
        _, chains, old_lp, _ = agent_sample(cfg)
        new_lp = agent_recompute(cfg, chains)
        raw = make_cfg(schedule, K=10)
        _, _, lp_raw, _ = sample(raw, 0)
    old_lp, raw_lp = np.asarray(old_lp), np.asarray(lp_raw)
    np.testing.assert_allclose(old_lp, raw_lp / 11 / A, rtol=RTOL)
    if schedule == "tapered":
        np.testing.assert_allclose(new_lp, raw_lp / 11 / A, rtol=RTOL)
        np.testing.assert_allclose(new_lp - old_lp, 0.0, atol=1e-10)
    else:
        np.testing.assert_allclose(new_lp, raw_lp / 10 / A, rtol=RTOL)
        log_ratio = new_lp - old_lp
        np.testing.assert_allclose(log_ratio, raw_lp / A * (1 / 10 - 1 / 11), rtol=RTOL)
        assert np.all(np.abs(log_ratio) > 1e-3)
        print(f"\n  constant noise, identical params: log-ratio {np.round(log_ratio, 4)}  ratio {np.round(np.exp(log_ratio), 4)}")
