"""OGPO vs OpenPI flow conventions: Euler integrator and tapered score correction (CPU, synthetic velocity, no pi0.5).

What is compared, all on identical initial states:
  * OGPO   : the ACTUAL `sample_flow_actions_ode`, `flow_score_from_velocity`, `sde_drift_correction`, extracted with `ast`
             from the OGPO clone's ogpo/agents/modules/pg_helper.py (importing that module needs distrax, which our env
             does not have; these three functions use only jax/jnp).
  * OpenPI : (a) our ACTUAL `Flow.sample` from scripts/inspect_pi05.py with its velocity replaced by a synthetic one;
             (b) the cond/step logic of third_party/openpi/src/openpi/models/pi0.py:228,271,276,278 transcribed into a
             real `jax.lax.while_loop` (the model itself cannot run without pi0.5 weights).
Convention claim under test: s = 1 - t and v_O(x, t) = -v_pi(x, 1 - t), hence x + D*v_O = x - D*v_pi.

Run (policy env, CPU):  scripts/py-policy -m pytest tests/ogpo_port -v -p no:cacheprovider   [OGPO_SRC=<clone>]
"""

from __future__ import annotations

import ast
import importlib.util
import os
import subprocess
import typing
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
_CANDIDATES = [os.environ.get("OGPO_SRC"), str(Path.home() / "IRL-project-refs/OGPO_public"),
               "/mnt/data/users/bbouri/IRL-project-refs/OGPO_public"]
OGPO_SRC = next((Path(c) for c in _CANDIDATES if c and (Path(c) / "ogpo").is_dir()), Path("/nonexistent"))
PG_HELPER = OGPO_SRC / "ogpo/agents/modules/pg_helper.py"
OGPO_FUNCS = ("flow_score_from_velocity", "sde_drift_correction", "sample_flow_actions_ode")
KS = (1, 2, 10)
SEEDS = (0, 1, 2, 3, 4)
B, H, D = 4, 10, 32  # pi0.5 chunk shape (horizon 10 x 32 padded dims); OGPO sees it flattened to (B, H*D)
TOL = dict(rtol=1e-6, atol=1e-6)  # float32 (Flow.sample casts to float32): eps 1.2e-7, 1-ulp time differences


def _load_ogpo():
    if not PG_HELPER.is_file():
        pytest.skip(f"OGPO clone not found (set OGPO_SRC); tried {_CANDIDATES}", allow_module_level=True)
    tree = ast.parse(PG_HELPER.read_text())
    funcs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in OGPO_FUNCS]
    assert sorted(f.name for f in funcs) == sorted(OGPO_FUNCS), "OGPO functions not found by name"
    ns = {"jax": jax, "jnp": jnp, **{k: getattr(typing, k) for k in ("Tuple", "Callable", "Optional", "Dict", "Any")}}
    exec(compile(ast.Module(body=funcs, type_ignores=[]), str(PG_HELPER), "exec"), ns)  # noqa: S102 - test-only, read-only source
    commit = subprocess.run(["git", "-C", str(OGPO_SRC), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    return ns, commit


def _load_flow_cls():
    spec = importlib.util.spec_from_file_location("inspect_pi05", ROOT / "scripts/inspect_pi05.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Flow


OGPO, OGPO_COMMIT = _load_ogpo()
Flow = _load_flow_cls()

# Synthetic, non-constant, nonlinear velocity in the OpenPI convention (s = 1 is noise): depends on x, s and obs.
_rng = np.random.default_rng(1234)
W = jnp.asarray(_rng.normal(size=(D, D)) / np.sqrt(D), jnp.float32)
BIAS = jnp.asarray(_rng.normal(size=(D,)), jnp.float32)
OBS = jnp.asarray(_rng.normal(size=(B, 6)), jnp.float32)
P = jnp.asarray(_rng.normal(size=(6, D)), jnp.float32)


def v_pi(x, s):
    """x: (B, H, D); s: scalar or (B, 1, 1). Returns (B, H, D)."""
    s = jnp.asarray(s, jnp.float32)
    return jnp.tanh(x @ W + BIAS) * (1.0 + s**2) + jnp.sin(3.0 * s) * x + 0.3 * (OBS @ P)[:, None, :] * (1.0 - s)


def x0(seed):
    return jnp.asarray(np.random.default_rng(seed).normal(size=(B, H, D)), jnp.float32)


def run_ogpo(seed, K):
    """OGPO's own ODE sampler, clipping disabled; records (x_k, t_k, v_k) at every actor call."""
    rec = []

    def actor_fn(obs, actions, t, is_encoded=False):
        x = actions.reshape(B, H, D)
        v = -v_pi(x, 1.0 - t.reshape(B, 1, 1))  # v_O(x, t) = -v_pi(x, 1 - t)
        rec.append((np.asarray(x), np.asarray(t), np.asarray(v)))
        return v.reshape(B, H * D)

    out = OGPO["sample_flow_actions_ode"](actor_fn, OBS, None, K, H * D, act_min=-jnp.inf, act_max=jnp.inf,
                                          clip_intermediate=False, noises=x0(seed).reshape(B, H * D))
    return np.asarray(out).reshape(B, H, D), rec


def run_flow_sample(seed, K):
    """Our actual Flow.sample (scripts/inspect_pi05.py:205-216) with the synthetic velocity."""
    f = object.__new__(Flow)
    f.jnp = jnp
    f.velocity = lambda x, t, rec=None: v_pi(x, t)
    xs, vs, ts = f.sample(x0(seed), num_steps=K)
    return [np.asarray(x) for x in xs], [np.asarray(v) for v in vs], [float(t) for t in ts]


def run_pi0_while_loop(seed, K, velocity=v_pi):
    """pi0.py:228 dt = -1/num_steps; :271 return x_t + dt * v_t, time + dt; :276 time >= -dt/2; :278 while_loop."""
    dt = -1.0 / K

    def step(carry):
        x_t, time, n = carry
        return x_t + dt * velocity(x_t, time), time + dt, n + 1

    def cond(carry):
        return carry[1] >= -dt / 2

    x, _, n = jax.lax.while_loop(cond, step, (x0(seed), jnp.float32(1.0), jnp.int32(0)))
    return np.asarray(x), int(n)


@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("seed", SEEDS)
def test_time_grids_correspond(seed, K):
    _, rec = run_ogpo(seed, K)
    _, _, ts = run_flow_sample(seed, K)
    t_ogpo = np.array([r[1][0, 0] for r in rec])
    assert len(t_ogpo) == len(ts) == K
    np.testing.assert_allclose(t_ogpo, np.arange(K) / K, **TOL)            # OGPO t_k = k/K
    np.testing.assert_allclose(ts, 1.0 - np.arange(K) / K, **TOL)          # OpenPI s_k = 1 - k/K
    np.testing.assert_allclose(np.array(ts), 1.0 - t_ogpo, **TOL)          # s = 1 - t
    assert run_pi0_while_loop(seed, K)[1] == K                              # pi0.py's while_loop takes exactly K steps


@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("seed", SEEDS)
def test_velocity_sign_and_each_euler_step(seed, K):
    out_o, rec = run_ogpo(seed, K)
    xs, vs, ts = run_flow_sample(seed, K)
    for k in range(K):
        x_o, t_o, v_o = rec[k]
        np.testing.assert_allclose(x_o, xs[k], **TOL)                                   # same state entering step k
        np.testing.assert_allclose(v_o, -vs[k], **TOL)                                  # v_O = -v_pi
        x_next_o = rec[k + 1][0] if k + 1 < K else out_o
        np.testing.assert_allclose(x_next_o, x_o + (1.0 / K) * v_o, **TOL)             # OGPO step x + D v_O
        np.testing.assert_allclose(xs[k + 1], xs[k] + (-1.0 / K) * vs[k], **TOL)       # OpenPI step x + dt v_pi
        np.testing.assert_allclose(x_next_o, xs[k + 1], **TOL)                          # the two steps agree


@pytest.mark.parametrize("K", KS)
@pytest.mark.parametrize("seed", SEEDS)
def test_complete_trajectories_agree(seed, K):
    out_o, _ = run_ogpo(seed, K)
    xs, _, _ = run_flow_sample(seed, K)
    x_while, _ = run_pi0_while_loop(seed, K)
    np.testing.assert_allclose(out_o, xs[-1], **TOL)
    np.testing.assert_allclose(out_o, x_while, **TOL)


@pytest.mark.parametrize("K", KS)
def test_negative_control_double_negation_fails(K):
    """Negative dt AND negated velocity: x + (-D)(-v_pi) = x + D v_pi integrates the flow BACKWARDS (toward more noise)."""
    out_o, _ = run_ogpo(0, K)
    x_wrong, _ = run_pi0_while_loop(0, K, velocity=lambda x, s: -v_pi(x, s))
    with pytest.raises(AssertionError):
        np.testing.assert_allclose(out_o, x_wrong, **TOL)
    assert np.abs(out_o - x_wrong).max() > 1e-2


def test_ogpo_default_clipping_is_a_difference_not_part_of_the_integrator():
    """OGPO's ODE sampler clips to [-1, 1] by default (pg_helper.py:54-55, 74-77); OpenPI's sampler never clips."""
    def actor_fn(obs, actions, t, is_encoded=False):
        return -v_pi(actions.reshape(B, H, D), 1.0 - t.reshape(B, 1, 1)).reshape(B, H * D)

    noise = (3.0 * x0(0)).reshape(B, H * D)
    clipped = np.asarray(OGPO["sample_flow_actions_ode"](actor_fn, OBS, None, 10, H * D, noises=noise))
    assert np.abs(clipped).max() <= 1.0
    unclipped = np.asarray(OGPO["sample_flow_actions_ode"](actor_fn, OBS, None, 10, H * D, act_min=-jnp.inf,
                                                            act_max=jnp.inf, clip_intermediate=False, noises=noise))
    assert np.abs(unclipped).max() > 1.0


@pytest.mark.parametrize("seed", SEEDS)
def test_score_and_tapered_corrected_mean_in_openpi_convention(seed):
    """flow_score_from_velocity(v_O, x, t) == -(x + (1-s) v_pi)/s and the OGPO tapered mean ==
    x - D [v_pi + sigma^2/2 (x + (1-s) v_pi)]; also the constant-noise branch for completeness. float64."""
    from jax.experimental import enable_x64

    r = np.random.default_rng(100 + seed)
    with enable_x64():
        for K in KS:
            Dt = 1.0 / K
            for k in range(K):                       # t_k = k/K < 1, so 1 - t never reaches 0
                t = k / K
                s = 1.0 - t
                sigma = float(r.uniform(0.005, 0.2))
                x = jnp.asarray(r.normal(size=(B, H * D)))
                vp = jnp.asarray(r.normal(size=(B, H * D)))
                vo = -vp
                tt = jnp.full((B, 1), t)
                score = np.asarray(OGPO["flow_score_from_velocity"](vo, x, tt))
                np.testing.assert_allclose(score, -(x + (1 - s) * vp) / s, rtol=1e-12, atol=1e-12)
                c_tap = np.asarray(OGPO["sde_drift_correction"](vo, x, tt, sigma, True))
                np.testing.assert_allclose(c_tap, 0.5 * sigma**2 * s * score, rtol=1e-12, atol=1e-12)  # (sigma_t^2/2)*score
                mu_ogpo = np.asarray(x + Dt * (vo + c_tap))
                mu_pi = np.asarray(x - Dt * (vp + 0.5 * sigma**2 * (x + (1 - s) * vp)))
                np.testing.assert_allclose(mu_ogpo, mu_pi, rtol=1e-12, atol=1e-12)
                c_const = np.asarray(OGPO["sde_drift_correction"](vo, x, tt, sigma, False))
                mu_pi_const = np.asarray(x - Dt * (vp + 0.5 * sigma**2 * (x + (1 - s) * vp) / s))
                np.testing.assert_allclose(np.asarray(x + Dt * (vo + c_const)), mu_pi_const, rtol=1e-12, atol=1e-12)


def test_record_sources():
    """Pins what was tested (printed with -s)."""
    print(f"\nOGPO commit {OGPO_COMMIT}; functions {OGPO_FUNCS} from {PG_HELPER}")
    assert OGPO_COMMIT.startswith("0b3be41")
