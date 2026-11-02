"""Exact, unnormalised log-probability of a complete chain of the frozen Task 6 `ogpo_discrete` sampler (real pi0.5).

For X = (x_0, ..., x_K) with K = 10, s_k = 1 - k/K, Delta = 1/K, base sigma = 0.05, sigma_k = sigma * sqrt(s_k):

    log p_theta(X | o) = log N(x_0; 0, I_320) + sum_{k=0}^{K-1} log N(x_{k+1}; mu_theta(x_k, s_k, o), sigma_k^2 I_320)

    v_k  = v_pi(x_k, s_k, o)                          (pi05_velocity.velocity: the verified adapter; bf16 model, fp32 value)
    c_k  = sigma^2 / 2 * [x_k + (1 - s_k) v_k]
    mu_k = x_k - Delta * (v_k + c_k)                  (exactly the Task 6 sampler equations, pi05_sde.py)
    log N(y; mu, sd^2 I) = -1/2 * sum_{d=1}^{320} [ (y_d - mu_d)^2 / sd^2 + log(2 pi sd^2) ]

All 320 coordinates, the initial density and all K transitions are included; no clipping, truncation or normalisation.
The stored chain is treated as FIXED DATA (stop_gradient): gradients flow only through the recomputed means mu_theta, and,
with the gradient-stopped prefix cache, only into the action expert and action heads (never the frozen VLM).
Means are RECOMPUTED from the stored states (never taken from the sampler). Use the same batch shape, weights, observation,
prefix cache and execution mode as when sampling (bf16 velocities depend on the GEMM shapes: Task 5 caveat).
Precision: fp32 latent states, means, residuals and per-transition sums (reductions over 320 coordinates in fp32).
"""

from __future__ import annotations

import math
from typing import Any

from clvla.ogpo import pi05_sde as sde
from clvla.ogpo import pi05_velocity as pv

LOG_2PI = math.log(2.0 * math.pi)


def path_log_prob(model: Any, cache: "pv.PrefixCache", chain: Any, cfg: "sde.SDEConfig" = sde.SDEConfig()) -> dict:
    """chain: (B, K+1, 10, 32). Returns fp32 arrays:
    total (B,), initial (B,), transitions (B, K), means (B, K, 10, 32) [recomputed], stds (K,)."""
    import jax
    import jax.numpy as jnp

    if cfg.noise_mode != sde.NOISE_MODE or not cfg.score_correction:
        raise ValueError("path_log_prob implements exactly the frozen ogpo_discrete sampler with score correction")
    if cfg.sigma <= 0:
        raise ValueError("sigma must be > 0: with sigma = 0 the transitions are Dirac masses and have no density")
    chain = jax.lax.stop_gradient(jnp.asarray(chain, jnp.float32))          # the trajectory is fixed data
    K = cfg.num_steps
    if chain.ndim != 4 or chain.shape[1] != K + 1:
        raise ValueError(f"chain must be (B, {K + 1}, H, D), got {chain.shape}")
    delta = 1.0 / K
    n = chain.shape[2] * chain.shape[3]                                       # 320 coordinates
    x0 = chain[:, 0]
    initial = -0.5 * (jnp.sum(x0**2, axis=(1, 2)) + n * LOG_2PI)
    trans, means = [], []
    for k, (s, sd) in enumerate(zip(cfg.times(), cfg.stds())):
        x, y = chain[:, k], chain[:, k + 1]
        v = pv.velocity(model, cache, x, s)
        c = 0.5 * cfg.sigma**2 * (x + (1.0 - s) * v)
        mu = x - delta * (v + c)
        var = jnp.float32(sd) ** 2
        lp = -0.5 * (jnp.sum((y - mu) ** 2, axis=(1, 2)) / var + n * jnp.log(2.0 * jnp.pi * var))
        trans.append(lp)
        means.append(mu)
    transitions = jnp.stack(trans, axis=1)
    return {"total": initial + jnp.sum(transitions, axis=1), "initial": initial, "transitions": transitions,
            "means": jnp.stack(means, axis=1), "stds": jnp.asarray(cfg.stds(), jnp.float32)}
