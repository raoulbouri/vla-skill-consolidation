"""`ogpo_discrete` stochastic flow sampler for the real pi0.5 action expert (OpenPI convention: s = 1 noise -> s = 0 action).

Frozen configuration (tests/ogpo_port/README.md): K = 10, s_k = 1 - k/K, Delta = 1/K, base sigma = 0.05, score correction ON,
sigma_k = sigma * sqrt(s_k) (`ogpo_discrete`, NOT Euler-Maruyama), all 10 x 32 = 320 latent coordinates stochastic, no
clipping of any kind, no normalisation. Per step (verified against OGPO's code on CPU, Tasks 1-2):

    v_k     = v_pi(x_k, s_k, o)                                  (pi05_velocity.velocity: the verified adapter)
    c_k     = sigma^2 / 2 * [x_k + (1 - s_k) v_k]
    mu_k    = x_k - Delta * (v_k + c_k)
    x_{k+1} = mu_k + sigma_k * eps_k,     eps_k ~ N(0, I_320)    (eps injected by the caller)

Precision: the transformer runs exactly as in OpenPI (bfloat16 weights/activations); `velocity` casts its output to float32
(exact). The latent state, means, standard deviations and noise are float32, the same precision OpenPI's own sampler keeps
x_t in (pi0.py: float32 noise, x_t + dt * v_t). The output is the RAW latent chain in normalised, 32-dim padded action space;
the environment transform (unnormalise, keep 7 dims, execute 5 rows) is applied elsewhere and never here.
This sampler's final latent is NOT the deterministic ODE output; only the sigma = 0 limit coincides with `ode_sample`.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any, List

from clvla.ogpo import pi05_velocity as pv

NOISE_MODE = "ogpo_discrete"


@dataclasses.dataclass(frozen=True)
class SDEConfig:
    num_steps: int = 10
    sigma: float = 0.05
    noise_mode: str = NOISE_MODE
    score_correction: bool = True

    def __post_init__(self):
        if self.noise_mode != NOISE_MODE:
            raise ValueError(f"only noise_mode={NOISE_MODE!r} is implemented (Euler-Maruyama is a separate ablation)")
        if self.num_steps < 1 or self.sigma < 0:
            raise ValueError("num_steps >= 1 and sigma >= 0 required")

    def times(self) -> List[float]:
        return [1.0 - k / self.num_steps for k in range(self.num_steps)]

    def stds(self) -> List[float]:
        return [self.sigma * math.sqrt(s) for s in self.times()]


def sde_sample(model: Any, cache: "pv.PrefixCache", x0: Any, eps: Any, cfg: SDEConfig = SDEConfig()) -> dict:
    """Stochastic chain from an existing prefix cache (built ONCE by the caller and reused for all K steps).

    x0: (B, 10, 32) initial noise; eps: (K, B, 10, 32) per-step standard normal noise. Returns float32 arrays:
      chain (B, K+1, 10, 32) = [x_0, ..., x_K];  means (B, K, 10, 32);  velocities (B, K, 10, 32);
      stds (K,) = sigma * sqrt(s_k);  times (K,) = s_k.
    """
    import jax.numpy as jnp

    delta = 1.0 / cfg.num_steps
    x = jnp.asarray(x0, jnp.float32)
    eps = jnp.asarray(eps, jnp.float32)
    if eps.shape != (cfg.num_steps,) + tuple(x.shape):
        raise ValueError(f"eps must have shape {(cfg.num_steps,) + tuple(x.shape)}, got {eps.shape}")
    chain, means, vels = [x], [], []
    for k, (s, sd) in enumerate(zip(cfg.times(), cfg.stds())):
        v = pv.velocity(model, cache, x, s)
        c = 0.5 * cfg.sigma**2 * (x + (1.0 - s) * v) if cfg.score_correction else jnp.zeros_like(x)
        mu = x - delta * (v + c)
        x = mu + jnp.float32(sd) * eps[k]
        chain.append(x)
        means.append(mu)
        vels.append(v)
    return {"chain": jnp.stack(chain, axis=1), "means": jnp.stack(means, axis=1), "velocities": jnp.stack(vels, axis=1),
            "stds": jnp.asarray(cfg.stds(), jnp.float32), "times": jnp.asarray(cfg.times(), jnp.float32),
            "noise_mode": cfg.noise_mode, "sigma": cfg.sigma}


def sde_sample_from_observation(model: Any, observation: Any, x0: Any, eps: Any, cfg: SDEConfig = SDEConfig()) -> dict:
    """Convenience: build the prefix KV cache once (gradient-stopped) and run `sde_sample`."""
    return sde_sample(model, pv.build_prefix(model, observation), x0, eps, cfg)
