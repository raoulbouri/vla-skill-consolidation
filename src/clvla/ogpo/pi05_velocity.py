"""Velocity adapter for the real pi0.5 action expert: v_pi(x_s, s, o) in the model's normalised latent action space.

It exposes exactly the per-step computation of OpenPI's own sampler (third_party/openpi/src/openpi/models/pi0.py,
`Pi0.sample_actions`, pinned commit 215abfb), split into its two halves:

  build_prefix   pi0.py:225, 234-237  preprocess the observation; images + prompt -> prefix tokens -> one VLM pass ->
                                      KV cache. Computed ONCE per observation and reused for every (x_s, s) query.
                                      The cache is wrapped in stop_gradient: gradients can only reach the action
                                      expert path (action_in_proj, time MLP, the `_1` expert weights, action_out_proj).
  velocity       pi0.py:239-269       the `step` body WITHOUT its Euler update: embed_suffix(o, x_s, s) -> action expert
                                      attending to the cached prefix -> action_out_proj -> v_pi (B, 10, 32).
  ode_sample     pi0.py:228, 271-278  the deterministic Euler loop x <- x + dt * v_pi, dt = -1/K, from s = 1 to s = 0.

Conventions (verified on CPU in tests/ogpo_port, Tasks 1-4): s = 1 is noise; v_pi ~ eps - a; OGPO's v_O = -v_pi(x, 1 - t).
Inputs are the model's own Observation (after the policy's input transforms: resize, normalisation, tokenisation);
outputs stay in the normalised, 32-dim padded action space (unnormalise / slice to 7 dims only at the environment).
Velocities are returned as float32 (the model computes in bfloat16; the cast is exact).
"""

from __future__ import annotations

import dataclasses
from typing import Any


@dataclasses.dataclass(frozen=True)
class PrefixCache:
    """Everything the action expert needs from the observation: the preprocessed observation (for embed_suffix),
    the prefix token mask and the per-layer KV cache of the prefix pass."""

    observation: Any
    prefix_mask: Any
    kv_cache: Any


def build_prefix(model: Any, observation: Any, *, stop_gradient: bool = True) -> PrefixCache:
    """pi0.py:225 + 234-237. `observation` is an openpi.models.model.Observation (batched)."""
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    obs = _model.preprocess_observation(None, observation, train=False)
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
    positions = jnp.cumsum(prefix_mask, axis=1) - 1
    _, kv_cache = model.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
    if stop_gradient:
        kv_cache = jax.lax.stop_gradient(kv_cache)
    return PrefixCache(observation=obs, prefix_mask=prefix_mask, kv_cache=kv_cache)


def velocity(model: Any, cache: PrefixCache, x_s: Any, s: Any) -> Any:
    """v_pi(x_s, s, o): pi0.py:239-269 without the Euler update. x_s (B, 10, 32); s scalar or (B,). -> float32 (B, 10, 32)."""
    import einops
    import jax.numpy as jnp

    from openpi.models.pi0 import make_attn_mask

    batch_size = x_s.shape[0]
    time = jnp.broadcast_to(jnp.asarray(s, jnp.float32), (batch_size,))
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(cache.observation, x_s, time)
    suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
    prefix_attn_mask = einops.repeat(cache.prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
    full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
    positions = jnp.sum(cache.prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
    (prefix_out, suffix_out), _ = model.PaliGemma.llm(
        [None, suffix_tokens], mask=full_attn_mask, positions=positions, kv_cache=cache.kv_cache,
        adarms_cond=[None, adarms_cond])
    assert prefix_out is None
    return model.action_out_proj(suffix_out[:, -model.action_horizon :]).astype(jnp.float32)


def ode_sample(model: Any, cache: PrefixCache, noise: Any, num_steps: int = 10) -> list:
    """Deterministic Euler integration in the OpenPI convention (pi0.py:228, 271, 276): returns [x_0=noise, ..., x_K]."""
    import jax.numpy as jnp

    dt = -1.0 / num_steps
    x, time = jnp.asarray(noise, jnp.float32), 1.0
    xs = [x]
    while time >= -dt / 2:
        x = x + dt * velocity(model, cache, x, time)
        time = time + dt
        xs.append(x)
    return xs
