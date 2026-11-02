"""Student rollout policy for LIBERO: every action chunk comes from the frozen Task 6 `ogpo_discrete` sampler, never from OpenPI's ODE `sample_actions`.

`infer` follows `openpi.policies.policy.Policy.infer` with one change: the model output is `chain[:, K]` of `pi05_sde.sde_sample`. The policy's own input and output transforms are reused unchanged.
Noise is seeded per replanning event, so an event can be replayed from its stored observation and noise. There is no fallback: a sampler failure ends the episode.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np

from clvla.ogpo import pi05_sde as sde
from clvla.ogpo import pi05_velocity as pv

KEEP = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")


def tree_sha(tree: Any) -> str:
    """sha256 over dtype, shape and bytes of every leaf, in tree order."""
    import jax

    h = hashlib.sha256()
    for leaf in jax.tree.leaves(tree):
        # a jax.Array caches its host copy, so hash a temporary copy: hashing a whole model must not keep 6 GB on the host
        a = np.asarray(jax.numpy.array(leaf, copy=True) if isinstance(leaf, jax.Array) else leaf)
        h.update(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes())
        del a
    return h.hexdigest()


def event_noise(seed: int, cfg: "sde.SDEConfig", horizon: int = 10, dim: int = 32):
    r = np.random.default_rng(seed)
    return r.standard_normal((1, horizon, dim)).astype(np.float32), r.standard_normal((cfg.num_steps, 1, horizon, dim)).astype(np.float32)


def model_of(policy: Any) -> Any:
    return next(v for v in vars(policy).values() if hasattr(v, "embed_prefix") and hasattr(v, "embed_suffix"))


def model_inputs(policy: Any, request: Dict[str, Any]):
    """Stored or live request -> batched model Observation, through the policy's own input transform."""
    import jax
    import jax.numpy as jnp
    from openpi.models import model as _model

    inputs = policy._input_transform(jax.tree.map(lambda x: x, dict(request)))
    return _model.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], {k: inputs[k] for k in KEEP})), inputs


def make_sampler(model: Any, cfg: "sde.SDEConfig"):
    """Compiled prefix pass and chain sampler as pure functions of the parameters, so rollout and replay run the same programme."""
    import jax
    from flax import nnx

    gd, state = nnx.split(model)

    def prefix(st, obs):
        c = pv.build_prefix(nnx.merge(gd, st), obs)
        return c.observation, c.prefix_mask, c.kv_cache

    def sample(st, ob, mask, kv, x0, eps):
        o = sde.sde_sample(nnx.merge(gd, st), pv.PrefixCache(ob, mask, kv), x0, eps, cfg)
        return {k: o[k] for k in ("chain", "means", "velocities", "stds", "times")}

    jp, js = jax.jit(prefix), jax.jit(sample)

    def run(obs, x0, eps):
        return js(state, *jp(state, obs), x0, eps)

    return run, state


class SDERolloutPolicy:
    """Served through OpenPI's WebsocketPolicyServer. Writes one compressed npz per replanning event to `record_dir`."""

    def __init__(self, policy: Any, record_dir: Path, base_seed: int, cfg: "sde.SDEConfig" = sde.SDEConfig(), metadata: Any = None):
        self.policy, self.cfg, self.base_seed, self.record_dir = policy, cfg, base_seed, Path(record_dir)
        self.record_dir.mkdir(parents=True, exist_ok=True)
        self.metadata = metadata or {}
        self.events, self.sample_s = 0, []
        self.run, self.state = make_sampler(model_of(policy), cfg)

        def refuse(*_a, **_k):
            raise RuntimeError("deterministic sample_actions must not be used in an SDE rollout")

        policy._sample_actions = refuse

    def infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        import jax

        seed = self.base_seed + self.events
        x0, eps = event_noise(seed, self.cfg)
        obs, inputs = model_inputs(self.policy, request)
        t0 = time.perf_counter()
        out = jax.block_until_ready(self.run(obs, x0, eps))
        dt = time.perf_counter() - t0
        out = {k: np.asarray(v) for k, v in out.items()}
        physical = np.asarray(self.policy._output_transform({"state": np.asarray(inputs["state"]), "actions": out["chain"][0, self.cfg.num_steps]})["actions"])
        np.savez_compressed(self.record_dir / f"event_{self.events:04d}.npz", seed=seed, x0=x0, eps=eps, physical=physical, sample_s=dt, prompt=str(request["prompt"]),
                            observation_image=request["observation/image"], observation_wrist_image=request["observation/wrist_image"], observation_state=request["observation/state"],
                            tokenized_prompt=np.asarray(inputs["tokenized_prompt"]), tokenized_prompt_mask=np.asarray(inputs["tokenized_prompt_mask"]), model_state=np.asarray(inputs["state"]),
                            model_ready_sha=tree_sha({k: inputs[k] for k in KEEP}), **out)
        self.events += 1
        self.sample_s.append(dt)
        stats = jax.devices()[0].memory_stats() or {}
        (self.record_dir / "server_status.json").write_text(json.dumps({"events": self.events, "sample_s": self.sample_s, "peak_gpu_bytes": stats.get("peak_bytes_in_use")}))
        return {"actions": physical}
