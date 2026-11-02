"""LoRA student for the single-task OPD pilot: pi05_libero with `gemma_300m_lora` on the action expert only; Gemma 2B and SigLIP unadapted.
Trainable: expert LoRA leaves and the action projections and time MLP, which have no LoRA form. lora_b starts at zero and lora_a from a fixed seed, so every student starts identical and equal to the base function.
Forward parameters are BF16, as in OpenPI inference; FP32 masters live with the optimizer.
"""

from __future__ import annotations

import numpy as np

HEADS = ("action_in_proj/", "action_out_proj/", "time_mlp_in/", "time_mlp_out/")
LORA_A_STD = 0.01


def student_config():
    from openpi.models import pi0_config

    return pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m_lora")


def is_trainable(key: str) -> bool:
    return "lora" in key or key.startswith(HEADS)


def shapes():
    import jax
    from flax import nnx, traverse_util

    model = nnx.eval_shape(student_config().create, jax.random.key(0))
    return {k: v.shape for k, v in traverse_util.flatten_dict(nnx.split(model)[1].to_pure_dict(), sep="/").items()}


def initial_adapter(base_params_path: str, seed: int = 7) -> dict:
    from flax import traverse_util
    from openpi.models import model as _model

    base = traverse_util.flatten_dict(_model.restore_params(base_params_path, restore_type=np.ndarray), sep="/")
    r, out = np.random.default_rng(seed), {}
    for k, s in sorted(shapes().items()):
        if "lora_b" in k:
            out[k] = np.zeros(s, np.float32)
        elif "lora_a" in k:
            out[k] = (r.standard_normal(s) * LORA_A_STD).astype(np.float32)
        elif k.startswith(HEADS):
            out[k] = np.asarray(base[k], np.float32)
    return out


def build_student_model(base_model, adapter: dict):
    """Frozen leaves are the base model's own arrays."""
    import jax.numpy as jnp
    from flax import nnx, traverse_util

    flat = traverse_util.flatten_dict(nnx.split(base_model)[1].to_pure_dict(), sep="/")
    params = {k: v for k, v in flat.items() if not k.startswith(HEADS)}
    params.update({k: jnp.asarray(v).astype(jnp.bfloat16) for k, v in adapter.items()})
    return student_config().load(traverse_util.unflatten_dict(params, sep="/"))


def build_student_policy(conf, adapter) -> object:
    """The official policy with its model replaced: same transforms, same norm stats."""
    from clvla.ogpo.pi05_sde_policy import model_of
    from clvla.policy import load_official_policy
    from openpi.shared import nnx_utils

    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    a = {k.replace("__", "/"): adapter[k] for k in adapter.files} if hasattr(adapter, "files") else adapter
    model = build_student_model(model_of(policy), a)
    policy._model = model
    policy._sample_actions = nnx_utils.module_jit(model.sample_actions)
    return policy


def load_delta_teacher(base_model, delta_dir, R=None):
    """Expert teacher rebuilt from its exported changed leaves and the local base, all as BF16. Refuses to continue unless every leaf hash matches the original checkpoint's."""
    import hashlib
    import json

    import jax.numpy as jnp
    from flax import nnx, traverse_util
    from openpi.models import pi0_config

    def leaf_sha(v):
        x = np.asarray(v)
        return hashlib.sha256(str(x.dtype).encode() + str(x.shape).encode() + x.tobytes()).hexdigest()

    sha = json.loads((delta_dir / "leaf_sha.json").read_text())
    z = np.load(delta_dir / "delta.npz")
    flat_b = traverse_util.flatten_dict(nnx.split(base_model)[1].to_pure_dict(), sep="/")
    assert all(leaf_sha(flat_b[k]) == sha["base_bf16"][k] for k in sha["base_bf16"]), "local base differs from the teacher's base"
    params = {k: v for k, v in flat_b.items() if k not in sha["delta_keys"]}
    params.update({k: jnp.asarray(z[k.replace("/", "__")]).astype(jnp.bfloat16) for k in sha["delta_keys"]})
    assert set(params) == set(sha["teacher_bf16"]), "teacher leaf set differs"
    bad = [k for k, v in params.items() if leaf_sha(v) != sha["teacher_bf16"][k]]
    assert not bad, f"rebuilt teacher differs from the checkpoint: {bad[:5]}"
    if R is not None:
        R["teacher_rebuild"] = {"leaves_verified": len(params), "own_leaves": len(sha["delta_keys"]), "shared_with_base": len(params) - len(sha["delta_keys"])}
    conf = pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b", action_expert_variant="gemma_300m")
    return conf.load(traverse_util.unflatten_dict(params, sep="/")), 250, str(delta_dir)


def build_delta_teacher_policy(conf, delta_dir) -> object:
    """The official policy with the rebuilt teacher in place of its model: same transforms, same norm stats."""
    from clvla.ogpo.pi05_sde_policy import model_of
    from clvla.policy import load_official_policy
    from openpi.shared import nnx_utils

    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    model, _, _ = load_delta_teacher(model_of(policy), delta_dir)
    policy._model = model
    policy._sample_actions = nnx_utils.module_jit(model.sample_actions)
    return policy
