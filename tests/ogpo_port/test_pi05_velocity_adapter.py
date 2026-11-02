"""Task 5: the real pi0.5 velocity adapter (src/clvla/ogpo/pi05_velocity.py) against OpenPI's own code paths.

Needs the pi05_libero checkpoint, the processed 256 px LIBERO HDF5 and (for reasonable speed) one GPU. Off by default:
  CLVLA_RUN_PI05_TESTS=1 CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy -m pytest tests/ogpo_port/test_pi05_velocity_adapter.py -v -s

Real observations: frames of the pre-action aligned 256 px HDF5 are turned into the raw-env observation dict and passed
through our evaluation's `clvla.policy.build_observation` (rotate 180, resize_with_pad 224, 8-D state, prompt) and the
policy's own input transform (normalisation, tokenisation): the exact path of the LIBERO evaluation client + server.
bf16 tolerance: the model runs in bfloat16 (eps 7.8e-3 relative); OpenPI-path comparisons use atol 3e-3 as before.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("CLVLA_RUN_PI05_TESTS") != "1":
    pytest.skip("pi0.5 adapter tests are opt-in (CLVLA_RUN_PI05_TESTS=1)", allow_module_level=True)

import h5py  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BF16_ATOL = 3e-3
FRAMES = ((78, "demo_0", 40), (35, "demo_3", 20), (42, "demo_5", 60))   # (LIBERO-90 index, demo, frame)
S_GRID = (1.0, 0.7, 0.3, 0.1)
RESULTS = {}


def _peak_gib():
    st = jax.local_devices()[0].memory_stats() or {}
    return round((st.get("peak_bytes_in_use") or 0) / 2**30, 2)


@pytest.fixture(scope="module")
def env():
    from clvla.config import load_experiment_config
    from clvla.ogpo import pi05_velocity as pv
    from clvla.policy import build_observation, load_official_policy
    from clvla.tasks import load_manifest

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    manifest = load_manifest(cfg.resolve(cfg.task_manifest))
    policy, _ = load_official_policy(
        cfg.checkpoint.openpi_config, cfg.checkpoint.local_dir, expected_asset_id=cfg.checkpoint.norm_stats_asset_id,
        state_dim=cfg.policy.state_dim, action_dim=cfg.policy.action_dim, action_horizon=cfg.policy.action_horizon)
    model = next(v for v in vars(policy).values() if hasattr(v, "embed_prefix") and hasattr(v, "embed_suffix"))
    proc = Path(os.environ["CLVLA_PROCESSED_ROOT"]) / "libero90_aligned256_v1" / "hdf5"
    by_idx = {t.libero_90_index: t for t in manifest.tasks}
    dicts, states = [], []
    for idx, demo, t in FRAMES:
        task = by_idx[idx]
        with h5py.File(proc / f"{task.name}_demo.hdf5", "r") as f:
            g = f["data"][demo]["obs"]
            raw = {"agentview_image": g["agentview_rgb"][t], "robot0_eye_in_hand_image": g["eye_in_hand_rgb"][t],
                   "robot0_eef_pos": g["ee_pos"][t], "robot0_eef_quat": g["ee_quat"][t],
                   "robot0_gripper_qpos": g["gripper_qpos"][t]}
            stored_state = g["state"][t]
        req = build_observation(raw, task.instruction, cfg.policy.resize_size)
        states.append((req["observation/state"], stored_state))
        dicts.append(policy._input_transform(req))
    keep = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")

    def observation(ids):
        from openpi.models import model as _model

        ds = [{k: dicts[i][k] for k in keep} for i in ids]
        return _model.Observation.from_dict(jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *ds))

    RESULTS["checkpoint"] = {"config": cfg.checkpoint.openpi_config, "dir": str(cfg.checkpoint.local_dir),
                             "action_horizon": model.action_horizon, "action_dim": model.action_dim,
                             "jax": jax.__version__, "backend": jax.default_backend(), "devices": str(jax.devices())}
    return {"cfg": cfg, "policy": policy, "model": model, "pv": pv, "observation": observation, "states": states}


def noise(seed, b, model):
    return jax.random.normal(jax.random.key(seed), (b, model.action_horizon, model.action_dim), jnp.float32)


def native_joint_velocity(model, observation, x_t, s):
    """OpenPI's native velocity path WITHOUT a KV cache: pi0.py compute_loss lines 203-212 (one joint prefix+suffix pass)."""
    from openpi.models import model as _model
    from openpi.models.pi0 import make_attn_mask

    obs = _model.preprocess_observation(None, observation, train=False)
    time = jnp.broadcast_to(jnp.asarray(s, jnp.float32), (x_t.shape[0],))
    prefix_tokens, prefix_mask, prefix_ar_mask = model.embed_prefix(obs)
    suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = model.embed_suffix(obs, x_t, time)
    input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
    ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
    attn_mask = make_attn_mask(input_mask, ar_mask)
    positions = jnp.cumsum(input_mask, axis=1) - 1
    (_, suffix_out), _ = model.PaliGemma.llm([prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions,
                                             adarms_cond=[None, adarms_cond])
    return model.action_out_proj(suffix_out[:, -model.action_horizon:]).astype(jnp.float32)


def test_0_observation_path_matches_stored_state(env):
    for built, stored in env["states"]:
        np.testing.assert_allclose(built, stored, atol=1e-5)    # build_observation's 8-D state == HDF5 obs/state


def test_1_shapes_and_2_finite(env):
    m, pv = env["model"], env["pv"]
    for ids in ((0,), (0, 1, 2)):
        cache = pv.build_prefix(m, env["observation"](ids))
        for s in S_GRID:
            v = pv.velocity(m, cache, noise(1, len(ids), m), s)
            assert v.shape == (len(ids), 10, 32) and v.dtype == jnp.float32
            assert bool(jnp.all(jnp.isfinite(v)))
    RESULTS["peak_gib_forward"] = _peak_gib()


def test_3_adapter_vs_native_joint_forward(env):
    m, pv = env["model"], env["pv"]
    obs = env["observation"]((0, 1, 2))
    cache = pv.build_prefix(m, obs)
    errs = {}
    for s in S_GRID:
        x = noise(2, 3, m) * s + (1 - s) * 0.3 * noise(3, 3, m)       # a plausible x_s
        a = np.asarray(pv.velocity(m, cache, x, s))
        n = np.asarray(native_joint_velocity(m, obs, x, s))
        errs[s] = float(np.abs(a - n).max())
        np.testing.assert_allclose(a, n, atol=BF16_ATOL, rtol=0)
    RESULTS["adapter_vs_native_joint_max_abs"] = errs
    print(f"\n  [3] adapter (KV cache) vs native joint pass, max |dv| per s: {errs}")


def test_4_deterministic_trajectory_vs_flow_sample_and_openpi_sample_actions(env):
    import importlib.util

    m, pv = env["model"], env["pv"]
    spec = importlib.util.spec_from_file_location("inspect_pi05", ROOT / "scripts/inspect_pi05.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    obs = env["observation"]((0, 1, 2))
    eps = noise(4, 3, m)
    xs_adapter = pv.ode_sample(m, pv.build_prefix(m, obs), eps, num_steps=10)
    xs_flow, _, ts = mod.Flow(m, obs).sample(eps, num_steps=10)
    assert len(xs_adapter) == len(xs_flow) == 11
    step_err = [float(np.abs(np.asarray(a) - np.asarray(b)).max()) for a, b in zip(xs_adapter, xs_flow)]
    x0_native = np.asarray(m.sample_actions(jax.random.key(0), obs, num_steps=10, noise=eps))
    native_err = float(np.abs(np.asarray(xs_adapter[-1]) - x0_native).max())
    RESULTS["trajectory_vs_flow_sample_max_abs_per_step"] = step_err
    RESULTS["final_vs_openpi_sample_actions_max_abs"] = native_err
    print(f"\n  [4] adapter vs Flow.sample, max |dx| per step: {[f'{e:.2e}' for e in step_err]}")
    print(f"      adapter x_0 vs OpenPI Pi0.sample_actions x_0 (same noise): max |dx| = {native_err:.2e}")
    for e in step_err:
        assert e <= BF16_ATOL
    assert native_err <= BF16_ATOL


def test_5_prefix_is_deterministic(env):
    m, pv = env["model"], env["pv"]
    obs = env["observation"]((0, 1, 2))
    c1, c2 = pv.build_prefix(m, obs), pv.build_prefix(m, obs)
    assert np.array_equal(np.asarray(c1.prefix_mask), np.asarray(c2.prefix_mask))
    for l1, l2 in zip(jax.tree_util.tree_leaves(c1.kv_cache), jax.tree_util.tree_leaves(c2.kv_cache)):
        assert np.array_equal(np.asarray(l1, np.float32), np.asarray(l2, np.float32))
    x = noise(5, 3, m)
    assert np.array_equal(np.asarray(pv.velocity(m, c1, x, 0.5)), np.asarray(pv.velocity(m, c2, x, 0.5)))
    RESULTS["kv_cache_shapes"] = [list(leaf.shape) for leaf in jax.tree_util.tree_leaves(c1.kv_cache)]


def _bf16_ulp(a):
    """Spacing of bfloat16 numbers at |a| (8 significand bits): 2^(floor(log2|a|) - 7); tiny values use the subnormal-free floor."""
    a = np.maximum(np.abs(a), 2.0**-126)
    return 2.0 ** (np.floor(np.log2(a)) - 7)


def test_6a_batch_members_do_not_influence_each_other(env):
    """Same batch SHAPE, so identical GEMM shapes: sample 0's velocity must be BITWISE identical whatever the other
    batch members are (different observations, permuted order). This is the test that catches cross-sample leakage."""
    m, pv = env["model"], env["pv"]
    x = noise(6, 3, m)
    xa = x
    v_a = np.asarray(pv.velocity(m, pv.build_prefix(m, env["observation"]((0, 1, 2))), xa, 0.6))
    v_b = np.asarray(pv.velocity(m, pv.build_prefix(m, env["observation"]((0, 2, 1))), xa[jnp.array([0, 2, 1])], 0.6))
    v_c = np.asarray(pv.velocity(m, pv.build_prefix(m, env["observation"]((0, 0, 0))),
                                 jnp.stack([xa[0], xa[0], xa[0]]), 0.6))
    assert np.array_equal(v_a[0], v_b[0])                 # other members permuted: sample 0 unchanged, bitwise
    assert np.array_equal(v_a[1], v_b[2]) and np.array_equal(v_a[2], v_b[1])
    assert np.array_equal(v_a[0], v_c[0])                 # other members replaced by copies of sample 0: unchanged
    assert np.array_equal(v_c[0], v_c[1]) and np.array_equal(v_c[0], v_c[2])


def test_6b_batch_of_three_vs_batches_of_one_within_bf16_rounding(env):
    """Different batch SHAPES change the GEMM shapes and hence bfloat16 accumulation order (XLA's deterministic flags
    make a given shape reproducible, not shape-invariant). The bound is ABSOLUTE at the chunk's scale: every entry must
    agree within 2 bf16 ulps of max|v| over the chunk. (A per-entry ulp bound is wrong: entries near 0 are differences of
    O(1) activations inside the last layer, so their rounding error is set by that O(1) scale, not by their own size.)"""
    m, pv = env["model"], env["pv"]
    x = noise(6, 3, m)
    vb = np.asarray(pv.velocity(m, pv.build_prefix(m, env["observation"]((0, 1, 2))), x, 0.6))
    stats = []
    for i in range(3):
        v1 = np.asarray(pv.velocity(m, pv.build_prefix(m, env["observation"]((i,))), x[i:i + 1], 0.6))[0]
        diff = np.abs(v1 - vb[i])
        scale_ulp = float(_bf16_ulp(max(np.abs(v1).max(), np.abs(vb[i]).max())))
        ulps = diff / scale_ulp
        stats.append({"max_abs": float(diff.max()), "mean_abs": float(diff.mean()), "max_ulps_at_chunk_scale": float(ulps.max()),
                      "chunk_max_abs_v": float(np.abs(vb[i]).max()), "chunk_scale_ulp": scale_ulp,
                      "frac_differing": float(np.mean(diff > 0)),
                      "hist_in_chunk_ulps": {str(k): int(np.sum(np.ceil(ulps) == k)) for k in range(0, 4)}})
        assert ulps.max() <= 2.0, stats[-1]
    RESULTS["batch1_vs_batch3"] = stats
    print("\n  [6b] batch-1 vs batch-3 (per sample): " + "; ".join(
        f"max|v| {d['chunk_max_abs_v']:.3f} (bf16 ulp {d['chunk_scale_ulp']:.4f}), max|dv| {d['max_abs']:.4f} = "
        f"{d['max_ulps_at_chunk_scale']:.2f} ulp, mean|dv| {d['mean_abs']:.4f}, {100 * d['frac_differing']:.0f}% of 320 differ, "
        f"hist(ceil ulps) {d['hist_in_chunk_ulps']}" for d in stats))


GROUP_RULES = (("vlm_siglip", r"^PaliGemma/img/"), ("action_expert", r"^PaliGemma/llm/.*_1(/|$)"),
               ("vlm_gemma", r"^PaliGemma/llm/"),
               ("action_heads", r"^(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/"))


def _grad_groups(model, obs, stop_gradient):
    from flax import nnx, traverse_util

    from clvla.ogpo import pi05_velocity as pv

    x = noise(7, obs.state.shape[0], model)

    def loss(m):
        return jnp.sum(pv.velocity(m, pv.build_prefix(m, obs, stop_gradient=stop_gradient), x, 0.5) ** 2)

    grads = nnx.grad(loss)(model)
    flat = traverse_util.flatten_dict(grads.to_pure_dict(), sep="/")
    out = {}
    for path, g in flat.items():
        name = next((n for n, rx in GROUP_RULES if re.search(rx, path)), "other")
        a = np.asarray(g, np.float32)
        d = out.setdefault(name, {"leaves": 0, "params": 0, "nonzero_leaves": 0, "finite": True, "sq": 0.0})
        d["leaves"] += 1; d["params"] += a.size; d["nonzero_leaves"] += int(np.any(a != 0))
        d["finite"] &= bool(np.all(np.isfinite(a))); d["sq"] += float(np.sum(a.astype(np.float64) ** 2))
    for d in out.values():
        d["norm"] = d.pop("sq") ** 0.5
    return out


def test_7_frozen_vlm_gets_no_gradient_action_expert_does(env):
    obs = env["observation"]((0,))
    g = _grad_groups(env["model"], obs, stop_gradient=True)
    RESULTS["grads_stop_gradient"] = g
    RESULTS["peak_gib_after_grad"] = _peak_gib()
    print("\n  [7] gradient of sum(v^2) w.r.t. every parameter (adapter, stop_gradient on the prefix KV cache):")
    for k, d in g.items():
        print(f"      {k:14s} leaves {d['leaves']:4d}  params {d['params']:>12,d}  nonzero leaves {d['nonzero_leaves']:4d}  "
              f"finite {d['finite']}  |grad| {d['norm']:.3e}")
    assert set(g) <= {"vlm_siglip", "vlm_gemma", "action_expert", "action_heads"}, f"unclassified params: {set(g)}"
    for k in ("vlm_siglip", "vlm_gemma"):
        assert g[k]["norm"] == 0.0 and g[k]["nonzero_leaves"] == 0
    for k in ("action_expert", "action_heads"):
        assert g[k]["finite"] and g[k]["norm"] > 0 and g[k]["nonzero_leaves"] > 0
    # negative control: WITHOUT stop_gradient the prefix pass is differentiated and the VLM receives gradient
    g2 = _grad_groups(env["model"], obs, stop_gradient=False)
    RESULTS["grads_no_stop_gradient_vlm_gemma_norm"] = g2["vlm_gemma"]["norm"]
    print(f"      control without stop_gradient: |grad| vlm_gemma {g2['vlm_gemma']['norm']:.3e}, "
          f"vlm_siglip {g2['vlm_siglip']['norm']:.3e}")
    assert g2["vlm_gemma"]["norm"] > 0


def test_zz_write_results(env):
    out = ROOT / "outputs" / "ogpo_port"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "task5_pi05_velocity_adapter.json"
    merged = json.loads(f.read_text()) if f.exists() else {}
    merged.update(RESULTS)
    f.write_text(json.dumps(merged, indent=1, default=str))
    print(f"\n  results -> {out / 'task5_pi05_velocity_adapter.json'}")
