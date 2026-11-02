"""Task 6: `ogpo_discrete` stochastic sampler on the real pi05_libero weights (src/clvla/ogpo/pi05_sde.py). No likelihoods.

Opt-in (needs the checkpoint, processed HDF5 and a GPU):
  CLVLA_RUN_PI05_TESTS=1 CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy -m pytest tests/ogpo_port/test_pi05_sde_sampler.py -v -s
Run each real-model module in its OWN pytest process: every module loads its own pi0.5 copy (~7 GB); two resident
copies plus a full-parameter gradient through the 10-step chain exceed a 32 GB GPU.
Observations: same real-frame path as Task 5 (build_observation + the policy's input transform). All noise is injected.
Every comparison is made at a MATCHING batch shape (Task 5 caveat). Nonzero-noise chains are never compared to the ODE.
Precision: model in bf16 (as trained); latent state / means / stds / eps in float32; references recomputed in float64.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
from pathlib import Path

import numpy as np
import pytest

if os.environ.get("CLVLA_RUN_PI05_TESTS") != "1":
    pytest.skip("pi0.5 sampler tests are opt-in (CLVLA_RUN_PI05_TESTS=1)", allow_module_level=True)

import h5py  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
FRAMES = ((78, "demo_0", 40), (35, "demo_3", 20), (42, "demo_5", 60))
K, SIGMA = 10, 0.05
BATCHES = {"b1": (0,), "b3": (0, 1, 2)}
RESULTS = {}


def _peak_gib():
    st = jax.local_devices()[0].memory_stats() or {}
    return round((st.get("peak_bytes_in_use") or 0) / 2**30, 2)


@pytest.fixture(scope="module")
def env():
    from clvla.config import load_experiment_config
    from clvla.ogpo import pi05_sde as sde
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
    dicts = []
    for idx, demo, t in FRAMES:
        task = by_idx[idx]
        with h5py.File(proc / f"{task.name}_demo.hdf5", "r") as f:
            g = f["data"][demo]["obs"]
            raw = {"agentview_image": g["agentview_rgb"][t], "robot0_eye_in_hand_image": g["eye_in_hand_rgb"][t],
                   "robot0_eef_pos": g["ee_pos"][t], "robot0_eef_quat": g["ee_quat"][t],
                   "robot0_gripper_qpos": g["gripper_qpos"][t]}
        dicts.append(policy._input_transform(build_observation(raw, task.instruction, cfg.policy.resize_size)))
    keep = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")

    def observation(ids):
        from openpi.models import model as _model

        ds = [{k: dicts[i][k] for k in keep} for i in ids]
        return _model.Observation.from_dict(jax.tree.map(lambda *xs: jnp.stack([jnp.asarray(x) for x in xs]), *ds))

    caches = {name: pv.build_prefix(model, observation(ids)) for name, ids in BATCHES.items()}
    return {"model": model, "pv": pv, "sde": sde, "observation": observation, "caches": caches}


def injected(seed, b):
    """Explicit x0 (B,10,32) and eps (K,B,10,32), float32, from NumPy (independent of JAX's RNG)."""
    r = np.random.default_rng(5000 + seed)
    return (r.standard_normal((b, 10, 32)).astype(np.float32), r.standard_normal((K, b, 10, 32)).astype(np.float32))


def run(env, name, seed=0, sigma=SIGMA):
    cache = env["caches"][name]
    x0, eps = injected(seed, len(BATCHES[name]))
    out = env["sde"].sde_sample(env["model"], cache, x0, eps, env["sde"].SDEConfig(sigma=sigma))
    return {k: (np.asarray(v) if hasattr(v, "shape") else v) for k, v in out.items()}, x0, eps


@pytest.mark.parametrize("name", list(BATCHES))
def test_1_2_shape_and_finite(env, name):
    out, _, _ = run(env, name)
    b = len(BATCHES[name])
    assert out["chain"].shape == (b, K + 1, 10, 32) and out["chain"].dtype == np.float32
    assert out["means"].shape == (b, K, 10, 32) and out["stds"].shape == (K,)
    for k in ("chain", "means", "velocities", "stds"):
        assert np.all(np.isfinite(out[k])), k
    RESULTS[f"peak_gib_after_{name}"] = _peak_gib()


@pytest.mark.parametrize("name", list(BATCHES))
def test_3_means_match_independent_float64_reference(env, name):
    """mu_k recomputed in float64 NumPy from the stored x_k and a fresh velocity query at the SAME batch shape."""
    out, _, _ = run(env, name)
    cache, m, pv = env["caches"][name], env["model"], env["pv"]
    errs = []
    for k in range(K):
        s = 1.0 - k / K
        x = out["chain"][:, k].astype(np.float64)
        v = np.asarray(pv.velocity(m, cache, jnp.asarray(out["chain"][:, k]), s)).astype(np.float64)
        np.testing.assert_array_equal(v, out["velocities"][:, k].astype(np.float64))     # same query, same shape
        mu_ref = x - (1.0 / K) * (v + 0.5 * SIGMA**2 * (x + (1.0 - s) * v))
        e = np.abs(out["means"][:, k].astype(np.float64) - mu_ref)
        errs.append((float(e.max()), float(e.mean())))
        np.testing.assert_allclose(out["means"][:, k], mu_ref, rtol=1e-6, atol=1e-6)     # float32 arithmetic vs float64
    RESULTS[f"mean_err_{name}"] = {"max": max(e[0] for e in errs), "mean": float(np.mean([e[1] for e in errs]))}


def test_4_stds_follow_ogpo_discrete_schedule(env):
    out, _, _ = run(env, "b1")
    expected = np.array([SIGMA * math.sqrt(1.0 - k / K) for k in range(K)])
    np.testing.assert_allclose(out["stds"], expected, rtol=1e-7)
    RESULTS["stds"] = [float(v) for v in out["stds"]]


@pytest.mark.parametrize("name", list(BATCHES))
def test_5_transitions_are_mean_plus_std_times_eps(env, name):
    out, x0, eps = run(env, name)
    np.testing.assert_array_equal(out["chain"][:, 0], x0)
    errs = []
    for k in range(K):
        ref = out["means"][:, k].astype(np.float64) + float(out["stds"][k]) * eps[k].astype(np.float64)
        e = np.abs(out["chain"][:, k + 1].astype(np.float64) - ref)
        errs.append((float(e.max()), float(e.mean())))
        np.testing.assert_allclose(out["chain"][:, k + 1], ref, rtol=1e-6, atol=1e-6)
    RESULTS[f"transition_err_{name}"] = {"max": max(e[0] for e in errs), "mean": float(np.mean([e[1] for e in errs]))}


@pytest.mark.parametrize("name", list(BATCHES))
def test_6_sigma_zero_matches_the_verified_ode_adapter(env, name):
    out, x0, _ = run(env, name, sigma=0.0)
    xs = env["pv"].ode_sample(env["model"], env["caches"][name], jnp.asarray(x0), num_steps=K)
    err = max(float(np.abs(out["chain"][:, k] - np.asarray(xs[k])).max()) for k in range(K + 1))
    RESULTS[f"ode_limit_max_abs_{name}"] = err
    assert err == 0.0, err                                    # same ops, same shape: bitwise


def test_7_final_step_is_stochastic(env):
    out, _, eps = run(env, "b1")
    np.testing.assert_allclose(out["stds"][-1], SIGMA * math.sqrt(0.1), rtol=1e-7)
    d = out["chain"][:, K] - out["means"][:, K - 1]
    assert np.abs(d).max() > 0
    np.testing.assert_allclose(d, out["stds"][-1] * eps[K - 1], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("name", list(BATCHES))
def test_8_identical_inputs_give_identical_chains(env, name):
    a, _, _ = run(env, name)
    b, _, _ = run(env, name)
    assert np.array_equal(a["chain"], b["chain"]) and np.array_equal(a["means"], b["means"])


def test_9_prefix_built_once_and_reused_for_all_steps(env, monkeypatch):
    sde, pv = env["sde"], env["pv"]
    calls = {"build_prefix": 0, "velocity_caches": []}
    real_build, real_vel = pv.build_prefix, pv.velocity

    def counting_build(*a, **kw):
        calls["build_prefix"] += 1
        return real_build(*a, **kw)

    def recording_vel(model, cache, x, s):
        calls["velocity_caches"].append(id(cache))
        return real_vel(model, cache, x, s)

    monkeypatch.setattr(pv, "build_prefix", counting_build)
    monkeypatch.setattr(pv, "velocity", recording_vel)
    x0, eps = injected(0, 1)
    sde.sde_sample_from_observation(env["model"], env["observation"]((0,)), x0, eps)
    assert calls["build_prefix"] == 1
    assert len(calls["velocity_caches"]) == K and len(set(calls["velocity_caches"])) == 1


GROUP_RULES = (("vlm_siglip", r"^PaliGemma/img/"), ("action_expert", r"^PaliGemma/llm/.*_1(/|$)"),
               ("vlm_gemma", r"^PaliGemma/llm/"),
               ("action_heads", r"^(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/"))


def test_10_gradients_reach_expert_and_heads_not_vlm(env):
    from flax import nnx, traverse_util

    sde = env["sde"]
    obs = env["observation"]((0,))
    x0, eps = injected(1, 1)

    def loss(m):
        out = sde.sde_sample_from_observation(m, obs, x0, eps)
        return jnp.sum(out["chain"][:, -1] ** 2)

    grads = nnx.grad(loss)(env["model"])
    groups = {}
    for path, g in traverse_util.flatten_dict(grads.to_pure_dict(), sep="/").items():
        name = next((n for n, rx in GROUP_RULES if re.search(rx, path)), "other")
        a = np.asarray(g, np.float32)
        d = groups.setdefault(name, {"leaves": 0, "nonzero": 0, "finite": True, "sq": 0.0})
        d["leaves"] += 1; d["nonzero"] += int(np.any(a != 0)); d["finite"] &= bool(np.all(np.isfinite(a)))
        d["sq"] += float(np.sum(a.astype(np.float64) ** 2))
    for d in groups.values():
        d["norm"] = d.pop("sq") ** 0.5
    RESULTS["grads_through_sde_chain"] = groups
    RESULTS["peak_gib_after_grad"] = _peak_gib()
    print("\n  [10] grad of sum(x_K^2) through the whole stochastic chain:")
    for k, d in groups.items():
        print(f"       {k:14s} leaves {d['leaves']:3d} nonzero {d['nonzero']:3d} finite {d['finite']} |grad| {d['norm']:.3e}")
    assert set(groups) <= {"vlm_siglip", "vlm_gemma", "action_expert", "action_heads"}
    assert groups["vlm_siglip"]["norm"] == 0.0 and groups["vlm_gemma"]["norm"] == 0.0
    for k in ("action_expert", "action_heads"):
        assert groups[k]["finite"] and groups[k]["norm"] > 0 and groups[k]["nonzero"] == groups[k]["leaves"]


@pytest.mark.parametrize("name", list(BATCHES))
def test_11_runtime_per_chain(env, name):
    """Wall time of one stochastic chain (10 velocity queries, prefix excluded) and of prefix + chain, after warm-up."""
    m, sde, pv = env["model"], env["sde"], env["pv"]
    x0, eps = injected(2, len(BATCHES[name]))
    obs = env["observation"](BATCHES[name])
    jax.block_until_ready(sde.sde_sample(m, env["caches"][name], x0, eps)["chain"])          # warm-up
    t = []
    for _ in range(3):
        t0 = time.perf_counter(); jax.block_until_ready(sde.sde_sample(m, env["caches"][name], x0, eps)["chain"])
        t.append(time.perf_counter() - t0)
    t0 = time.perf_counter()
    jax.block_until_ready(sde.sde_sample(m, pv.build_prefix(m, obs), x0, eps)["chain"])
    full = time.perf_counter() - t0
    RESULTS[f"runtime_{name}"] = {"chain_only_s_median": float(np.median(t)), "prefix_plus_chain_s": full,
                                  "per_sample_chain_s": float(np.median(t)) / len(BATCHES[name])}
    print(f"\n  [11] {name}: chain only {np.median(t):.3f} s (median of 3), prefix + chain {full:.3f} s")


def test_zz_write_results():
    out = ROOT / "outputs" / "ogpo_port"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "task6_pi05_sde_sampler.json"
    f.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\n  results -> {f}")
    for k, v in RESULTS.items():
        if not k.startswith("grads"):
            print(f"    {k}: {v}")
