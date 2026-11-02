"""Task 7: exact unnormalised path log-probability of frozen Task 6 chains on the real pi05_libero weights.

Opt-in, run in its OWN process (one model copy per process):
  CLVLA_RUN_PI05_TESTS=1 CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy -m pytest tests/ogpo_port/test_pi05_path_logprob.py -v -s
Chains come from the frozen sampler (pi05_sde.sde_sample) with explicitly injected x0/eps; likelihoods are recomputed by
pi05_path_logprob.path_log_prob with the SAME model, observation, prefix cache and batch shape. Batch 1 and batch 3 are
separate experiments and are never compared with each other.
Tolerances for Gaussian sums: fp32 accumulation over 320 terms, bounded per value by 16 * eps32 * sum|terms| (computed in
the float64 reference); every bound and every measured error is recorded.
"""

from __future__ import annotations

import json
import math
import os
import re
import time

import numpy as np
import pytest

if os.environ.get("CLVLA_RUN_PI05_TESTS") != "1":
    pytest.skip("pi0.5 path-likelihood tests are opt-in (CLVLA_RUN_PI05_TESTS=1)", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pi05_real_env import ROOT, load_real_env  # noqa: E402

K, SIGMA = 10, 0.05
BATCHES = {"b1": (0,), "b3": (0, 1, 2)}
EPS32 = float(np.finfo(np.float32).eps)
LOG2PI = math.log(2.0 * math.pi)
RESULTS = {}


def _peak_gib():
    st = jax.local_devices()[0].memory_stats() or {}
    return round((st.get("peak_bytes_in_use") or 0) / 2**30, 2)


def injected(seed, b):
    r = np.random.default_rng(7000 + seed)
    return (r.standard_normal((b, 10, 32)).astype(np.float32), r.standard_normal((K, b, 10, 32)).astype(np.float32))


@pytest.fixture(scope="module")
def env():
    from clvla.ogpo import pi05_path_logprob as plp
    from clvla.ogpo import pi05_sde as sde
    from clvla.ogpo import pi05_velocity as pv

    e = load_real_env()
    m = e["model"]
    runs = {}
    for name, ids in BATCHES.items():
        cache = pv.build_prefix(m, e["observation"](ids))
        x0, eps = injected(0, len(ids))
        samp = sde.sde_sample(m, cache, x0, eps)                      # FROZEN Task 6 sampler
        t0 = time.perf_counter()
        lp = plp.path_log_prob(m, cache, samp["chain"])
        jax.block_until_ready(lp["total"])
        RESULTS[f"first_logprob_call_s_{name}"] = time.perf_counter() - t0
        runs[name] = {"cache": cache, "x0": x0, "eps": eps,
                      "samp": {k: np.asarray(v) for k, v in samp.items() if hasattr(v, "shape")},
                      "lp": {k: np.asarray(v) for k, v in lp.items()}}
    RESULTS["peak_gib_after_sampling_and_logprob"] = _peak_gib()
    return {**e, "pv": pv, "sde": sde, "plp": plp, "runs": runs}


def gauss_ref(y, mu, sd):
    """float64 log N(y; mu, sd^2 I) per sample, summed over (10, 32), plus sum|terms| for the fp32 error bound."""
    y, mu = y.astype(np.float64), mu.astype(np.float64)
    terms = -0.5 * ((y - mu) ** 2 / sd**2 + math.log(2.0 * math.pi * sd**2))
    return terms.sum(axis=(1, 2)), np.abs(terms).sum(axis=(1, 2))


def init_ref(x0):
    terms = -0.5 * (x0.astype(np.float64) ** 2 + LOG2PI)
    return terms.sum(axis=(1, 2)), np.abs(terms).sum(axis=(1, 2))


def check(got, ref, absum, label, store):
    bound = 16 * EPS32 * absum
    err = np.abs(np.asarray(got, np.float64) - ref)
    store.append({"label": label, "max_abs_err": float(err.max()), "max_rel_err": float((err / np.abs(ref)).max()),
                  "bound_min": float(bound.min()), "err_over_bound_max": float((err / bound).max())})
    assert np.all(err <= bound), store[-1]


@pytest.mark.parametrize("name", list(BATCHES))
def test_1_shape_and_finite(env, name):
    lp, b = env["runs"][name]["lp"], len(BATCHES[name])
    assert lp["total"].shape == (b,) and lp["initial"].shape == (b,) and lp["transitions"].shape == (b, K)
    assert lp["means"].shape == (b, K, 10, 32) and lp["total"].dtype == np.float32
    for k in ("total", "initial", "transitions", "means"):
        assert np.all(np.isfinite(lp[k])), k


@pytest.mark.parametrize("name", list(BATCHES))
def test_2_independent_float64_reference(env, name):
    """Every transition log-density and the initial density vs an independent float64 NumPy Gaussian calculation,
    using the module's RECOMPUTED means and the stored chain."""
    run = env["runs"][name]
    lp, chain, stds = run["lp"], run["samp"]["chain"], run["lp"]["stds"]
    errs = []
    ref0, abs0 = init_ref(chain[:, 0])
    check(lp["initial"], ref0, abs0, "initial", errs)
    for k in range(K):
        ref, absum = gauss_ref(chain[:, k + 1], lp["means"][:, k], float(stds[k]))
        check(lp["transitions"][:, k], ref, absum, f"transition_{k}", errs)
    RESULTS[f"reference_errors_{name}"] = {
        "max_abs_err": max(e["max_abs_err"] for e in errs), "mean_abs_err": float(np.mean([e["max_abs_err"] for e in errs])),
        "max_rel_err": max(e["max_rel_err"] for e in errs), "max_err_over_bound": max(e["err_over_bound_max"] for e in errs),
        "per_term": errs}
    np.testing.assert_allclose(stds, [SIGMA * math.sqrt(1 - k / K) for k in range(K)], rtol=1e-7)


@pytest.mark.parametrize("name", list(BATCHES))
def test_3_recomputed_means_equal_sampler_means(env, name):
    run = env["runs"][name]
    diff = np.abs(run["lp"]["means"] - run["samp"]["means"])
    RESULTS[f"recomputed_vs_sampler_means_max_abs_{name}"] = float(diff.max())
    assert np.array_equal(run["lp"]["means"], run["samp"]["means"])     # same shape, weights, cache, ops: bitwise


@pytest.mark.parametrize("name", list(BATCHES))
def test_4_full_path_accounting(env, name):
    lp = env["runs"][name]["lp"]
    parts = lp["initial"].astype(np.float64) + lp["transitions"].astype(np.float64).sum(axis=1)
    absum = np.abs(lp["initial"].astype(np.float64)) + np.abs(lp["transitions"].astype(np.float64)).sum(axis=1)
    err = np.abs(lp["total"].astype(np.float64) - parts)
    RESULTS[f"accounting_max_abs_{name}"] = float(err.max())
    assert lp["transitions"].shape[1] == K                              # all ten transitions present
    assert np.all(err <= 16 * EPS32 * absum)


@pytest.mark.parametrize("name", list(BATCHES))
def test_5_self_consistency_with_sampler_saved_means(env, name):
    """Recomputed path log-likelihood vs the likelihood computed directly (float64) from the sampler's SAVED means."""
    run = env["runs"][name]
    chain, smeans, stds = run["samp"]["chain"], run["samp"]["means"], run["samp"]["stds"]
    ref, absum = init_ref(chain[:, 0])
    for k in range(K):
        r, a = gauss_ref(chain[:, k + 1], smeans[:, k], float(stds[k]))
        ref, absum = ref + r, absum + a
    err = np.abs(run["lp"]["total"].astype(np.float64) - ref)
    RESULTS[f"self_consistency_{name}"] = {"total_logprob": run["lp"]["total"].tolist(), "direct_float64": ref.tolist(),
                                           "max_abs_err": float(err.max()), "max_rel_err": float((err / np.abs(ref)).max())}
    assert np.all(err <= 16 * EPS32 * absum)


@pytest.mark.parametrize("name", list(BATCHES))
def test_6_identical_policy_ratio_is_one(env, name):
    run, m, plp = env["runs"][name], env["model"], env["plp"]
    a = np.asarray(plp.path_log_prob(m, run["cache"], run["samp"]["chain"])["total"])
    b = np.asarray(plp.path_log_prob(m, run["cache"], run["samp"]["chain"])["total"])
    d = a.astype(np.float64) - b.astype(np.float64)
    RESULTS[f"identical_policy_{name}"] = {"log_ratio": d.tolist(), "ratio": np.exp(d).tolist()}
    assert np.all(d == 0.0) and np.all(np.exp(d) == 1.0)
    assert np.array_equal(a, run["lp"]["total"])                       # also equals the fixture's first evaluation


GROUP_RULES = (("vlm_siglip", r"^PaliGemma/img/"), ("action_expert", r"^PaliGemma/llm/.*_1(/|$)"),
               ("vlm_gemma", r"^PaliGemma/llm/"),
               ("action_heads", r"^(action_in_proj|action_out_proj|time_mlp_in|time_mlp_out)/"))


def test_7_gradient_routing_for_a_fixed_trajectory(env):
    from flax import nnx, traverse_util

    run, plp, pv = env["runs"]["b1"], env["plp"], env["pv"]
    obs = env["observation"](BATCHES["b1"])
    chain = jnp.asarray(run["samp"]["chain"])

    def loss(m):
        return jnp.sum(plp.path_log_prob(m, pv.build_prefix(m, obs), chain)["total"])

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
    # the stored trajectory is fixed data: no gradient w.r.t. the chain itself
    g_chain = jax.grad(lambda c: jnp.sum(plp.path_log_prob(env["model"], run["cache"], c)["total"]))(chain)
    RESULTS["grads_logprob"] = groups
    RESULTS["grad_wrt_chain_max_abs"] = float(jnp.abs(g_chain).max())
    RESULTS["peak_gib_after_grad"] = _peak_gib()
    print("\n  [7] grad of sum log p(X|o) (fixed X):")
    for k, d in groups.items():
        print(f"      {k:14s} leaves {d['leaves']:3d} nonzero {d['nonzero']:3d} finite {d['finite']} |grad| {d['norm']:.3e}")
    print(f"      grad w.r.t. the stored chain: max |g| = {RESULTS['grad_wrt_chain_max_abs']:.1e}")
    assert set(groups) <= {"vlm_siglip", "vlm_gemma", "action_expert", "action_heads"}
    assert groups["vlm_siglip"]["norm"] == 0.0 and groups["vlm_gemma"]["norm"] == 0.0
    for k in ("action_expert", "action_heads"):
        assert groups[k]["finite"] and groups[k]["norm"] > 0 and groups[k]["nonzero"] == groups[k]["leaves"]
    assert RESULTS["grad_wrt_chain_max_abs"] == 0.0


@pytest.mark.parametrize("name", list(BATCHES))
def test_8_batch_shapes_are_separate_experiments_and_runtime(env, name):
    """Each batch shape is sampled AND recomputed at that shape (fixture). Also records warm runtime."""
    run, m, plp = env["runs"][name], env["model"], env["plp"]
    assert run["samp"]["chain"].shape[0] == len(BATCHES[name]) == run["lp"]["total"].shape[0]
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(plp.path_log_prob(m, run["cache"], run["samp"]["chain"])["total"])
        t.append(time.perf_counter() - t0)
    RESULTS[f"runtime_logprob_{name}"] = {"median_s": float(np.median(t)), "per_sample_s": float(np.median(t)) / len(BATCHES[name])}
    RESULTS[f"logprob_values_{name}"] = {"total": run["lp"]["total"].tolist(), "initial": run["lp"]["initial"].tolist(),
                                         "transitions_sample0": run["lp"]["transitions"][0].tolist()}


def test_zz_write_results():
    out = ROOT / "outputs" / "ogpo_port"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "task7_pi05_path_logprob.json"
    f.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\n  results -> {f}")
