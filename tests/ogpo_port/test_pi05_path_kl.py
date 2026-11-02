"""Task 8: teacher-student stochastic path KL on the real pi05_libero weights (EVALUATION ONLY: no optimiser, no KL gradient).

Opt-in, run in its OWN process (one model copy per process):
  CLVLA_RUN_PI05_TESTS=1 CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy -m pytest tests/ogpo_port/test_pi05_path_kl.py -v -s
Teacher = the loaded checkpoint, never modified. Students are separate nnx modules merged from the teacher's graph and state,
sharing every parameter array except ONE leaf (no second model copy): the identical student re-creates that leaf with zero
perturbation; the perturbed student adds a seeded Gaussian perturbation to it (recorded after the bf16 cast). The VLM is never
touched. Chains come from the STUDENT through the frozen Task 6 sampler with injected noise; both policies are evaluated on
that same stored chain at the same batch shape. Batch 1 and batch 3 are separate experiments, never compared with each other.
"""

from __future__ import annotations

import json
import math
import os
import time

import numpy as np
import pytest

if os.environ.get("CLVLA_RUN_PI05_TESTS") != "1":
    pytest.skip("pi0.5 path-KL tests are opt-in (CLVLA_RUN_PI05_TESTS=1)", allow_module_level=True)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from pi05_real_env import ROOT, load_real_env  # noqa: E402

K, SIGMA, DELTA = 10, 0.05, 0.1
BATCHES = {"b1": (0,), "b3": (0, 1, 2)}
EPS32 = float(np.finfo(np.float32).eps)
LEAF = "action_out_proj/kernel"        # verified action-head leaf (Task 5/7 gradient routing), state-dependent effect on v
# perturbation = ALPHA * std(W) * N(0, 1), added in fp32, cast back to the leaf dtype. ALPHA fixed from a one-off pilot
# (path KL ~ 12 at 0.1, ~ 105 at 0.3, scaling as ALPHA^2) to give a path KL of order 50.
ALPHA, PERT_SEED = 0.2, 8800
MC_BATCH, MC_MIN_BATCHES, MC_MAX_BATCHES, MC_STEP = "b3", 40, 160, 40
MC_REL_HALFWIDTH = 0.05                # conclusive iff the 95% CI half-width of mean(R - K) <= 5% of mean(K)
RESULTS = {}


def _peak_gib():
    st = jax.local_devices()[0].memory_stats() or {}
    return round((st.get("peak_bytes_in_use") or 0) / 2**30, 2)


def injected(seed, b):
    r = np.random.default_rng(seed)
    return (r.standard_normal((b, 10, 32)).astype(np.float32), r.standard_normal((K, b, 10, 32)).astype(np.float32))


def _flat(state):
    from flax import traverse_util

    return traverse_util.flatten_dict(state.to_pure_dict(), sep="/")


def make_student(teacher, alpha, seed):
    """Separate module sharing every array with the teacher except LEAF, which is rebuilt as cast(W + alpha std(W) z)."""
    from flax import nnx, traverse_util

    graphdef, state = nnx.split(teacher)
    flat = _flat(state)
    w = flat[LEAF]
    w32 = np.asarray(w, np.float32)
    z = np.random.default_rng(seed).standard_normal(w.shape).astype(np.float32)
    intended = np.float32(alpha * float(w32.std())) * z
    new = jnp.asarray(w32 + intended).astype(w.dtype)                           # fresh buffer even when alpha = 0
    flat[LEAF] = new
    state.replace_by_pure_dict(traverse_util.unflatten_dict(flat, sep="/"))
    eff = np.asarray(new, np.float32) - w32                                      # effective perturbation after the cast
    rec = {"leaf": LEAF, "shape": list(w.shape), "dtype": str(w.dtype), "alpha": alpha, "seed": seed,
           "w_std": float(w32.std()), "w_norm": float(np.linalg.norm(w32)),
           "intended_norm": float(np.linalg.norm(intended)), "effective_norm": float(np.linalg.norm(eff)),
           "effective_rel_norm": float(np.linalg.norm(eff) / np.linalg.norm(w32)),
           "effective_max_abs": float(np.abs(eff).max()), "frac_entries_changed": float(np.mean(eff != 0)),
           "effective_over_intended_norm": float(np.linalg.norm(eff) / max(np.linalg.norm(intended), 1e-30)),
           "cast_error_norm": float(np.linalg.norm(eff - intended))}
    return nnx.merge(graphdef, state), rec


@pytest.fixture(scope="module")
def env():
    from flax import nnx

    from clvla.ogpo import pi05_path_kl as pkl
    from clvla.ogpo import pi05_sde as sde
    from clvla.ogpo import pi05_velocity as pv

    e = load_real_env()
    teacher = e["model"]
    t_flat0 = _flat(nnx.split(teacher)[1])          # references only (jax arrays are immutable): no host copy of 3B params
    leaf0 = np.asarray(t_flat0[LEAF], np.float32)
    same, rec0 = make_student(teacher, 0.0, PERT_SEED)
    pert, rec = make_student(teacher, ALPHA, PERT_SEED)
    obs = {n: e["observation"](ids) for n, ids in BATCHES.items()}
    caches = {}
    for n in BATCHES:                       # each policy builds its own prefix cache from the SAME observation object
        caches[n] = {"teacher": pv.build_prefix(teacher, obs[n]), "same": pv.build_prefix(same, obs[n]),
                     "pert": pv.build_prefix(pert, obs[n])}
    RESULTS["perturbation"], RESULTS["identical_student_perturbation"] = rec, rec0
    return {**e, "teacher": teacher, "same": same, "pert": pert, "obs": obs, "caches": caches, "t_flat0": t_flat0,
            "leaf0": leaf0, "pkl": pkl, "sde": sde, "pv": pv}


def run_kl(env, name, student_key, seed):
    c = env["caches"][name]
    x0, eps = injected(seed, len(BATCHES[name]))
    samp = env["sde"].sde_sample(env[student_key], c[student_key], x0, eps)        # the STUDENT generates the chain
    out = env["pkl"].path_kl(env["teacher"], c["teacher"], env[student_key], c[student_key], samp["chain"])
    return x0, eps, {k: np.asarray(v) for k, v in samp.items() if hasattr(v, "shape")}, jax.tree.map(np.asarray, out)


def kl_reference(out):
    """float64 KL from the module's fp32 velocities (= the bf16 model values): sum_k w_k ||v_S - v_T||^2, w_k in float64."""
    from clvla.ogpo.pi05_path_kl import kl_coefficients

    w = kl_coefficients()["weights"]
    dv = out["velocities_student"].astype(np.float64) - out["velocities_teacher"].astype(np.float64)
    return w * np.sum(dv**2, axis=(2, 3)), dv


def mean_rounding(x, v, mu):
    """First-order fp32 rounding model for mu = x - Delta (v + sigma^2/2 (x + (1 - s) v)) (about five fp32 operations, each
    with relative error <= eps32 of its result): |mu_fp32 - mu_exact| <= 4 eps32 (|x| + Delta |v| + |mu|) per coordinate."""
    return 4 * EPS32 * (np.abs(x) + DELTA * np.abs(v) + np.abs(mu))


# ---------------------------------------------------------------- A. identical policy
@pytest.mark.parametrize("name", list(BATCHES))
def test_A_identical_policy(env, name):
    _, _, samp, out = run_kl(env, name, "same", 8000)
    c = env["caches"][name]
    kv_equal = all(np.array_equal(np.asarray(a), np.asarray(b)) for a, b in
                   zip(jax.tree.leaves(c["teacher"].kv_cache), jax.tree.leaves(c["same"].kv_cache)))
    lp_s, lp_t = out["logprob_student"], out["logprob_teacher"]
    RESULTS[f"A_identical_{name}"] = {
        "prefix_kv_cache_bitwise_equal": kv_equal, "max_kl_direct": float(np.abs(out["kl_direct"]).max()),
        "max_kl_velocity": float(np.abs(out["kl_velocity"]).max()), "path_kl_direct": out["path_kl_direct"].tolist(),
        "path_kl_velocity": out["path_kl_velocity"].tolist(), "logprob_student": lp_s["total"].tolist(),
        "logprob_teacher": lp_t["total"].tolist(), "log_ratio": out["log_ratio"].tolist(),
        "initial_diff": out["initial_diff"].tolist()}
    assert RESULTS["identical_student_perturbation"]["effective_norm"] == 0.0
    assert out["kl_direct"].shape == out["kl_velocity"].shape == (len(BATCHES[name]), K)
    assert np.all(out["kl_direct"] == 0) and np.all(out["kl_velocity"] == 0) and np.all(out["dv_sq"] == 0)
    assert np.all(out["path_kl_direct"] == 0) and np.all(out["path_kl_velocity"] == 0)
    assert np.array_equal(lp_s["total"], lp_t["total"]) and np.array_equal(lp_s["transitions"], lp_t["transitions"])
    assert np.all(out["log_ratio"] == 0) and np.all(out["initial_diff"] == 0)
    assert np.array_equal(lp_s["means"], samp["means"])                  # the chain really is this student's
    assert kv_equal


# ---------------------------------------------------------------- B. controlled perturbation
def test_B_controlled_student_perturbation(env):
    from flax import nnx

    rec = RESULTS["perturbation"]
    t_now = _flat(nnx.split(env["teacher"])[1])
    s_now = _flat(nnx.split(env["pert"])[1])
    assert set(s_now) == set(t_now)
    differing = [k for k in s_now if s_now[k] is not t_now[k] and not np.array_equal(np.asarray(s_now[k]), np.asarray(t_now[k]))]
    teacher_unchanged = (all(t_now[k] is v for k, v in env["t_flat0"].items())
                         and np.array_equal(np.asarray(t_now[LEAF], np.float32), env["leaf0"]))
    shared = sum(s_now[k] is t_now[k] for k in s_now)
    rec.update({"leaves_total": len(s_now), "leaves_differing_from_teacher": differing,
                "leaves_sharing_teacher_buffer": int(shared), "teacher_unchanged": teacher_unchanged})
    print(f"\n  [B] perturbation: {json.dumps(rec)}")
    assert differing == [LEAF]                                             # exactly one leaf, never the VLM
    assert teacher_unchanged
    assert rec["effective_norm"] > 0       # survives the bf16 cast; its effect on v at every step is asserted in test C


# ---------------------------------------------------------------- C. direct vs velocity form
@pytest.mark.parametrize("name", list(BATCHES))
def test_C_direct_vs_velocity_form(env, name):
    _, _, samp, out = run_kl(env, name, "pert", 8100)
    ref, dv = kl_reference(out)
    lp_s, lp_t, chain = out["logprob_student"], out["logprob_teacher"], samp["chain"]
    # velocity form vs float64: only the weight cast and the fp32 reduction (terms >= 0, so sum|terms| = KL)
    err_v = np.abs(out["kl_velocity"].astype(np.float64) - ref)
    bound_v = 16 * EPS32 * ref + 2 * EPS32 * ref
    # direct form vs float64: fp32 rounding of both means (model above), of their difference, and of the reduction
    x, vs, vt = chain[:, :K].astype(np.float64), out["velocities_student"], out["velocities_teacher"]
    e = mean_rounding(x, vs, lp_s["means"]) + mean_rounding(x, vt, lp_t["means"])
    d = np.abs(lp_s["means"].astype(np.float64) - lp_t["means"].astype(np.float64))
    e = e + EPS32 * d
    var = (SIGMA**2 * (1 - np.arange(K) / K))[None, :]
    bound_d = np.sum(2 * d * e + e**2, axis=(2, 3)) / (2 * var) + 16 * EPS32 * ref
    err_d = np.abs(out["kl_direct"].astype(np.float64) - ref)
    err_dv = np.abs(out["kl_direct"].astype(np.float64) - out["kl_velocity"].astype(np.float64))
    # teacher means: recomputed at the STUDENT states, consistent with the teacher velocities there
    coef = DELTA * (1 + SIGMA**2 * (1 - np.arange(K) / K) / 2)
    s = (1 - np.arange(K) / K)[None, :, None, None]
    mu_t64 = x - DELTA * (vt + SIGMA**2 / 2 * (x + (1 - s) * vt))
    err_mt = np.abs(lp_t["means"] - mu_t64) / mean_rounding(x, vt, lp_t["means"])
    # mean difference = -coef * dv (x_k cancels)
    err_dmu = np.abs((lp_s["means"] - lp_t["means"]) + coef[None, :, None, None] * dv)
    groups_err = np.abs(out["kl_groups"].astype(np.float64).sum(-1) - out["kl_velocity"])
    shares = out["kl_groups"].sum(axis=(0, 1)) / out["kl_groups"].sum()
    RESULTS[f"C_forms_{name}"] = {
        "path_kl_velocity": out["path_kl_velocity"].tolist(), "path_kl_direct": out["path_kl_direct"].tolist(),
        "path_kl_float64": ref.sum(1).tolist(), "kl_per_step_sample0": ref[0].tolist(), "weights": out["weights"].tolist(),
        "dv_sq_sample0": out["dv_sq"][0].tolist(),
        "velocity_vs_f64": {"max_abs": float(err_v.max()), "max_rel": float((err_v / ref).max()), "max_over_bound": float((err_v / bound_v).max())},
        "direct_vs_f64": {"max_abs": float(err_d.max()), "max_rel": float((err_d / ref).max()), "max_over_bound": float((err_d / bound_d).max())},
        "direct_vs_velocity": {"max_abs": float(err_dv.max()), "max_rel": float((err_dv / ref).max()),
                               "path_max_rel": float((np.abs(out["path_kl_direct"] - out["path_kl_velocity"]) / ref.sum(1)).max())},
        "teacher_means_vs_f64_formula_max_over_rounding_model": float(err_mt.max()),
        "mean_diff_vs_minus_coef_dv_max_abs": float(err_dmu.max()), "groups_sum_max_abs_err": float(groups_err.max()),
        "group_shares_executed_unexecuted_padded": shares.tolist(),
        "frac_velocity_entries_differing": float(np.mean(dv != 0)), "min_step_kl": float(ref.min())}
    print(f"\n  [C] {name}: {json.dumps({k: v for k, v in RESULTS[f'C_forms_{name}'].items() if 'sample0' not in k})}")
    assert np.all(np.isfinite(out["kl_direct"])) and np.all(np.isfinite(out["kl_velocity"]))
    assert np.all(ref > 0)                                                  # perturbation visible at every step and sample
    assert np.all(err_v <= bound_v) and np.all(err_d <= bound_d)
    assert np.all(err_mt <= 1.0)
    assert np.array_equal(lp_s["means"], samp["means"])                     # chain generated by this student
    assert np.all(groups_err <= 16 * EPS32 * out["kl_velocity"])


# ---------------------------------------------------------------- D. Monte Carlo identity + initial-density cancellation
def test_D_monte_carlo_identity(env):
    name = MC_BATCH
    b = len(BATCHES[name])
    rows, t_batches = [], []
    n_batches, target = 0, MC_MIN_BATCHES
    while True:
        while n_batches < target:
            t0 = time.perf_counter()
            x0, eps, samp, out = run_kl(env, name, "pert", 9000 + n_batches)
            t_batches.append(time.perf_counter() - t0)
            dmu = out["means_student"].astype(np.float64) - out["means_teacher"].astype(np.float64)
            sd = out["stds"].astype(np.float64)[None, :, None, None]
            mart = np.sum(np.moveaxis(eps, 0, 1).astype(np.float64) * dmu / sd, axis=(1, 2, 3))  # sum_k eps_k . d_k / sigma_k
            lr_trans = out["log_ratio_transitions"].astype(np.float64).sum(1)
            for i in range(b):
                rows.append({"obs": BATCHES[name][i], "batch": n_batches, "K": float(out["path_kl_velocity"][i]),
                             "K_direct": float(out["path_kl_direct"][i]), "R": float(out["log_ratio"][i]),
                             "M": float(mart[i]), "initial_diff": float(out["initial_diff"][i]),
                             "R_minus_sum_transition_ratios": float(out["log_ratio"][i] - lr_trans[i]),
                             "lp_student": float(out["logprob_student"]["total"][i]),
                             "lp_teacher": float(out["logprob_teacher"]["total"][i]),
                             "abs_parts": float(sum(np.abs(lp["initial"][i]) + np.abs(lp["transitions"][i]).sum()
                                                    for lp in (out["logprob_student"], out["logprob_teacher"])))})
            n_batches += 1
        Kv, R = np.array([r["K"] for r in rows]), np.array([r["R"] for r in rows])
        D = R - Kv
        n, mean, se = len(D), float(D.mean()), float(D.std(ddof=1) / math.sqrt(len(D)))
        half = 1.96 * se
        if half <= MC_REL_HALFWIDTH * Kv.mean() or n_batches >= MC_MAX_BATCHES:
            break
        target = min(n_batches + MC_STEP, MC_MAX_BATCHES)                   # inconclusive: increase the sample count
    resid = np.array([r["R"] - r["K"] - r["M"] for r in rows])
    per_obs = {}
    for o in BATCHES[name]:
        sel = [r for r in rows if r["obs"] == o]
        d_o = np.array([r["R"] - r["K"] for r in sel])
        per_obs[str(o)] = {"n": len(sel), "mean_K": float(np.mean([r["K"] for r in sel])),
                           "mean_R": float(np.mean([r["R"] for r in sel])), "mean_diff": float(d_o.mean()),
                           "se_diff": float(d_o.std(ddof=1) / math.sqrt(len(d_o)))}
    RESULTS["D_monte_carlo"] = {
        "batch_shape": name, "n_trajectories": n, "n_batches": n_batches, "mean_K": float(Kv.mean()),
        "mean_K_direct": float(np.mean([r["K_direct"] for r in rows])), "mean_R": float(R.mean()),
        "mean_diff_R_minus_K": mean, "sd_diff": float(D.std(ddof=1)), "se_diff": se, "ci95": [mean - half, mean + half],
        "ci_halfwidth_over_mean_K": half / float(Kv.mean()), "z": mean / se,
        "predicted_var_diff_2meanK": float(2 * Kv.mean()), "empirical_var_diff": float(D.var(ddof=1)),
        "max_abs_R_minus_K_minus_M": float(np.abs(resid).max()),
        "max_abs_R_minus_K_minus_M_over_sd_diff": float(np.abs(resid).max() / D.std(ddof=1)),
        "initial_diff_max_abs": float(np.abs([r["initial_diff"] for r in rows]).max()),
        "R_vs_sum_transition_ratios_max_abs": float(np.abs([r["R_minus_sum_transition_ratios"] for r in rows]).max()),
        "per_observation": per_obs, "batch_seconds_median": float(np.median(t_batches)), "peak_gib": _peak_gib(),
        "rows": rows}
    print(f"\n  [D] {json.dumps({k: v for k, v in RESULTS['D_monte_carlo'].items() if k != 'rows'})}")
    assert all(r["initial_diff"] == 0.0 for r in rows)                      # shared N(0, I) density cancels exactly
    abs_tot = np.array([r["abs_parts"] for r in rows])                     # summation bound over the 2 x 11 path terms
    assert np.all(np.abs([r["R_minus_sum_transition_ratios"] for r in rows]) <= 16 * EPS32 * abs_tot)
    assert half <= MC_REL_HALFWIDTH * Kv.mean(), "statistically inconclusive at MC_MAX_BATCHES: increase the sample count"
    assert abs(mean) <= half, f"paired 95% CI for E[R - K] excludes 0: {RESULTS['D_monte_carlo']['ci95']}"
    # R - K is exactly the zero-mean term sum_k eps_k . (mu_S - mu_T)_k / sigma_k up to fp32 rounding; require the
    # residual to be < 1% of the per-trajectory MC spread so that it cannot affect the statistical conclusion
    assert np.abs(resid).max() <= 0.01 * D.std(ddof=1)


# ---------------------------------------------------------------- E. runtime / memory
@pytest.mark.parametrize("name", list(BATCHES))
def test_E_runtime(env, name):
    c, pkl = env["caches"][name], env["pkl"]
    _, _, samp, _ = run_kl(env, name, "pert", 8100)
    t = []
    for _ in range(3):
        t0 = time.perf_counter()
        out = pkl.path_kl(env["teacher"], c["teacher"], env["pert"], c["pert"], samp["chain"])
        jax.block_until_ready(out["path_kl_velocity"])
        t.append(time.perf_counter() - t0)
    RESULTS[f"runtime_path_kl_{name}"] = {"median_s": float(np.median(t)), "per_sample_s": float(np.median(t)) / len(BATCHES[name])}
    RESULTS["peak_gib"] = _peak_gib()


def test_zz_write_results():
    out = ROOT / "outputs" / "ogpo_port"
    out.mkdir(parents=True, exist_ok=True)
    f = out / "task8_pi05_path_kl.json"
    f.write_text(json.dumps(RESULTS, indent=1, default=str))
    print(f"\n  results -> {f}")
