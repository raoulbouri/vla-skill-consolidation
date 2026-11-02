"""Task 11 Part C: OPD gradient estimators on a linear Markov flow with a closed-form path KL. CPU, float64. No frozen module is imported.
  JAX_PLATFORMS=cpu scripts/py-policy tests/ogpo_port/task11_synthetic.py
Same transition as the frozen sampler: s_k = 1 - k/K, Delta = 1/K, sigma_k = sigma sqrt(s_k), mu = x - Delta (v + sigma^2/2 (x + (1 - s) v)), here with v = A_k x + b_k.
The state stays Gaussian, so E[K] follows from moment propagation; the exact gradient is the derivative of that formula, checked by central differences.

Estimators of d/dtheta E_{P_theta}[l(X)], l = log P_theta - log P_T:
  pathwise   grad of sum_k D_k(theta, x_k(theta))                                 state attached, teacher parameters fixed
  sf         l grad log P                                                         E[grad l] = E[grad log P] = 0, so no extra term
  sf_loo     (l - b) grad log P, b = mean of l over the rest of the group         b independent of the sample
  rtg        sum_k grad log p_k sum_{j >= k} l_j                                  earlier ratios are independent in expectation of a later score
  akl        sum_k d_theta D_k + sum_j grad log p_j sum_{k > j} D_k               analytic KL needs its explicit derivative term
Negative controls, which must be detected as biased:
  local      sum_k grad log p_k l_k                                               own-step reward only, drops the future
  akl_noexp  sum_j grad log p_j sum_{k > j} D_k                                   analytic KL as a detached reward, explicit term missing
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

K, DELTA, SIGMA = 5, 0.2, 0.05
SEEDS = tuple(range(9000, 9008))
SAMPLES = (1_000, 10_000, 100_000, 1_000_000)
GROUP = 8
CHUNK = 20_000
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ogpo_port" / "task11"
OUT.mkdir(parents=True, exist_ok=True)
times = 1.0 - jnp.arange(K) / K
stds = SIGMA * jnp.sqrt(times)
co = 1.0 + 0.5 * SIGMA**2 * (1.0 - times)
weights = (DELTA * co) ** 2 / (2.0 * stds**2)


def mean_of(th, k, x):
    v = th["A"][k] @ x + th["b"][k]
    return x - DELTA * (v + 0.5 * SIGMA**2 * (x + (1.0 - times[k]) * v))


def closed_form_kl(th, te):
    d = th["A"].shape[1]
    m, C = jnp.zeros(d), jnp.eye(d)
    total = 0.0
    per = []
    for k in range(K):
        dA, db = th["A"][k] - te["A"][k], th["b"][k] - te["b"][k]
        r = dA @ m + db
        D = weights[k] * (r @ r + jnp.trace(dA @ C @ dA.T))
        per.append(D)
        total = total + D
        M = jnp.eye(d) - DELTA * (co[k] * th["A"][k] + 0.5 * SIGMA**2 * jnp.eye(d))
        m = M @ m - DELTA * co[k] * th["b"][k]
        C = M @ C @ M.T + stds[k] ** 2 * jnp.eye(d)
    return total, jnp.stack(per)


def rollout(th, x0, eps):
    xs = [x0]
    for k in range(K):
        xs.append(mean_of(th, k, xs[-1]) + stds[k] * eps[k])
    return jnp.stack(xs)


def step_logp(th, chain):
    def one(k):
        y, mu = chain[k + 1], mean_of(th, k, chain[k])
        return -0.5 * (jnp.sum((y - mu) ** 2) / stds[k] ** 2 + y.size * jnp.log(2 * jnp.pi * stds[k] ** 2))

    return jnp.stack([one(k) for k in range(K)])


def step_kl(th, te, chain):
    return jnp.stack([jnp.sum((mean_of(th, k, chain[k]) - mean_of(te, k, chain[k])) ** 2) / (2 * stds[k] ** 2) for k in range(K)])


def flat(t):
    return jnp.concatenate([t["A"].ravel(), t["b"].ravel()])


def per_sample(th, te, x0, eps):
    chain = rollout(th, x0, eps)
    ls, lt = step_logp(th, chain), step_logp(te, chain)
    ell = ls - lt
    Ds = step_kl(th, te, chain)
    dlp = jax.jacrev(lambda t: step_logp(t, chain))(th)
    scores = jnp.concatenate([dlp["A"].reshape(K, -1), dlp["b"].reshape(K, -1)], axis=1)   # (K, P), one row per transition
    explicit = flat(jax.grad(lambda t: jnp.sum(step_kl(t, te, chain)))(th))
    path = flat(jax.grad(lambda t: jnp.sum(step_kl(t, te, rollout(t, x0, eps))))(th))
    total_score = jnp.sum(scores, axis=0)
    togo = jnp.cumsum(ell[::-1])[::-1]
    future_D = jnp.concatenate([jnp.cumsum(Ds[::-1])[::-1][1:], jnp.zeros(1)])
    return {"pathwise": path, "sf": jnp.sum(ell) * total_score, "rtg": togo @ scores, "akl": explicit + future_D @ scores, "local": ell @ scores, "akl_noexp": future_D @ scores,
            "score": total_score, "ell": jnp.sum(ell), "D": jnp.sum(Ds), "ell_k": ell, "D_k": Ds}


per_batch = jax.jit(jax.vmap(per_sample, in_axes=(None, None, 0, 0)))


def estimate(th, te, seed, n):
    """Accumulates means and variances chunk by chunk. sf_loo uses disjoint groups of GROUP samples within a chunk."""
    key = jax.random.PRNGKey(seed)
    d = th["A"].shape[1]
    names = ("pathwise", "sf", "sf_loo", "rtg", "akl", "local", "akl_noexp")
    sums, sq = {k: 0.0 for k in names}, {k: 0.0 for k in names}
    ell_sum, ell_sq, D_sum, D_sq, ek, dk = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    done = 0
    while done < n:
        m = min(CHUNK, n - done)
        m -= m % GROUP
        key, a, b = jax.random.split(key, 3)
        o = per_batch(th, te, jax.random.normal(a, (m, d)), jax.random.normal(b, (K, m, d)).transpose(1, 0, 2))
        ell = o["ell"]
        g = ell.reshape(-1, GROUP)
        loo = ((g.sum(1, keepdims=True) - g) / (GROUP - 1)).reshape(-1)
        o["sf_loo"] = (ell - loo)[:, None] * o["score"]
        for k in names:
            sums[k] = sums[k] + np.asarray(jnp.sum(o[k], 0))
            sq[k] = sq[k] + np.asarray(jnp.sum(o[k] ** 2, 0))
        ell_sum += float(jnp.sum(ell))
        ell_sq += float(jnp.sum(ell**2))
        D_sum += float(jnp.sum(o["D"]))
        D_sq += float(jnp.sum(o["D"] ** 2))
        ek = ek + np.asarray(jnp.sum(o["ell_k"], 0))
        dk = dk + np.asarray(jnp.sum(o["D_k"], 0))
        done += m
    means = {k: sums[k] / done for k in names}
    ses = {k: np.sqrt(np.maximum(sq[k] / done - means[k] ** 2, 0.0) / done) for k in names}
    return {"n": done, "means": means, "ses": ses, "ell": ell_sum / done, "ell_se": float(np.sqrt(max(ell_sq / done - (ell_sum / done) ** 2, 0.0) / done)), "D": D_sum / done, "D_se": float(np.sqrt(max(D_sq / done - (D_sum / done) ** 2, 0.0) / done)),
            "ell_k": ek / done, "D_k": dk / done}


def models(d, scale, seed=1234):
    r = np.random.default_rng(seed)
    te = {"A": jnp.asarray(r.normal(size=(K, d, d)) * 0.3 - 0.5 * np.eye(d)), "b": jnp.asarray(r.normal(size=(K, d)) * 0.3)}
    th = {"A": te["A"] + scale * jnp.asarray(r.normal(size=(K, d, d))) * 0.1, "b": te["b"] + scale * jnp.asarray(r.normal(size=(K, d))) * 0.1}
    return th, te


def exact_grad(th, te):
    return flat(jax.grad(lambda t: closed_form_kl(t, te)[0])(th))


def fd_grad(th, te, h=1e-5):
    x0 = np.asarray(flat(th))
    shapeA = th["A"].shape
    nA = int(np.prod(shapeA))

    def unflat(v):
        return {"A": jnp.asarray(v[:nA]).reshape(shapeA), "b": jnp.asarray(v[nA:]).reshape(th["b"].shape)}

    out = np.zeros_like(x0)
    for i in range(x0.size):
        e = np.zeros_like(x0)
        e[i] = h
        out[i] = (float(closed_form_kl(unflat(x0 + e), te)[0]) - float(closed_form_kl(unflat(x0 - e), te)[0])) / (2 * h)
    return out


def cos_rel(g, e):
    return float(g @ e / (np.linalg.norm(g) * np.linalg.norm(e))), float(np.linalg.norm(g - e) / np.linalg.norm(e))


def main():
    res, checks = {"config": {"K": K, "sigma": SIGMA, "d": 3, "seeds": SEEDS, "samples": SAMPLES, "group": GROUP}}, {}
    # identical student and teacher: every ratio, KL and gradient estimator with a reward is exactly zero
    th0, te0 = models(3, 0.0)
    o0 = per_batch(th0, te0, jax.random.normal(jax.random.PRNGKey(1), (64, 3)), jax.random.normal(jax.random.PRNGKey(2), (K, 64, 3)).transpose(1, 0, 2))
    res["identical"] = {"max_abs_ell": float(jnp.abs(o0["ell"]).max()), "max_abs_D": float(jnp.abs(o0["D"]).max()), "max_abs_sf": float(jnp.abs(o0["sf"]).max()), "max_abs_pathwise": float(jnp.abs(o0["pathwise"]).max()), "closed_form": float(closed_form_kl(th0, te0)[0])}
    checks["identical_exact_zero"] = all(v == 0.0 for v in res["identical"].values())

    for label, scale in (("small", 0.3), ("large", 1.0)):
        th, te = models(3, scale)
        kl, per = closed_form_kl(th, te)
        kl = float(kl)
        ex = np.asarray(exact_grad(th, te))
        fd = fd_grad(th, te)
        r = {"closed_form_kl": kl, "closed_form_per_step": np.asarray(per).tolist(), "exact_norm": float(np.linalg.norm(ex)), "fd_vs_exact_rel": float(np.linalg.norm(fd - ex) / np.linalg.norm(ex)), "table": {}}
        checks[f"{label}_fd"] = r["fd_vs_exact_rel"] < 1e-7
        ests = {}
        for n in SAMPLES:
            rows = [estimate(th, te, s, n) for s in SEEDS]
            ests[n] = rows
            t = {}
            for name in rows[0]["means"]:
                cr = np.array([cos_rel(x["means"][name], ex) for x in rows])
                pooled = np.mean([x["means"][name] for x in rows], axis=0)
                pooled_se = np.sqrt(np.mean([x["ses"][name] ** 2 for x in rows], axis=0) / len(rows))
                t[name] = {"cos_mean": float(cr[:, 0].mean()), "cos_min": float(cr[:, 0].min()), "rel_mean": float(cr[:, 1].mean()), "rel_max": float(cr[:, 1].max()), "pooled_cos": cos_rel(pooled, ex)[0], "pooled_rel": cos_rel(pooled, ex)[1],
                           "mean_se_over_norm": float(np.mean([np.linalg.norm(x["ses"][name]) for x in rows]) / np.linalg.norm(ex)), "max_abs_z": float(np.max(np.abs(pooled - ex) / pooled_se)), "chi2_per_dim": float(np.mean(((pooled - ex) / pooled_se) ** 2))}
            ell_m = float(np.mean([x["ell"] for x in rows]))
            ell_se = float(np.sqrt(np.mean([x["ell_se"] ** 2 for x in rows]) / len(rows)))
            D_m = float(np.mean([x["D"] for x in rows]))
            D_se = float(np.sqrt(np.mean([x["D_se"] ** 2 for x in rows]) / len(rows)))
            r["table"][str(n)] = {"est": t, "ell_mean": ell_m, "ell_se": ell_se, "ell_z": (ell_m - kl) / ell_se, "reward_mean": -ell_m, "D_mean": D_m, "D_se": D_se, "D_z": (D_m - kl) / D_se}
            print(f"[{label} n={n}] kl {kl:.5f} ell {ell_m:.5f}+-{ell_se:.5f} D {D_m:.5f}+-{D_se:.5f} | " + " ".join(f"{k}:rel {t[k]['rel_mean']:.3f} cos {t[k]['cos_mean']:.4f} z {t[k]['max_abs_z']:.1f}" for k in t), flush=True)
        top = r["table"][str(SAMPLES[-1])]
        # unbiased estimators: pooled over 8 seeds at the largest n, no coordinate beyond a Bonferroni z limit and chi^2 per dimension near one
        for name in ("pathwise", "sf", "sf_loo", "rtg", "akl"):
            checks[f"{label}_{name}_unbiased"] = top["est"][name]["max_abs_z"] < 4.0 and top["est"][name]["chi2_per_dim"] < 1.6
        for name in ("local", "akl_noexp"):
            checks[f"{label}_{name}_detected_biased"] = top["est"][name]["max_abs_z"] > 6.0
        checks[f"{label}_ell_to_kl"] = abs(top["ell_z"]) < 4.0 and abs(top["D_z"]) < 4.0
        checks[f"{label}_reward_negative"] = top["reward_mean"] < 0
        # Monte Carlo rate: se norm falls as n^-1/2
        slope = np.polyfit(np.log10(SAMPLES), np.log10([r["table"][str(n)]["est"]["sf_loo"]["mean_se_over_norm"] for n in SAMPLES]), 1)[0]
        r["sf_loo_se_slope"] = float(slope)
        checks[f"{label}_se_rate"] = abs(slope + 0.5) < 0.1
        # the baseline changes the variance, not the expectation: pooled sf and sf_loo differ only within their standard errors
        rows = ests[SAMPLES[-1]]
        a = np.mean([x["means"]["sf"] for x in rows], axis=0)
        b = np.mean([x["means"]["sf_loo"] for x in rows], axis=0)
        sa = np.sqrt(np.mean([x["ses"]["sf"] ** 2 for x in rows], axis=0) / len(rows))
        sb = np.sqrt(np.mean([x["ses"]["sf_loo"] ** 2 for x in rows], axis=0) / len(rows))
        r["baseline_diff_max_z"] = float(np.max(np.abs(a - b) / np.sqrt(sa**2 + sb**2)))
        r["variance_ratio_sf_over_sf_loo"] = float(np.sum(np.mean([x["ses"]["sf"] ** 2 for x in rows], axis=0)) / np.sum(np.mean([x["ses"]["sf_loo"] ** 2 for x in rows], axis=0)))
        checks[f"{label}_baseline_same_mean"] = r["baseline_diff_max_z"] < 4.0
        res[label] = r
        print(f"[{label}] fd rel {r['fd_vs_exact_rel']:.2e} slope {slope:.3f} baseline z {r['baseline_diff_max_z']:.2f} var ratio {r['variance_ratio_sf_over_sf_loo']:.2f}", flush=True)
        (OUT / "synthetic.json").write_text(json.dumps({**res, "checks": checks}, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else float(o)))

    # Variance against dimension, same scale per coordinate: samples needed for 10% relative error, read from the per-sample standard deviation
    res["dimension"] = {}
    for d in (3, 32, 320):
        r_ = np.random.default_rng(55)
        te = {"A": jnp.asarray(np.stack([np.diag(-0.5 + 0.1 * r_.normal(size=d)) for _ in range(K)])), "b": jnp.asarray(r_.normal(size=(K, d)) * 0.1)}
        th = {"A": te["A"] + jnp.asarray(np.stack([np.diag(0.03 * r_.normal(size=d)) for _ in range(K)])), "b": te["b"] + jnp.asarray(r_.normal(size=(K, d)) * 0.03)}
        kl = float(closed_form_kl(th, te)[0])
        ex = np.asarray(exact_grad(th, te))
        n = 400 if d == 320 else 4000
        global CHUNK
        keep = CHUNK
        CHUNK = 40 if d == 320 else 2000
        e = estimate(th, te, 777, n)
        CHUNK = keep
        row = {"kl": kl, "exact_norm": float(np.linalg.norm(ex)), "n": e["n"]}
        for name in ("pathwise", "sf_loo", "rtg", "akl"):
            per_sample_var = float(np.sum(e["ses"][name] ** 2) * e["n"])
            row[name] = {"per_sample_rel_std": per_sample_var**0.5 / float(np.linalg.norm(ex)), "samples_for_10pct_rel": per_sample_var / (0.1 * float(np.linalg.norm(ex))) ** 2}
        res["dimension"][str(d)] = row
        print(f"[dimension {d}] kl {kl:.3f} " + " ".join(f"{k}: n10% {row[k]['samples_for_10pct_rel']:.3g}" for k in ("pathwise", "sf_loo", "rtg", "akl")), flush=True)
    res["checks"] = checks
    res["all_pass"] = bool(all(checks.values()))
    (OUT / "synthetic.json").write_text(json.dumps(res, indent=1, default=lambda o: o.item() if hasattr(o, 'item') else float(o)))
    print({k: bool(v) for k, v in checks.items()})
    print("ALL_PASS", res["all_pass"], flush=True)
    return 0 if res["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
