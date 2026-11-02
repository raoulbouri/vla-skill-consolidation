"""Task 11 Parts D and G on the real pi0.5: sampled density ratios, analytic KL, score-function routing and one disposable update on a temporary parameter copy. Frozen Task 5-9 modules are imported, never edited.
  CLVLA_USE_GPU=1 CLVLA_GPU=1 PYTHONPATH=tests/ogpo_port scripts/py-policy tests/ogpo_port/task11_real.py
Canonical path: BF16 jit, deterministic flags. Teacher = pi05_libero, student = teacher with Task 8's perturbation of action_out_proj/kernel. This fixture is not a fine-tuned teacher.
Dev seeds 20000+ and 30000+ only: held-out 12000-12007, 13000-13007 and reserved 15000 are refused. Sampled ratios are summed in float64 from float32 means; Task 7's float32 totals are recorded next to them.
Estimators per chain on stored chains, gradient of sum_k c_k log p_k + e sum_k D_k: total score (c = 1, e = 0), rtg (c = future ratios, e = 0), akl (c = future KL, e = 1). sf and sf_loo are the total score times l and l - b.
Split-half cosine measures a gradient mean's noise without any reference. Cosines against the pathwise mean are not validation: Task 10 stays inconclusive.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

os.environ.setdefault("TAG", "task11")
import task10b as T  # sets the deterministic flags before JAX is imported

import jax
import jax.numpy as jnp
import numpy as np
from clvla.ogpo import pi05_path_kl as pkl
from clvla.ogpo import pi05_path_logprob as plp
from clvla.ogpo import pi05_sde as sde
from flax import nnx
from test_pi05_pathwise_gradient import group_of

pv = T.pv
ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "ogpo_port" / "task11"
TMP = ROOT / "tmp" / "task11_update"
OUT.mkdir(parents=True, exist_ok=True)
NCHAINS, NGRAD, NEVAL, OBS = 256, 32, 16, (0, 1, 2)
STEP_NORMS = (1e-3, 3e-3)
FORBIDDEN = set(range(12000, 12008)) | set(range(13000, 13008)) | {15000}
cfg = sde.SDEConfig()
stds = np.asarray(cfg.stds(), np.float64)
RES = {"fixture": "teacher pi05_libero, student Task 8 perturbation; not a fine-tuned teacher", "chains_per_obs": NCHAINS}


def save():
    (OUT / "real.json").write_text(json.dumps(RES, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def seed_of(base, i):
    s = base + i
    assert s not in FORBIDDEN
    return s


def stats(a):
    a = np.asarray(a, np.float64)
    return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / len(a) ** 0.5), "std": float(a.std(ddof=1)), "min": float(a.min()), "max": float(a.max()), "n": len(a)}


def host_ratio(chain, ms, mt):
    """float64 per-step ratio and KL from float32 means, shape (K,)."""
    y = np.asarray(chain[0, 1:], np.float64)
    a, b = np.asarray(ms[0], np.float64), np.asarray(mt[0], np.float64)
    ell = (np.sum((y - b) ** 2, axis=(1, 2)) - np.sum((y - a) ** 2, axis=(1, 2))) / (2 * stds**2)
    D = np.sum((a - b) ** 2, axis=(1, 2)) / (2 * stds**2)
    return ell, D


def main():
    H = T.helpers()
    flat = H["_flat"]
    RES["env"] = T.env_record()
    e = H["load_real_env"]()
    teacher = e["model"]
    student, rec = H["make_student"](teacher, H["ALPHA"], H["PERT_SEED"])
    RES["perturbation"] = rec
    in_g = H["in_groups"]
    gd_s, theta, other_s = nnx.split(student, in_g(H["TRAINABLE"]), ...)
    gd_t, t_all = nnx.split(teacher)
    _, theta_t, other_t = nnx.split(teacher, in_g(H["TRAINABLE"]), ...)
    RES["input_sha256"] = {"theta": H["sha"](theta), "teacher": H["sha"](t_all)}
    RES["n_trainable"] = int(sum(v.size for v in flat(theta).values()))
    data = {}
    for o in OBS:
        ob = e["observation"]((o,))
        sc, tc = pv.build_prefix(student, ob), pv.build_prefix(teacher, ob)
        data[o] = {"obs": sc.observation, "s_cache": (sc.prefix_mask, sc.kv_cache), "t_cache": (tc.prefix_mask, tc.kv_cache)}

    # action path: final latent -> physical chunk through the policy's own output transform
    try:
        x = np.asarray(H["injected"](21000, 1)[0][0])
        out = e["policy"]._output_transform({"actions": x, "state": np.zeros(8, np.float32)})
        a = np.asarray(out["actions"])
        RES["action_transform"] = {"output_shape": list(a.shape), "finite": bool(np.isfinite(a).all()), "in_shape": list(x.shape), "note": "10 x 32 latent to 10 x 7 physical chunk; run_episode executes the first 5"}
    except Exception as err:  # noqa: BLE001 - reported, not hidden
        RES["action_transform"] = {"error": f"{type(err).__name__}: {err}"}
    print("[action]", json.dumps(RES["action_transform"]), flush=True)
    save()
    policy_ref = e.pop("policy")
    del policy_ref, teacher, student
    e.clear()
    gc.collect()

    def merge_s(th, I):
        return nnx.merge(gd_s, th, I["rest_s"]), pv.PrefixCache(I["obs"], *I["s_cache"])

    def merge_t(I):
        return nnx.merge(gd_t, I["teacher"]), pv.PrefixCache(I["obs"], *I["t_cache"])

    def sample_(th, I, x0, eps):
        m, c = merge_s(th, I)
        o = sde.sde_sample(m, c, x0, eps, cfg)
        return o["chain"], o["means"]

    def score_(th, I, chain):
        m, c = merge_s(th, I)
        mt, ct = merge_t(I)
        a, b = plp.path_log_prob(m, c, chain, cfg), plp.path_log_prob(mt, ct, chain, cfg)
        return {"s_total": a["total"], "t_total": b["total"], "s_trans": a["transitions"], "t_trans": b["transitions"], "ms": a["means"], "mt": b["means"]}

    def weighted_(th, I, chain, c_k, e_):
        m, c = merge_s(th, I)
        a = plp.path_log_prob(m, c, chain, cfg)
        mt, ct = merge_t(I)
        tm = jax.lax.stop_gradient(plp.path_log_prob(mt, ct, chain, cfg)["means"])
        D = jnp.sum((a["means"] - tm) ** 2, axis=(2, 3)) / (2.0 * jnp.asarray(cfg.stds(), jnp.float32) ** 2)
        return jnp.sum(jnp.asarray(c_k, jnp.float32) * a["transitions"]) + e_ * jnp.sum(D)

    sample = jax.jit(sample_)
    score = jax.jit(score_)
    grad_w = jax.jit(jax.grad(weighted_))
    f = T.canonical(gd_s, gd_t)
    grad_path = jax.jit(jax.value_and_grad(f, has_aux=True))

    def I_of(o, th_rest=other_s):
        d = data[o]
        return {"rest_s": th_rest, "teacher": t_all, "obs": d["obs"], "s_cache": d["s_cache"], "t_cache": d["t_cache"]}

    # D1-D8: ratios against analytic KL on three observations
    store = {}
    RES["ratio"] = {}
    for o in OBS:
        I = I_of(o)
        t0 = time.perf_counter()
        rows = []
        for i in range(NCHAINS):
            x0, eps = H["injected"](seed_of(20000 + 1000 * o, i), 1)
            chain, means = sample(theta, I, jnp.asarray(x0), jnp.asarray(eps))
            s = score(theta, I, chain)
            ell, D = host_ratio(chain, s["ms"], s["mt"])
            means_np, ms_np = np.asarray(means), np.asarray(s["ms"])
            # the same stored chain scored under the sampler's means (old) and the recomputed means (new), identical parameters
            y = np.asarray(chain[0, 1:], np.float64)
            old = -np.sum((y - np.asarray(means_np[0], np.float64)) ** 2, axis=(1, 2)) / (2 * stds**2)
            new = -np.sum((y - np.asarray(ms_np[0], np.float64)) ** 2, axis=(1, 2)) / (2 * stds**2)
            rows.append({"ell": ell, "D": D, "fp32_ratio": float(s["s_total"][0] - s["t_total"][0]), "s_total": float(s["s_total"][0]), "t_total": float(s["t_total"][0]), "means_bitwise": bool(np.array_equal(means_np, ms_np)), "old_new": float(np.sum(new) - np.sum(old)), "chain": np.asarray(chain)})
        store[o] = rows
        ells = np.array([r["ell"].sum() for r in rows])
        Ds = np.array([r["D"].sum() for r in rows])
        RES["ratio"][str(o)] = {"seconds": time.perf_counter() - t0, "ell": stats(ells), "K": stats(Ds), "paired_ell_minus_K": stats(ells - Ds), "paired_z": float((ells - Ds).mean() / ((ells - Ds).std(ddof=1) / NCHAINS**0.5)), "reward_mean": float(-ells.mean()),
                                "fp32_ratio_minus_fp64": stats(np.array([r["fp32_ratio"] for r in rows]) - ells), "s_logp": stats([r["s_total"] for r in rows]), "t_logp": stats([r["t_total"] for r in rows]), "sampler_vs_recomputed_means_bitwise": all(r["means_bitwise"] for r in rows),
                                "same_params_log_ratio_old_new": stats([r["old_new"] for r in rows]), "per_step_ell": np.mean([r["ell"] for r in rows], 0).tolist(), "per_step_D": np.mean([r["D"] for r in rows], 0).tolist()}
        print(f"[ratio obs {o}] {json.dumps({k: RES['ratio'][str(o)][k] for k in ('seconds', 'ell', 'K', 'paired_z', 'sampler_vs_recomputed_means_bitwise', 'same_params_log_ratio_old_new')})}", flush=True)
        save()

    # D9: identical parameters. Teacher parameters in the student slot on the student's own programme.
    I0_id = {**I_of(0, other_t), "s_cache": data[0]["t_cache"]}
    rows = []
    for i in range(4):
        x0, eps = H["injected"](seed_of(24000, i), 1)
        chain, _ = sample(theta_t, I0_id, jnp.asarray(x0), jnp.asarray(eps))
        s = score(theta_t, I0_id, chain)
        ell, D = host_ratio(chain, s["ms"], s["mt"])
        rows.append({"ell": float(ell.sum()), "K": float(D.sum()), "fp32": float(s["s_total"][0] - s["t_total"][0]), "means_equal": bool(np.array_equal(np.asarray(s["ms"]), np.asarray(s["mt"])))})
    RES["identical"] = rows
    print("[identical]", json.dumps(rows), flush=True)
    save()

    # D routing: one chain, all student and teacher parameters differentiated, with the ratio detached inside the loss
    def routing(s_all, t_all_, I, chain):
        m = nnx.merge(gd_s, s_all)
        mt = nnx.merge(gd_t, t_all_)
        a = plp.path_log_prob(m, pv.PrefixCache(I["obs"], *I["s_cache"]), chain, cfg)
        b = plp.path_log_prob(mt, pv.PrefixCache(I["obs"], *I["t_cache"]), chain, cfg)
        return jnp.sum(jax.lax.stop_gradient(a["total"] - b["total"]) * a["total"])

    # both parameter sets in one derivative needs a 41.8 GB allocation, so each is differentiated in its own programme
    try:
        s_all = nnx.split(nnx.merge(gd_s, theta, other_s))[1]
        c0 = jnp.asarray(store[0][0]["chain"])
        g_s = jax.jit(jax.grad(routing, argnums=0))(s_all, t_all, I_of(0), c0)
        jax.clear_caches()
        g_t = jax.jit(jax.grad(routing, argnums=1))(s_all, t_all, I_of(0), c0)
        grp = {}
        for name, tree in (("student", g_s), ("teacher", g_t)):
            for k, v in flat(tree).items():
                if not jnp.issubdtype(v.dtype, jnp.floating):
                    continue
                gname = group_of(k)
                r = grp.setdefault(f"{name}/{gname}", {"norm2": 0.0, "nonzero": 0, "size": 0, "finite": True})
                r["norm2"] += float(jnp.sum(jnp.square(v.astype(jnp.float32))))
                r["nonzero"] += int(jnp.count_nonzero(v))
                r["size"] += int(v.size)
                r["finite"] = r["finite"] and bool(jnp.isfinite(v.astype(jnp.float32)).all())
        for r in grp.values():
            r["norm"] = r.pop("norm2") ** 0.5
        RES["routing"] = grp
        del g_s, g_t
    except Exception as err:  # noqa: BLE001 - reported, not hidden
        RES["routing"] = {"error": f"{type(err).__name__}: {str(err)[:300]}"}
    print("[routing]", json.dumps(RES["routing"]), flush=True)
    save()
    jax.clear_caches()
    gc.collect()
    sample = jax.jit(sample_)
    score = jax.jit(score_)
    grad_w = jax.jit(jax.grad(weighted_))
    grad_path = jax.jit(jax.value_and_grad(f, has_aux=True))

    # Gradient noise on observation 0: split-half cosines, so no reference is assumed
    I = I_of(0)
    rows = store[0][:NGRAD]
    ells = np.array([r["ell"].sum() for r in rows])
    keys = sorted(flat(theta))
    names = ("sf", "sf_loo", "rtg", "akl", "pathwise")
    half = {n: [{k: 0.0 for k in keys}, {k: 0.0 for k in keys}] for n in names}
    norms = {n: [] for n in names}
    finite = {n: True for n in names}
    t0 = time.perf_counter()
    host = lambda g: {k: np.asarray(v).astype(np.float32) for k, v in flat(g).items()}  # noqa: E731

    def add(n, i, g):
        h = half[n][i % 2]
        for k in keys:
            h[k] = h[k] + g[k]
        norms[n].append(sum(float(np.sum(g[k].astype(np.float64) ** 2)) for k in keys) ** 0.5)
        finite[n] = finite[n] and all(bool(np.isfinite(g[k]).all()) for k in keys)

    for i, r in enumerate(rows):
        chain = jnp.asarray(r["chain"])
        score_g = host(grad_w(theta, I, chain, np.ones(10, np.float32), 0.0))
        loo = (ells.sum() - ells[i]) / (NGRAD - 1)
        add("sf", i, {k: float(ells[i]) * score_g[k] for k in keys})
        add("sf_loo", i, {k: float(ells[i] - loo) * score_g[k] for k in keys})
        del score_g
        togo = np.cumsum(r["ell"][::-1])[::-1].astype(np.float32)
        add("rtg", i, host(grad_w(theta, I, chain, togo, 0.0)))
        fut = np.concatenate([np.cumsum(r["D"][::-1])[::-1][1:], [0.0]]).astype(np.float32)
        add("akl", i, host(grad_w(theta, I, chain, fut, 1.0)))
        x0, eps = H["injected"](seed_of(20000, i), 1)
        (_, _), g = grad_path(theta, {**I, "rest_s": (other_s,), "teacher": (t_all,), "x0": jnp.asarray(x0), "eps": jnp.asarray(eps)})
        add("pathwise", i, host(g))
        del g
        if i % 8 == 7:
            print(f"[gradients] {i + 1}/{NGRAD} {time.perf_counter() - t0:.0f} s", flush=True)

    def dot(a, b):
        return sum(float(np.sum(a[k].astype(np.float64) * b[k].astype(np.float64))) for k in keys)

    mean = {n: {k: (half[n][0][k] + half[n][1][k]) / NGRAD for k in keys} for n in names}
    cos = lambda a, b: dot(a, b) / (dot(a, a) * dot(b, b)) ** 0.5  # noqa: E731
    RES["gradients"] = {"seconds": time.perf_counter() - t0, "n": NGRAD, "estimators": {}}
    for n in names:
        gr = {}
        for k in keys:
            gname = group_of(k)
            gr[gname] = gr.get(gname, 0.0) + float(np.sum(mean[n][k].astype(np.float64) ** 2))
        RES["gradients"]["estimators"][n] = {"finite": finite[n], "per_sample_norm": stats(norms[n]), "mean_norm": dot(mean[n], mean[n]) ** 0.5, "group_norm": {g: v**0.5 for g, v in gr.items()}, "split_half_cos": cos(half[n][0], half[n][1]), "cos_with_pathwise_mean": cos(mean[n], mean["pathwise"])}
        print(f"[gradient {n}] {json.dumps(RES['gradients']['estimators'][n])}", flush=True)
    save()
    del half
    gc.collect()

    # Part G: one step on a temporary copy, evaluated with paired noise on chains the gradient never saw
    th_np = {k: np.asarray(v) for k, v in flat(theta).items()}
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat(theta).items()}
    order = [by_id[id(l)] for l in leaves]

    def evaluate(th_):
        out = []
        for i in range(NEVAL):
            x0, eps = H["injected"](seed_of(30000, i), 1)
            chain, _ = sample(th_, I, jnp.asarray(x0), jnp.asarray(eps))
            s = score(th_, I, chain)
            _, D = host_ratio(chain, s["ms"], s["mt"])
            out.append(float(D.sum()))
        return np.array(out)

    def log_prob_new(th_, r):
        return float(score(th_, I, jnp.asarray(r["chain"]))["s_total"][0])

    base = evaluate(theta)
    RES["update"] = {"before_K": stats(base), "runs": {}, "master_precision": "fp32 temporary copy, cast back to the model's bf16 for evaluation"}
    for est in ("sf_loo", "pathwise"):
        g = mean[est]
        gn = dot(g, g) ** 0.5
        for h in STEP_NORMS:
            master = {k: th_np[k].astype(np.float32) - np.float32(h / gn) * g[k] for k in keys}
            new_np = {k: master[k].astype(th_np[k].dtype) if th_np[k].dtype != np.dtype("V2") else master[k] for k in keys}
            th_new = jax.tree.unflatten(treedef, [jnp.asarray(master[k]).astype(l.dtype) for k, l in zip(order, leaves)])
            nh = host(th_new)
            eff = sum(float(np.sum((nh[k].astype(np.float64) - th_np[k].astype(np.float64)) ** 2)) for k in keys) ** 0.5
            changed = {}
            for k in keys:
                changed.setdefault(group_of(k), [0, 0])
                changed[group_of(k)][0] += int(np.count_nonzero(nh[k] != th_np[k].astype(np.float32)))
                changed[group_of(k)][1] += int(nh[k].size)
            after = evaluate(th_new)
            lp_old = np.array([r["s_total"] for r in rows])
            lp_new = np.array([log_prob_new(th_new, r) for r in rows])
            ratio = lp_new - lp_old
            run = {"gradient_norm": gn, "nominal_step_norm": h, "effective_step_norm": eff, "param_norm_before": sum(float(np.sum(th_np[k].astype(np.float64) ** 2)) for k in keys) ** 0.5, "param_norm_after": sum(float(np.sum(nh[k].astype(np.float64) ** 2)) for k in keys) ** 0.5,
                   "entries_changed_by_group": changed, "after_K": stats(after), "paired_delta_K": stats(after - base), "first_order_prediction": -h * dot(g, mean["pathwise"]) / gn, "log_ppo_ratio": stats(ratio), "ppo_ratio_median": float(np.median(np.exp(ratio))), "ppo_ratio_max": float(np.exp(ratio).max())}
            RES["update"]["runs"][f"{est}_h{h}"] = run
            print(f"[update {est} h={h}] {json.dumps(run)}", flush=True)
            if est == "sf_loo" and h == STEP_NORMS[0]:
                # isolated checkpoint round trip; canonical checkpoints are never opened for writing
                TMP.mkdir(parents=True, exist_ok=True)
                p = TMP / "theta_after_one_step.npz"
                np.savez(p, **{k.replace("/", "__"): nh[k] for k in keys})
                z = np.load(p)
                ok = all(np.array_equal(z[k.replace("/", "__")], nh[k]) for k in keys)
                RES["update"]["temporary_checkpoint"] = {"path": str(p), "bytes": p.stat().st_size, "round_trip_bitwise": ok, "sha256_prefix": hashlib.sha256(np.concatenate([nh[k][:8].ravel() for k in keys]).tobytes()).hexdigest()[:16]}
                z.close()
                shutil.rmtree(TMP)
                RES["update"]["temporary_checkpoint"]["deleted_after_check"] = True
            del th_new, nh, master, new_np
            save()
    RES["unchanged_theta_after_all_updates"] = H["sha"](theta) == RES["input_sha256"]["theta"]
    RES["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}
    save()
    print("[end]", json.dumps(RES["memory"]), RES["unchanged_theta_after_all_updates"], flush=True)


if __name__ == "__main__":
    main()
