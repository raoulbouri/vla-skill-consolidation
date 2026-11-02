"""Task 10D: final validation of the production BF16 pathwise gradient (C3) against the exact FP32 reference (C4A). No optimiser, no update.
  PHASE=fd|c3|c4a  CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy tests/ogpo_port/task10d.py
Criteria are read from the preregistration file written before any held-out sample exists.
fd   C4A finite differences at seed 10000, observation 0.
c3   canonical BF16 jit on the held-out pairs; each sample's gradients are held in a temporary file until C4A consumes it.
c4a  FP32 eager on the same pairs; compares, accumulates aggregates online, deletes each temporary file. No per-sample gradient is kept.
"""

from __future__ import annotations

import os

DET = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
NT = DET + " --xla_gpu_enable_triton_gemm=false"
PHASE = os.environ["PHASE"]
os.environ["TENB_FLAGS"] = DET if PHASE == "c3" else NT     # applied by task10b at import, before JAX is imported
os.environ["TAG"] = PHASE

import gc  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import task10b as T  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from task10c_compare import metrics  # noqa: E402

pv = T.pv
ROOT = Path(__file__).resolve().parents[2]
C = ROOT / "outputs" / "ogpo_port" / "task10c"
OUT = ROOT / "outputs" / "ogpo_port" / "task10d"
TMP = OUT / "tmp"
PRE = json.loads((OUT / "preregistration.json").read_text())
PAIRS = [(o, s) for o, seeds in PRE["held_out"].items() for s in seeds]
SMALL = (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)


def tiny(path, a):      # verified bitwise harmless in Task 10A
    return jnp.zeros((18,) if "layers" in jax.tree_util.keystr(path) else (1,), a.dtype)


def cache_file(obs):
    return C / "bf16_cache_canonical.npz" if obs == "0" else OUT / f"bf16_cache_canonical_obs{obs}.npz"


def load_cache(z, prefix, mask_key, leaves, treedef):
    return (jnp.asarray(z[mask_key]), jax.tree.unflatten(treedef, [jnp.asarray(z[f"{prefix}{i}"]).astype(jnp.bfloat16) for i in range(len(leaves))]))


def setup(H, res):
    nnx, sha = H["nnx"], H["sha"]
    e = H["load_real_env"]()
    teacher = e["model"]
    student, rec = H["make_student"](teacher, H["ALPHA"], H["PERT_SEED"])
    res["perturbation"], res["env"] = rec, T.env_record()
    res["input_sha256"] = {"student_all": sha(nnx.split(student)[1]), "teacher_all": sha(nnx.split(teacher)[1])}
    canon = json.loads((C / "run_C3.json").read_text())
    assert res["input_sha256"]["student_all"] == canon["input_sha256"]["student_all"] and res["input_sha256"]["teacher_all"] == canon["input_sha256"]["teacher_all"], "parameter hash mismatch"
    obs_data = {}
    for o in (["0"] if PHASE == "fd" else sorted({p[0] for p in PAIRS})):
        obs = e["observation"]((int(o),))
        sc, tc = pv.build_prefix(student, obs), pv.build_prefix(teacher, obs)
        sl, sd = jax.tree.flatten(sc.kv_cache)
        tl, td = jax.tree.flatten(tc.kv_cache)
        if PHASE == "c3" and o != "0":
            np.savez(cache_file(o), **{f"s{i}": np.asarray(a).astype(np.float32) for i, a in enumerate(sl)}, **{f"t{i}": np.asarray(a).astype(np.float32) for i, a in enumerate(tl)},
                     mask_s=np.asarray(sc.prefix_mask), mask_t=np.asarray(tc.prefix_mask))
        z = np.load(cache_file(o))
        can_s, can_t = load_cache(z, "s", "mask_s", sl, sd), load_cache(z, "t", "mask_t", tl, td)
        obs_data[o] = {"obs": sc.observation, "s": can_s, "t": can_t}
        res.setdefault("observations", {})[o] = {"obs_sha": sha(sc.observation), "cache_s_sha": sha(can_s), "cache_t_sha": sha(can_t), "own_flags_cache_s_sha": sha((sc.prefix_mask, sc.kv_cache))}
    assert res["observations"]["0"]["cache_s_sha"] == canon["canonical_cache_sha"]["s"] and res["observations"]["0"]["cache_t_sha"] == canon["canonical_cache_sha"]["t"], "canonical cache hash mismatch"
    assert res["observations"]["0"]["obs_sha"] == canon["input_sha256"]["obs"], "observation hash mismatch"
    if PHASE == "c3":
        gd_s, theta, other_s = nnx.split(student, H["in_groups"](H["TRAINABLE"]), ...)
        gd_t, t_all = nnx.split(teacher)
        rest_s, rest_t = (other_s,), (t_all,)
    else:
        from task10a_derivatives import promote_models

        t32, s32 = promote_models(e["cfg"].checkpoint.openpi_config, teacher, student)
        res["promotion_exact"] = sha(jax.tree.map(lambda a: a.astype(jnp.bfloat16), nnx.split(s32)[1])) == res["input_sha256"]["student_all"]
        assert res["promotion_exact"]
        up = lambda t: jax.tree.map(lambda a: a.astype(jnp.float32) if a.dtype == jnp.bfloat16 else a, t)  # noqa: E731
        for d in obs_data.values():
            d["s"], d["t"] = up(d["s"]), up(d["t"])
        gd_s, theta, vlm_s, other_s = nnx.split(s32, H["in_groups"](H["TRAINABLE"]), H["in_groups"](H["VLM"]), ...)
        gd_t, vlm_t, other_t = nnx.split(t32, H["in_groups"](H["VLM"]), ...)
        vlm_s, vlm_t = (jax.tree_util.tree_map_with_path(tiny, v) for v in (vlm_s, vlm_t))
        rest_s, rest_t = (vlm_s, other_s), (vlm_t, other_t)
        del s32, t32
    del teacher, student
    e.clear()
    jax.clear_caches()
    gc.collect()
    print("[setup]", json.dumps({k: res.get(k) for k in ("observations", "input_sha256", "promotion_exact")}), flush=True)
    return gd_s, gd_t, theta, rest_s, rest_t, obs_data


def make_I(H, rest_s, rest_t, d, seed):
    x0, eps = H["injected"](seed, 1)
    return {"rest_s": rest_s, "teacher": rest_t, "obs": d["obs"], "s_cache": d["s"], "t_cache": d["t"], "x0": jnp.asarray(x0), "eps": jnp.asarray(eps)}, H["sha"](x0), H["sha"](eps)


def phase_fd(H, res):
    flat = H["_flat"]
    gd_s, gd_t, theta, rest_s, rest_t, obs_data = setup(H, res)
    I, sx, se = make_I(H, rest_s, rest_t, obs_data["0"], PRE["fd"]["seed"])
    f = T.canonical(gd_s, gd_t)
    obj = lambda t: f(t, I)[0]  # noqa: E731
    keys = sorted(flat(theta))
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat(theta).items()}
    names = [by_id[id(l)] for l in leaves]
    (o, a), g = jax.value_and_grad(f, has_aux=True)(theta, I)
    gfull32 = T.grad_host(H, g)
    del g
    ref = np.load(C / "reference_gradients" / f"C4A_{PRE['fd']['seed']}.npz")
    prim = np.load(C / f"primal_C4A_{PRE['fd']['seed']}.npz")
    res["reproduces_C4A"] = {"objective": float(o), "objective_task10c": float(prim["objective"]), "objective_bitwise": float(o) == float(prim["objective"]), "gradient_bitwise": all(np.array_equal(gfull32[k], ref[f"f__{k}"]) for k in keys),
                             "chain_bitwise": bool(np.array_equal(np.asarray(a["chain"]), prim["chain"]))}
    print("[reproduce C4A]", json.dumps(res["reproduces_C4A"]), flush=True)
    assert all(v for k, v in res["reproduces_C4A"].items() if k.endswith("bitwise")), "C4A not reproduced"
    gfull = {k: gfull32[k].astype(np.float64) for k in keys}
    del gfull32
    gdet = {k: ref[f"d__{k}"].astype(np.float64) for k in keys}

    def dot(x, y):
        return sum(float(np.sum(x[k] * y[k])) for k in x)

    gn = dot(gfull, gfull) ** 0.5
    r = np.random.default_rng(PRE["fd"]["random_seed"])
    dirs = {"full": (gfull, SMALL), "full_minus_detached": ({k: gfull[k] - gdet[k] for k in keys}, SMALL),
            "random_all": ({k: r.standard_normal(gfull[k].shape) for k in keys}, SMALL[2:] + (0.3, 1.0, 3.0, 10.0)),
            "random_heads": ({k: (r.standard_normal(gfull[k].shape) if not k.startswith("PaliGemma") else np.zeros_like(gfull[k])) for k in keys}, SMALL + (0.3, 1.0))}
    th = {k: np.asarray(v, np.float32).astype(np.float64) for k, v in flat(theta).items()}

    def build(arrs):
        return jax.tree.unflatten(treedef, [jnp.asarray(arrs[k], jnp.float32) for k in names])

    res["fd"] = {}
    for name, (d, steps) in dirs.items():
        n = dot(d, d) ** 0.5
        u = {k: d[k] / n for k in keys}
        rows = []
        for h in steps:
            plus = {k: (th[k] + h * u[k]).astype(np.float32) for k in keys}
            minus = {k: (th[k] - h * u[k]).astype(np.float32) for k in keys}
            fp, fm = float(obj(build(plus))), float(obj(build(minus)))
            ad = sum(float(np.sum(gfull[k] * (plus[k].astype(np.float64) - minus[k].astype(np.float64)))) for k in keys) / (2.0 * h)   # on the step actually taken
            del plus, minus
            fd = (fp - fm) / (2.0 * h)
            rows.append({"h": h, "f_plus": fp, "f_minus": fm, "fd": fd, "ad_on_step": ad, "rel_err": abs(fd - ad) / abs(ad), "scaled_err": abs(fd - ad) / gn})
            print(f"[fd {name}] {json.dumps(rows[-1])}", flush=True)
        best = min(rows, key=lambda x: x["rel_err"])
        tail = rows[-3:]
        slope = float(np.polyfit(np.log10([x["h"] for x in tail]), np.log10([max(abs(x["fd"] - x["ad_on_step"]), 1e-30) for x in tail]), 1)[0])
        rule = PRE["fd"]
        ok_err = best["rel_err"] <= rule["rel_err_max"]
        ok_slope = (rule["slope_min"] <= slope <= rule["slope_max"]) if name in rule["slope_required_for"] else None
        res["fd"][name] = {"rows": rows, "best_h": best["h"], "best_rel_err": best["rel_err"], "best_scaled_err": best["scaled_err"], "slope_largest_3_h": slope, "err_pass": ok_err, "slope_pass": ok_slope,
                           "pass": bool(ok_err and ok_slope is not False), "ad_projection": dot(gfull, u)}
        print(f"[fd summary {name}] {json.dumps({k: v for k, v in res['fd'][name].items() if k != 'rows'})}", flush=True)
    res["phase1_pass"] = all(v["pass"] for v in res["fd"].values())
    res["noise_sha"] = {"x0": sx, "eps": se}


def phase_c3(H, res):
    gd_s, gd_t, theta, rest_s, rest_t, obs_data = setup(H, res)
    f = T.canonical(gd_s, gd_t)
    full = jax.jit(jax.value_and_grad(f, has_aux=True))
    det = jax.jit(jax.grad(lambda t, inp, st: f(t, inp, "detached", st)[0]))
    TMP.mkdir(parents=True, exist_ok=True)
    res["samples"] = {}
    for o, seed in PAIRS:
        I, sx, se = make_I(H, rest_s, rest_t, obs_data[o], seed)
        t0 = time.perf_counter()
        (ob, a), g = full(theta, I)
        jax.block_until_ready(g)
        t_full = time.perf_counter() - t0
        p, gh = T.primal(ob, a), T.grad_host(H, g)
        del g
        t0 = time.perf_counter()
        gd = det(theta, I, jnp.asarray(p["chain"]))
        jax.block_until_ready(gd)
        t_det = time.perf_counter() - t0
        gdh = T.grad_host(H, gd)
        del gd
        assert all(np.all(np.isfinite(v)) for v in gh.values()) and all(np.all(np.isfinite(v)) for v in gdh.values()), "non-finite gradient"
        np.savez(TMP / f"C3_{o}_{seed}.npz", **{f"f__{k}": v for k, v in gh.items()}, **{f"d__{k}": v for k, v in gdh.items()}, **{f"p__{k}": v for k, v in p.items()})
        res["samples"][f"{o}_{seed}"] = {"x0_sha": sx, "eps_sha": se, "objective": float(p["objective"]), "full_s": t_full, "det_s": t_det, "leaf_counts_full": H["report"](gh)["groups"], "leaf_counts_det": H["report"](gdh)["groups"], "zero_leaves": H["report"](gh)["zero_leaf_paths"]}
        print(f"[c3 {o} {seed}] objective {float(p['objective'])!r} {t_full:.1f}s {t_det:.1f}s peak {H['_peak_gib']()}", flush=True)
        del gh, gdh, p


def phase_c4a(H, res):
    gd_s, gd_t, theta, rest_s, rest_t, obs_data = setup(H, res)
    f = T.canonical(gd_s, gd_t)
    full = jax.value_and_grad(f, has_aux=True)
    det = jax.grad(lambda t, inp, st: f(t, inp, "detached", st)[0])
    c3run = json.loads((OUT / "run_c3.json").read_text())
    res["observations_match_c3"] = {o: {k: res["observations"][o][k] == c3run["observations"][o][k] for k in ("obs_sha", "cache_s_sha", "cache_t_sha")} for o in res["observations"]}
    assert all(all(v.values()) for v in res["observations_match_c3"].values()), "held-out inputs differ between C3 and C4A"
    agg = {o: None for o in sorted({p[0] for p in PAIRS})}
    res["samples"] = {}
    thr = PRE["thresholds"]

    def verdict(m_full, m_state, obj_rel=None):
        v = {"full_cosine": m_full["overall"]["cosine"] >= thr["full_cosine_min"], "full_rel": m_full["overall"]["rel_l2_diff"] <= thr["full_rel_max"],
             "state_cosine": m_state["overall"]["cosine"] >= thr["state_cosine_min"], "state_rel": m_state["overall"]["rel_l2_diff"] <= thr["state_rel_max"]}
        if obj_rel is not None:
            v["objective_rel"] = obj_rel <= thr["objective_rel_max"]
        return v

    def trim(m):
        return {k: v for k, v in m.items() if k != "per_leaf"}

    for o, seed in PAIRS:
        I, sx, se = make_I(H, rest_s, rest_t, obs_data[o], seed)
        key = f"{o}_{seed}"
        assert sx == c3run["samples"][key]["x0_sha"] and se == c3run["samples"][key]["eps_sha"], "noise differs between C3 and C4A"
        t0 = time.perf_counter()
        (ob, a), g = full(theta, I)
        jax.block_until_ready(g)
        t_full = time.perf_counter() - t0
        p, gh = T.primal(ob, a), T.grad_host(H, g)
        del g
        t0 = time.perf_counter()
        gd = det(theta, I, jnp.asarray(p["chain"]))
        jax.block_until_ready(gd)
        t_det = time.perf_counter() - t0
        gdh = T.grad_host(H, gd)
        del gd
        assert all(np.all(np.isfinite(v)) for v in gh.values()) and all(np.all(np.isfinite(v)) for v in gdh.values()), "non-finite gradient"
        tmp = TMP / f"C3_{o}_{seed}.npz"
        z = np.load(tmp)
        keys = sorted(gh)
        c3 = {"f": {k: z[f"f__{k}"].astype(np.float64) for k in keys}, "d": {k: z[f"d__{k}"].astype(np.float64) for k in keys}}
        pc = {k: z[f"p__{k}"].astype(np.float64) for k in ("chain", "vs", "vt", "kl", "objective")}
        z.close()
        r = {"f": {k: gh[k].astype(np.float64) for k in keys}, "d": {k: gdh[k].astype(np.float64) for k in keys}}
        for x in (c3, r):
            x["s"] = {k: x["f"][k] - x["d"][k] for k in keys}
        mf, md, ms = metrics(c3["f"], r["f"]), metrics(c3["d"], r["d"]), metrics(c3["s"], r["s"])
        obj_rel = abs(float(pc["objective"]) - float(p["objective"])) / float(p["objective"])
        v = verdict(mf, ms, obj_rel)
        pr = {k: np.asarray(p[k]).astype(np.float64) for k in ("chain", "vs", "vt", "kl")}
        res["samples"][key] = {"observation": o, "seed": seed, "x0_sha": sx, "eps_sha": se, "objective_c3": float(pc["objective"]), "objective_c4a": float(p["objective"]), "objective_rel": obj_rel, "full": mf, "detached": md, "state": ms, "pass": v, "all_pass": all(v.values()),
                               "state_over_full_norm": ms["overall"]["norm_b"] / mf["overall"]["norm_b"], "x_max_abs": float(np.abs(pc["chain"] - pr["chain"]).max()), "vs_rel_l2": float(np.linalg.norm(pc["vs"] - pr["vs"]) / np.linalg.norm(pr["vs"])),
                               "vt_rel_l2": float(np.linalg.norm(pc["vt"] - pr["vt"]) / np.linalg.norm(pr["vt"])), "kl_rel_max": float((np.abs(pc["kl"] - pr["kl"]) / pr["kl"]).max()), "c4a_full_s": t_full, "c4a_det_s": t_det,
                               "c4a_zero_leaves": H["report"](gh)["zero_leaf_paths"], "c4a_groups": H["report"](gh)["groups"]}
        if agg[o] is None:
            agg[o] = {"c3_f": {k: np.zeros_like(c3["f"][k]) for k in keys}, "c3_d": {k: np.zeros_like(c3["f"][k]) for k in keys}, "r_f": {k: np.zeros_like(c3["f"][k]) for k in keys}, "r_d": {k: np.zeros_like(c3["f"][k]) for k in keys}, "n": 0}
        for k in keys:
            agg[o]["c3_f"][k] += c3["f"][k]; agg[o]["c3_d"][k] += c3["d"][k]; agg[o]["r_f"][k] += r["f"][k]; agg[o]["r_d"][k] += r["d"][k]
        agg[o]["n"] += 1
        print(f"[pair {key}] full cos {mf['overall']['cosine']:.5f} rel {mf['overall']['rel_l2_diff']:.4f} | state cos {ms['overall']['cosine']:.5f} rel {ms['overall']['rel_l2_diff']:.4f} | exp {mf['action_expert']['cosine']:.5f}/{mf['action_expert']['rel_l2_diff']:.4f} {ms['action_expert']['cosine']:.5f}/{ms['action_expert']['rel_l2_diff']:.4f} | heads {mf['action_heads']['cosine']:.5f}/{mf['action_heads']['rel_l2_diff']:.4f} {ms['action_heads']['cosine']:.5f}/{ms['action_heads']['rel_l2_diff']:.4f} | obj {float(pc['objective']):.4f}/{float(p['objective']):.4f} rel {obj_rel:.4f} | pass {all(v.values())} | {t_full:.0f}s {t_det:.0f}s", flush=True)
        (OUT / "pairs_partial.json").write_text(json.dumps({"samples": {a_: {k: (trim(v_) if k in ("full", "detached", "state") else v_) for k, v_ in b_.items()} for a_, b_ in res["samples"].items()}}, default=str))
        tmp.unlink()                        # the per-sample C3 arrays are not kept
        del c3, r, gh, gdh, p, pc, pr
        gc.collect()

    def agg_metrics(sets):
        n = sum(s["n"] for s in sets)
        mk = lambda name: {k: sum(s[name][k] for s in sets) / n for k in sets[0][name]}  # noqa: E731
        cf, cd, rf, rd = mk("c3_f"), mk("c3_d"), mk("r_f"), mk("r_d")
        cs, rs = {k: cf[k] - cd[k] for k in cf}, {k: rf[k] - rd[k] for k in rf}
        mf, md, ms = metrics(cf, rf), metrics(cd, rd), metrics(cs, rs)
        return {"n": n, "full": mf, "detached": md, "state": ms, "pass": verdict(mf, ms), "all_pass": all(verdict(mf, ms).values())}

    res["aggregates"] = {f"observation_{o}": agg_metrics([agg[o]]) for o in agg}
    res["aggregates"]["all"] = agg_metrics(list(agg.values()))
    del agg
    S = res["samples"].values()

    def spread(fn):
        a_ = np.array([fn(s) for s in S], float)
        return {"median": float(np.median(a_)), "min": float(a_.min()), "max": float(a_.max())}

    res["distribution"] = {g: {sub: {"cosine": spread(lambda s, g=g, sub=sub: s[g][sub]["cosine"]), "rel_l2": spread(lambda s, g=g, sub=sub: s[g][sub]["rel_l2_diff"])} for sub in ("overall", "action_expert", "action_heads")} for g in ("full", "detached", "state")}
    res["distribution"]["objective_rel"] = spread(lambda s: s["objective_rel"])
    res["subgroup_flags"] = {key: [f"{g}/{sub}" for g, c_, r_ in (("full", thr["full_cosine_min"], thr["full_rel_max"]), ("state", thr["state_cosine_min"], thr["state_rel_max"])) for sub in ("action_expert", "action_heads") if s[g][sub]["cosine"] < c_ or s[g][sub]["rel_l2_diff"] > r_] for key, s in res["samples"].items()}
    res["phase2_samples_all_pass"] = all(s["all_pass"] for s in S)
    res["phase2_failures"] = {k: [c for c, ok in s["pass"].items() if not ok] for k, s in res["samples"].items() if not s["all_pass"]}
    res["phase2_aggregates_all_pass"] = all(a_["all_pass"] for a_ in res["aggregates"].values())
    print("[distribution]", json.dumps(res["distribution"]), flush=True)
    print("[aggregates]", json.dumps({k: {"n": v["n"], "pass": v["pass"], **{g: {sub: {"cosine": v[g][sub]["cosine"], "rel": v[g][sub]["rel_l2_diff"]} for sub in ("overall", "action_expert", "action_heads")} for g in ("full", "detached", "state")}} for k, v in res["aggregates"].items()}), flush=True)
    print("[verdict]", json.dumps({k: res[k] for k in ("phase2_samples_all_pass", "phase2_failures", "phase2_aggregates_all_pass")}), "subgroup flags", {k: v for k, v in res["subgroup_flags"].items() if v}, flush=True)
    (OUT / "pairs_partial.json").unlink()


def main():
    H = T.helpers()
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"phase": PHASE, "xla_flags": os.environ["XLA_FLAGS"], "prereg_sha256": hashlib.sha256((OUT / "preregistration.json").read_bytes()).hexdigest()}
    {"fd": phase_fd, "c3": phase_c3, "c4a": phase_c4a}[PHASE](H, res)
    res["memory"] = {"peak_gib": H["_peak_gib"](), "rss_gib": H["rss_gib"]()}
    (OUT / f"run_{PHASE}.json").write_text(json.dumps(res, indent=1, default=str))
    print("[end]", json.dumps(res["memory"]), "phase1_pass", res.get("phase1_pass"), flush=True)


if __name__ == "__main__":
    main()
