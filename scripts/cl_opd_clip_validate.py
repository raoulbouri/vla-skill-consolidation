"""Stage 2 offline validation of per-trajectory gradient clipping (no training: no optimizer step, nothing written to a run).
  GPU:  CLVLA_USE_GPU=1 CLVLA_GPU=G PYTHONPATH=scripts:tests/ogpo_port scripts/py-policy scripts/cl_opd_clip_validate.py --out outputs/cl_opd/analysis/clip_stage2
  CPU:  PYTHONPATH=scripts:tests/ogpo_port scripts/py-policy scripts/cl_opd_clip_validate.py --out outputs/cl_opd/analysis/clip_stage2 --sensitivity

GPU mode, per stored batch (the six exact-weight anchors, then batches at checkpoint weights where Stage 1 simulated the
largest clipping effect): pass 1 recomputes every trajectory gradient as the trainer does (unclipped sums + exact norms
and fingerprints); the logged grad_norm / grad_expert / grad_heads must be reproduced at the anchors (clipping disabled);
then for k = 3 and k = 2 the trainer's own clip_pass2 / clip_finish (scripts/opd_lora_train.py) build the clipped update,
which is compared with the Stage 1 simulation of the same rule (outputs/cl_opd/analysis/grad_stage1). Runtime of every
part and device memory are measured. CPU mode: threshold sensitivity over all Stage 1 batches (norms are exact there).
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import opd_lora_train as OT  # noqa: E402  (deterministic XLA flags first)

import cl_opd_grad_report as RP
import cl_opd_grad_stage1 as S1
import jax
import jax.numpy as jnp
import numpy as np
from clvla.cl_opd.traj_clip import CLASSES, TrajClip, loss_weights, scales, thresholds

ROOT = Path(__file__).resolve().parents[1]
STAGE1 = ROOT / "outputs/cl_opd/analysis/grad_stage1"
KS = (3.0, 2.0)
TOL_DEG = 1.0


def deg(c):
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def choose(report):
    """Exact anchors first; then per run the 4 checkpoint-weight batches with the largest simulated class-median x3 angle
    and the 2 Phase-1 batches at the task-0 student with the largest one (the small fresh class)."""
    jobs = [(r, u, lab, True) for r in ("replay", "noreplay") for u, lab in ((0, "init"), (100, "c100"), (200, "c200"))]
    for r in ("replay", "noreplay"):
        rows = [x for x in report["batches"][r]["rows"] if not x["exact"]]
        key = lambda x: x["class_median_x3:angle_to_unclipped_deg"]  # noqa: E731
        ck = sorted((x for x in rows if x["weights"] != "init"), key=key, reverse=True)[:4]
        p1 = sorted((x for x in rows if x["weights"] == "init"), key=key, reverse=True)[:2]
        jobs += [(r, x["u"], x["weights"], False) for x in ck + p1]
    return jobs


def sensitivity(out):
    """CPU: thresholds k in {1.5, 2, 3, 4, 5} over every Stage 1 batch (exact norms; directions from the sketched Gram)."""
    res = {}
    for r in ("replay", "noreplay"):
        w = 0.5 if r == "replay" else None
        rows = []
        hist = S1.history(RP.STUDENT[r])
        for f in sorted((STAGE1 / r).glob("u*_*.npz")):
            b = RP.load(f)
            a = RP.weights(b, w)
            ctl = "teacher" if b["s"]["chains"] == 2 else "student"
            spike = float(hist[b["s"]["u"] + 1]["grad_norm"]) > S1.SPIKE
            for k in (1.5, 2.0, 3.0, 4.0, 5.0):
                t = thresholds(b["norm"], b["cls"], k)
                s = scales(b["norm"], t)
                gg0, c0, _ = RP.influence(b["G"], a)
                gg, c, _ = RP.influence(b["G"], a * s)
                cos = float((a * s) @ b["G"] @ a) / np.sqrt(gg * gg0)
                row = {"u": b["s"]["u"], "weights": b["s"]["weights"], "exact": b["s"]["exact"], "controller": ctl, "logged_spike": spike, "k": k,
                       "clip_frac": float(np.mean(s < 1)), "angle_deg": deg(cos), "norm_ratio": float(np.sqrt(gg / gg0)), "max_infl_before": float(c0.max()), "max_infl_after": float(c.max())}
                for cl in CLASSES:
                    m = b["cls"] == cl
                    if m.any():
                        row[f"clip_frac_{cl}"] = float(np.mean(s[m] < 1))
                rows.append(row)
            # Phase 1: the fresh class has 16 trajectories (8 observations x 2 chains); how stable is its median?
            if ctl == "teacher":
                m = b["cls"] == "fresh"
                n = b["norm"][m]
                rng = np.random.default_rng(0)
                boots = np.median(rng.choice(n, size=(2000, n.size), replace=True), axis=1)
                flips = {}
                for k in (2.0, 3.0):
                    base = n > k * np.median(n)
                    flips[str(k)] = float(np.mean([np.mean((n > k * bm) != base) for bm in boots]))
                rows[-1]["phase1_fresh"] = {"n": int(n.size), "median": float(np.median(n)), "boot_cv": float(boots.std() / boots.mean()),
                                            "boot_p05_p95": [float(np.percentile(boots, 5)), float(np.percentile(boots, 95))], "decision_flip_rate": flips,
                                            "max_over_median": float(n.max() / np.median(n))}
        res[r] = rows
    (out / "sensitivity.json").write_text(json.dumps(res, indent=1))
    lines = ["## Threshold sensitivity over all Stage 1 batches (medians; spike = logged grad > 1e4)", "",
             "| run | controller | batches | k | clip frac fresh / replay / old | angle to unclipped (deg) | max influence before -> after |", "|---|---|---|---|---|---|---|"]
    for r, rows in res.items():
        for ctl in ("teacher", "student"):
            for sp in (False, True):
                for k in (1.5, 2.0, 3.0, 4.0, 5.0):
                    x = [y for y in rows if y["controller"] == ctl and y["logged_spike"] == sp and y["k"] == k]
                    if not x:
                        continue
                    cf = " / ".join(f"{np.median([y[f'clip_frac_{c}'] for y in x if f'clip_frac_{c}' in y]):.3f}" if any(f"clip_frac_{c}" in y for y in x) else "-" for c in CLASSES)
                    lines.append(f"| {r} | {ctl}{' spike' if sp else ''} | {len(x)} | {k} | {cf} | {np.median([y['angle_deg'] for y in x]):.1f} | {np.median([y['max_infl_before'] for y in x]):.3f} -> {np.median([y['max_infl_after'] for y in x]):.3f} |")
    lines += ["", "## Phase 1 fresh class (16 trajectories): stability of the class median", "", "| run | batches | bootstrap CV of the median (median) | decision flip rate k=3 / k=2 (median, max) | max/median norm (max) |", "|---|---|---|---|---|"]
    for r, rows in res.items():
        p = [y["phase1_fresh"] for y in rows if "phase1_fresh" in y]
        if p:
            lines.append(f"| {r} | {len(p)} | {np.median([q['boot_cv'] for q in p]):.3f} | {np.median([q['decision_flip_rate']['3.0'] for q in p]):.3f} ({max(q['decision_flip_rate']['3.0'] for q in p):.3f}) / "
                         f"{np.median([q['decision_flip_rate']['2.0'] for q in p]):.3f} ({max(q['decision_flip_rate']['2.0'] for q in p):.3f}) | {max(q['max_over_median'] for q in p):.2f} |")
    (out / "sensitivity.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--sensitivity", action="store_true", help="CPU only: threshold sensitivity from the Stage 1 files")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    if a.sensitivity:
        sensitivity(out)
        return
    from clvla.cl_opd.buffer import read_meta
    from clvla.cl_opd.config import VARIANTS
    from clvla.config import load_experiment_config
    from clvla.ogpo import pi05_pathwise_kl as pw
    from clvla.ogpo import pi05_velocity as pv
    from clvla.ogpo.pi05_sde_policy import event_noise, model_inputs, model_of
    from clvla.policy import load_official_policy
    from flax import nnx
    from opd_lora_student import build_student_model, initial_full, is_expert, is_trainable_full, load_delta_teacher

    cfg = S1.cfg
    flat = OT.T.helpers()["_flat"]
    conf = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    base_model = model_of(policy)
    teacher, _, _ = load_delta_teacher(base_model, ROOT / S1.T063, None, *VARIANTS["lora"])
    gd_t, t_state = nnx.split(teacher)
    adapter0 = initial_full(str(Path(conf.checkpoint.local_dir) / "params"))
    keys = sorted(adapter0)

    def load_master(path):
        z = np.load(ROOT / path)
        return {k: np.asarray(z[k.replace("/", "__")], np.float32) for k in keys}

    init = load_master(S1.INIT)
    student = build_student_model(base_model, init, True)
    trainable = nnx.All(nnx.Param, lambda path, _: is_trainable_full("/".join(str(p) for p in path)))
    vlm_f = nnx.All(nnx.Param, lambda path, _: "/".join(str(p) for p in path).startswith("PaliGemma/img") or ("/llm/" in "/".join(str(p) for p in path) and "_1" not in "/".join(str(p) for p in path)))
    gd_s, theta0, vlm, other = nnx.split(student, trainable, vlm_f, ...)
    assert sorted(flat(theta0)) == keys
    gd_all, s_full = nnx.split(student)
    leaves, treedef = jax.tree.flatten(theta0)
    by_id = {id(v): k for k, v in flat(theta0).items()}
    order = [by_id[id(l)] for l in leaves]
    dtypes = {k: l.dtype for k, l in zip(order, leaves)}

    def group(k):  # the trainer's group() in full mode
        return "expert" if is_expert(k) else "heads"

    def forward(master):
        return jax.tree.unflatten(treedef, [jnp.asarray(master[k]).astype(dtypes[k]) for k in order])

    def merge_s(th):
        return nnx.merge(gd_s, th, vlm, other)

    def obj(th, ob, sm, sk, tm, tk, x0, eps):
        o = pw.pathwise_kl(merge_s(th), pv.PrefixCache(ob, sm, sk), pw.frozen_view(nnx.merge(gd_t, t_state)), pv.PrefixCache(ob, tm, tk), x0, eps, cfg, mode="full")
        return jnp.sum(o["path_kl"]), {"vs": o["velocities_student"]}

    jp_s = jax.jit(lambda st, obs: (lambda c: (c.observation, c.prefix_mask, c.kv_cache))(pv.build_prefix(nnx.merge(gd_all, st), obs)))
    jp_t = jax.jit(lambda st, obs: (lambda c: (c.observation, c.prefix_mask, c.kv_cache))(pv.build_prefix(nnx.merge(gd_t, st), obs)))
    j_grad = jax.jit(jax.value_and_grad(obj, has_aux=True))
    # the trainer's two jitted helpers, verbatim
    j_leaf_stats = jax.jit(lambda g: {k: (jnp.vdot(g[k].reshape(-1).astype(jnp.float32), g[k].reshape(-1).astype(jnp.float32)), jnp.sum(g[k].astype(jnp.float32))) for k in keys})
    j_leaf_dots = jax.jit(lambda x, y: {k: jnp.vdot(x[k].reshape(-1).astype(jnp.float32), y[k].reshape(-1).astype(jnp.float32)) for k in keys})
    old_t = {}

    def old_teacher(path):
        if path not in old_t:
            m_, _, _ = load_delta_teacher(base_model, ROOT / path, None, *VARIANTS["lora"])
            gd, st = nnx.split(m_)

            def obj_o(th, tt, ob, sm, sk, tm, tk, x0, eps):
                o = pw.pathwise_kl(merge_s(th), pv.PrefixCache(ob, sm, sk), pw.frozen_view(nnx.merge(gd, tt)), pv.PrefixCache(ob, tm, tk), x0, eps, cfg, mode="full")
                return jnp.sum(o["path_kl"]), {"vs": o["velocities_student"]}

            pre = jax.jit(lambda st_, obs: (lambda c: (c.observation, c.prefix_mask, c.kv_cache))(pv.build_prefix(nnx.merge(gd, st_), obs)))
            old_t[path] = (st, pre, jax.jit(jax.value_and_grad(obj_o, has_aux=True)))
        return old_t[path]

    dev = jax.devices()[0]

    def mem():
        s = dev.memory_stats() or {}
        return {"in_use_gib": s.get("bytes_in_use", 0) / 2**30, "peak_gib": s.get("peak_bytes_in_use", 0) / 2**30}

    def dot(x, y):  # the trainer's host dot
        return sum(float(np.sum(np.asarray(x[k], np.float64) * np.asarray(y[k], np.float64))) for k in keys)

    def by(x):
        o = {"expert": 0.0, "heads": 0.0}
        for k in keys:
            o[group(k)] += float(np.sum(np.asarray(x[k], np.float64) ** 2))
        return {n: v**0.5 for n, v in o.items()}

    report = json.loads((STAGE1 / "report.json").read_text())
    jobs = choose(report)
    (out / "plan.json").write_text(json.dumps(jobs))
    masters, hists, prompts = {}, {}, {}
    for r in ("replay", "noreplay"):
        spec = S1.RUNS[r]
        masters[r] = {"init": init, "c100": load_master(f"{spec['student']}/checkpoints/adapter_100.npz"), "c200": load_master(f"{spec['student']}/checkpoints/adapter_200.npz")}
        hists[r] = S1.history(spec["student"])
        prompts[r] = {"new": str(np.load(sorted((ROOT / spec["student"]).parent.glob("teacher_episodes/u000_*.npz"))[0])["prompt"])}
        if spec["old_buffer"]:
            prompts[r]["old"] = read_meta(ROOT / spec["old_buffer"], "t035")["prompt"]
    results = []
    res_f = out / "results.jsonl"
    done = {(x["run"], x["u"], x["weights"]) for x in map(json.loads, res_f.read_text().splitlines())} if res_f.exists() else set()
    for r, u, lab, exact in jobs:
        if (r, u, lab) in done:
            continue
        spec = S1.RUNS[r]
        w = spec["old_weight"]
        theta = forward(masters[r][lab])
        z = np.load(ROOT / spec["student"] / "batches" / f"update_{u:03d}.npz")
        sources = [str(s) for s in z["sources"]]
        chains = 2 if str(z["controller"]) == "teacher" else 1
        is_old = [s.startswith("old_") for s in sources]
        n_new = sum(not x for x in is_old)
        n_fresh = sum(not s.startswith(("old_", "replay_")) for s in sources)
        assert all(sources[j].startswith("replay_") for j in range(n_fresh, n_new)) and not any(is_old[:n_new]), "batch layout: fresh, replay, old"

        def traj_grad(j, m, theta=theta, z=z, is_old=is_old, r=r, u=u):
            req = {"observation/image": z[f"image_{j}"], "observation/wrist_image": z[f"wrist_{j}"], "observation/state": z[f"state_{j}"], "prompt": prompts[r]["old" if is_old[j] else "new"]}
            obs, _ = model_inputs(policy, req)
            ob, sm, sk = jp_s(s_full, obs)
            x0, eps = (z[f"x0_{j}"], z[f"eps_{j}"]) if m == 0 else event_noise(OT.seed_ok(450000 + 1000 * u + 10 * j + m), cfg)
            if is_old[j]:
                st_o, pre_o, grad_o = old_teacher(spec["old_teacher"])
                _, tm, tk = pre_o(st_o, obs)
                return flat(grad_o(theta, st_o, ob, sm, sk, tm, tk, x0, eps)[1])
            _, tm, tk = jp_t(t_state, obs)
            return flat(j_grad(theta, ob, sm, sk, tm, tk, x0, eps)[1])

        # pass 1, as the trainer: unclipped sums + per-trajectory norms and fingerprints
        base = TrajClip(0.0, w if any(is_old) else None)
        acc = {k: jnp.zeros(adapter0[k].shape, jnp.float32) for k in keys}
        acc_old = {k: jnp.zeros(adapter0[k].shape, jnp.float32) for k in keys} if any(is_old) else None
        t_grad, t_norm, m0 = 0.0, 0.0, mem()
        for j in range(len(sources)):
            for m in range(chains):
                t0 = time.perf_counter()
                gh = traj_grad(j, m)
                jax.block_until_ready(gh)
                t1 = time.perf_counter()
                fp, sq = OT._leaf_check(j_leaf_stats(gh), keys, group)
                t2 = time.perf_counter()
                base.record(j, m, "fresh" if j < n_fresh else "replay" if j < n_new else "old", sq, fp)
                tgt = acc_old if is_old[j] else acc
                for k in keys:
                    tgt[k] = tgt[k] + gh[k].astype(jnp.float32)
                del gh
                t_grad, t_norm = t_grad + t1 - t0, t_norm + t2 - t1
        n_traj, n_old_traj = n_new * chains, (len(sources) - n_new) * chains
        if n_old_traj:  # the trainer's unclipped update, line for line
            g_u = {k: np.asarray((1.0 - w) * acc[k] / n_traj + w * acc_old[k] / n_old_traj, np.float32) for k in keys}
        else:
            g_u = {k: np.asarray(acc[k] / n_traj, np.float32) for k in keys}
        gn_u, gr_u = dot(g_u, g_u) ** 0.5, by(g_u)
        m1 = mem()
        h = hists[r][u + 1]
        rec = {"run": r, "u": u, "row": u + 1, "weights": lab, "exact": exact, "controller": str(z["controller"]), "n_traj": len(base.meta),
               "unclipped": {"grad_norm": gn_u, "grad_expert": gr_u["expert"], "grad_heads": gr_u["heads"]},
               "logged": {k: float(h[k]) for k in ("grad_norm", "grad_expert", "grad_heads")}, "seconds": {"pass1_grad": t_grad, "pass1_norms": t_norm, "per_traj_grad": t_grad / len(base.meta), "per_traj_norms": t_norm / len(base.meta), "logged_update": float(h["seconds"])},
               "memory": {"before_pass1": m0, "after_pass1": m1}}
        if exact:
            rec["disabled_vs_logged_rel"] = {k: abs(rec["unclipped"][k] - rec["logged"][k]) / rec["logged"][k] for k in rec["logged"]}
        # Stage 1 for the same batch and weights (norms exact there; directions from the sketched Gram)
        b1 = RP.load(STAGE1 / r / f"u{u:03d}_{lab}.npz")
        base.plan()
        rec["norms_vs_stage1_rel_max"] = float(np.max(np.abs(base.norm - b1["norm"]) / b1["norm"]))
        assert list(base.cls) == list(b1["cls"]), "trajectory order differs from Stage 1"
        a1 = RP.weights(b1, w if any(is_old) else None)
        assert np.allclose(a1, loss_weights(base.cls, w if any(is_old) else None))
        rec["k"] = {}
        for k in KS:
            clip = copy.deepcopy(base)
            clip.k = k
            d, info = OT.clip_pass2(clip, traj_grad, j_leaf_stats, j_leaf_dots, keys, group, True, acc, acc_old)
            m2 = mem()
            info["seconds_norms"] = t_norm
            t3 = time.perf_counter()
            g_c, gn, gr, row = OT.clip_finish(clip, g_u, gn_u, gr_u, d, info, dot, by, keys)
            t_fin = time.perf_counter() - t3
            del d
            # Stage 1 simulation of the same rule
            sim = RP.clip_eval(b1, a1, "class_median", k, None)
            s_sim = scales(b1["norm"], thresholds(b1["norm"], b1["cls"], k))
            b_mod, b_sim = clip.a * clip.s, a1 * s_sim
            G = b1["G"]
            cos_ms = float(b_mod @ G @ b_sim) / np.sqrt(float(b_mod @ G @ b_mod) * float(b_sim @ G @ b_sim))
            _, c_after, _ = RP.influence(G, b_mod)
            _, c_before, _ = RP.influence(G, a1)
            rec["k"][str(k)] = {
                "row": row, "seconds_finish": t_fin, "memory_after_pass2": m2,
                "same_clip_set": sorted(clip.todo) == sorted(np.flatnonzero(s_sim < 1).tolist()), "scale_max_abs_diff": float(np.max(np.abs(clip.s - s_sim))),
                "angle_enabled_vs_simulated_deg": deg(cos_ms), "angle_to_unclipped": {"enabled_exact": row["clip_angle_deg"], "simulated": sim["angle_to_unclipped_deg"]},
                "norm_ratio": {"enabled_exact": row["clip_norm_ratio"], "simulated": sim["norm_over_unclipped"]},
                "max_influence_sketch": {"before": float(c_before.max()), "after": float(c_after.max()), "before_argmax": int(c_before.argmax()), "after_argmax": int(c_after.argmax())},
                "top_influence_exact": {"unclipped_top": clip.top(False), "before": row["clip_top_unclipped_infl_before"], "after": row["clip_top_unclipped_infl_after"], "clipped_top": clip.top(True), "clipped_top_after": row["clip_top_clipped_infl_after"]},
                "per_traj": {"cls": clip.cls.tolist(), "norm": clip.norm.tolist(), "s": clip.s.tolist(), "tau": [float(x) if np.isfinite(x) else None for x in clip.tau]},
                "group_norms": {"clipped": gr, "unclipped": gr_u, "correction": {n: row[f"clip_correction_{n}"] for n in ("expert", "heads")}},
                "pass": bool(deg(cos_ms) < TOL_DEG and abs(row["clip_angle_deg"] - sim["angle_to_unclipped_deg"]) < TOL_DEG and sorted(clip.todo) == sorted(np.flatnonzero(s_sim < 1).tolist()) and row["clip_recompute_bitwise"]),
            }
            if k == 3.0 and exact:
                g_c3, clip3 = g_c, clip
            del g_c
        del acc, acc_old
        if exact:
            # the full two-pass alternative: recompute EVERY trajectory and sum s_i g_i directly (no cancellation), to measure
            # the FP32 error of the correction form used by the trainer and the cost of a full second pass
            t4 = time.perf_counter()
            dn = {k_: jnp.zeros(adapter0[k_].shape, jnp.float32) for k_ in keys}
            do = {k_: jnp.zeros(adapter0[k_].shape, jnp.float32) for k_ in keys} if any(is_old) else None
            for i, (j, m, _) in enumerate(clip3.meta):
                gh = traj_grad(j, m)
                tgt = do if is_old[j] else dn
                for k_ in keys:
                    tgt[k_] = tgt[k_] + float(clip3.s[i]) * gh[k_].astype(jnp.float32)
                del gh
            if n_old_traj:
                g_dir = {k_: np.asarray((1.0 - w) * dn[k_] / n_traj + w * do[k_] / n_old_traj, np.float32) for k_ in keys}
            else:
                g_dir = {k_: np.asarray(dn[k_] / n_traj, np.float32) for k_ in keys}
            t_full = time.perf_counter() - t4
            del dn, do
            nd = dot(g_dir, g_dir) ** 0.5
            diff = {k_: g_c3[k_].astype(np.float64) - g_dir[k_] for k_ in keys}
            rec["full_two_pass_k3"] = {"seconds": t_full, "rel_error_correction_vs_full": dot(diff, diff) ** 0.5 / nd, "angle_deg": deg(dot(g_c3, g_dir) / (dot(g_c3, g_c3) ** 0.5 * nd)),
                                       "group_rel_error": {n: by(diff)[n] / max(by(g_dir)[n], 1e-30) for n in ("expert", "heads")}, "memory": mem()}
            del g_dir, diff, g_c3
        del g_u
        with res_f.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        k3 = rec["k"]["3.0"]
        print(f"[batch] {r} u={u} {lab} exact={exact} traj={rec['n_traj']} grad_u={gn_u:.1f} logged={rec['logged']['grad_norm']:.1f} k3: clipped={k3['row']['clip_n']} angle={k3['row']['clip_angle_deg']:.2f} "
              f"sim={k3['angle_to_unclipped']['simulated']:.2f} vs_sim={k3['angle_enabled_vs_simulated_deg']:.3f} bitwise={k3['row']['clip_recompute_bitwise']} pass={k3['pass']} pass1={t_grad:.0f}s", flush=True)
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
