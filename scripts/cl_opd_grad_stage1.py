"""Stage 1 (read-only): recompute every per-trajectory gradient of saved continual-OPD batches at saved weights.
  CLVLA_USE_GPU=1 CLVLA_GPU=G PYTHONPATH=scripts:tests/ogpo_port scripts/py-policy scripts/cl_opd_grad_stage1.py --out outputs/cl_opd/analysis/grad_stage1 [--only-validate]

Nothing under outputs/cl_opd/{full_*,t063_noreplay_*} is written; every result goes to --out (one .npz per run/batch/weights).
Weights: batch file update_U.npz was trained at the master after U updates, so checkpoint adapter_U (and, for U = 0, the
task-0 student) is EXACT for it; any other checkpoint is an approximation whose error this script exists to measure.
Per trajectory (observation x noise chain, the unit the trainer differentiates) it stores: exact squared norms by
parameter group, the pathwise-KL parts, velocity RMS and, for inner products, a CountSketch of the gradient
(2^22 buckets, unbiased inner products, error ~ |a||b|/2048); the Gram matrix of a batch is computed from the sketches.
Exact batch sums (new half, old half) are accumulated as in the trainer, so the logged grad_norm, KL means, velocity RMS
and action errors can be reproduced at the exact anchors before anything else is computed (--only-validate stops there).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

import opd_lora_train as OT  # noqa: E402  (imports task10b first: deterministic XLA flags)

import jax
import jax.numpy as jnp
import numpy as np
from clvla.cl_opd.buffer import read_meta
from clvla.cl_opd.config import VARIANTS
from clvla.config import load_experiment_config
from clvla.ogpo import pi05_pathwise_kl as pw
from clvla.ogpo import pi05_sde as sde
from clvla.ogpo import pi05_velocity as pv
from clvla.ogpo.pi05_sde_policy import event_noise, model_inputs, model_of
from clvla.policy import load_official_policy
from flax import nnx
from opd_lora_student import build_student_model, initial_full, is_expert, is_trainable_full, load_delta_teacher

ROOT = Path(__file__).resolve().parents[1]
cfg = sde.SDEConfig()
K_LOG = 22
K = 1 << K_LOG
A1, B1, A2, B2 = 0x9E3779B1, 0x7F4A7C15, 0x85EBCA77, 0xC2B2AE3D  # fixed odd multipliers: the same sketch for every gradient
TEACHERS = "outputs/teachers/lora-25demos-s500"
T063 = f"{TEACHERS}/t063_LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray"
T035 = f"{TEACHERS}/t035_KITCHEN_SCENE7_open_the_microwave"
INIT = "outputs/cl_opd/full_20261108T214604Z/k0_t035/student/checkpoints/adapter_400.npz"
RUNS = {
    "replay": {"student": "outputs/cl_opd/full_20261108T214604Z/k1_t063/student", "old_teacher": T035, "old_buffer": "outputs/cl_opd/full_20261108T214604Z/buffer", "old_weight": 0.5},
    "noreplay": {"student": "outputs/cl_opd/t063_noreplay_20261109T150326Z/k1_t063/student", "old_teacher": None, "old_buffer": None, "old_weight": None},
}
SPIKE = 1e4
COMPARE = ("grad_norm", "grad_expert", "grad_heads", "kl_mean", "kl_fresh", "kl_replay", "kl_old", "kl_executed", "kl_unexecuted", "kl_padded", "velocity_rms_student", "velocity_rms_teacher", "act_xyz", "act_rot", "act_grip_agree")


def history(student):
    rows = []
    for f in ("training_metrics.csv", "training_metrics_from_u100.csv"):
        rows += list(csv.DictReader(open(ROOT / student / f)))
    return {int(r["update"]): r for r in rows}


def selection(hist, last_row=201, seed=0):
    """Batches U (CSV row U + 1): the exact anchors 0, 100, 200; every spike (grad_norm > 1e4) in rows 1..last_row; and as
    many ordinary updates of the same phase (grad_norm <= 2 x the phase median), drawn with a fixed seed."""
    rows = {r: h for r, h in hist.items() if r <= last_row}
    spikes = sorted(r - 1 for r, h in rows.items() if float(h["grad_norm"]) > SPIKE)
    rng = np.random.default_rng(seed)
    normal = []
    for ctl in ("teacher", "student"):
        g = {r: float(h["grad_norm"]) for r, h in rows.items() if h["controller"] == ctl}
        med = float(np.median(list(g.values())))
        pool = sorted(r - 1 for r, v in g.items() if v <= 2 * med and r - 1 not in (0, 100, 200))
        n = sum(1 for u in spikes if hist[u + 1]["controller"] == ctl)
        normal += sorted(int(x) for x in rng.choice(pool, size=min(n, len(pool)), replace=False))
    return {"anchors": [0, 100, 200], "spikes": spikes, "normal": sorted(normal)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--only-validate", action="store_true")
    ap.add_argument("--runs", default="replay,noreplay")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    flat = OT.T.helpers()["_flat"]
    conf = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    base_model = model_of(policy)
    teacher, _, _ = load_delta_teacher(base_model, ROOT / T063, None, *VARIANTS["lora"])
    gd_t, t_state = nnx.split(teacher)
    adapter0 = initial_full(str(Path(conf.checkpoint.local_dir) / "params"))
    keys = sorted(adapter0)

    def load_master(path):
        z = np.load(ROOT / path)
        assert sorted(k.replace("__", "/") for k in z.files) == keys, f"{path}: leaves differ"
        return {k: np.asarray(z[k.replace("/", "__")], np.float32) for k in keys}

    init = load_master(INIT)
    init_sha = hashlib.sha256(b"".join(init[k].tobytes() for k in keys)).hexdigest()
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
    sizes = {k: int(np.prod(adapter0[k].shape)) for k in keys}
    offs, o_ = {}, 0
    for k in keys:
        offs[k], o_ = o_, o_ + sizes[k]
    assert o_ < 2**32
    group = {k: "expert" if is_expert(k) else "heads" for k in keys}

    def forward(master):
        return jax.tree.unflatten(treedef, [jnp.asarray(master[k]).astype(dtypes[k]) for k in order])

    def merge_s(th):
        return nnx.merge(gd_s, th, vlm, other)

    def s_prefix(st, obs):
        c = pv.build_prefix(nnx.merge(gd_all, st), obs)
        return c.observation, c.prefix_mask, c.kv_cache

    def t_prefix(st, obs):
        c = pv.build_prefix(nnx.merge(gd_t, st), obs)
        return c.observation, c.prefix_mask, c.kv_cache

    def obj(th, ob, sm, sk, tm, tk, x0, eps):
        o = pw.pathwise_kl(merge_s(th), pv.PrefixCache(ob, sm, sk), pw.frozen_view(nnx.merge(gd_t, t_state)), pv.PrefixCache(ob, tm, tk), x0, eps, cfg, mode="full")
        return jnp.sum(o["path_kl"]), {"vs": o["velocities_student"], "vt": o["velocities_teacher"]}

    jp_s, jp_t = jax.jit(s_prefix), jax.jit(t_prefix)
    j_grad = jax.jit(jax.value_and_grad(obj, has_aux=True))
    j_vs = jax.jit(lambda th, ob, sm, sk, x, s_: pv.velocity(merge_s(th), pv.PrefixCache(ob, sm, sk), x, s_))
    j_vt = jax.jit(lambda tt, ob, tm, tk, x, s_: pv.velocity(nnx.merge(gd_t, tt), pv.PrefixCache(ob, tm, tk), x, s_))
    old_cache = {}

    def old_teacher(path):
        if path not in old_cache:
            m, _, _ = load_delta_teacher(base_model, ROOT / path, None, *VARIANTS["lora"])
            gd, st = nnx.split(m)

            def pre(st_, obs):
                c = pv.build_prefix(nnx.merge(gd, st_), obs)
                return c.observation, c.prefix_mask, c.kv_cache

            def obj_o(th, tt, ob, sm, sk, tm, tk, x0, eps):
                o = pw.pathwise_kl(merge_s(th), pv.PrefixCache(ob, sm, sk), pw.frozen_view(nnx.merge(gd, tt)), pv.PrefixCache(ob, tm, tk), x0, eps, cfg, mode="full")
                return jnp.sum(o["path_kl"]), {"vs": o["velocities_student"], "vt": o["velocities_teacher"]}

            old_cache[path] = (st, jax.jit(pre), jax.jit(jax.value_and_grad(obj_o, has_aux=True)))
        return old_cache[path]

    @jax.jit
    def sketch_and_sq(g):
        sk_ = jnp.zeros(K, jnp.float32)
        sq = {"expert": jnp.float32(0), "heads": jnp.float32(0)}
        for k in keys:
            v = g[k].reshape(-1).astype(jnp.float32)
            idx = jnp.arange(v.size, dtype=jnp.uint32) + jnp.uint32(offs[k])
            b = ((idx * jnp.uint32(A1) + jnp.uint32(B1)) >> (32 - K_LOG)).astype(jnp.int32)
            s = jnp.where(((idx * jnp.uint32(A2) + jnp.uint32(B2)) >> 31) == 1, 1.0, -1.0).astype(jnp.float32)
            sk_ = sk_.at[b].add(v * s)
            sq[group[k]] = sq[group[k]] + jnp.sum(v * v)
        return sk_, sq

    @jax.jit
    def acc_add(acc, g):
        return {k: acc[k] + g[k].astype(jnp.float32) for k in keys}

    j_leaf_dots = jax.jit(lambda x, y: {k: jnp.vdot(x[k].reshape(-1), y[k].reshape(-1)) for k in keys})

    def dots(x, y):
        """Exact inner products of two FP32 sums, per parameter group (leaf dots on the device, the group sums in FP64)."""
        d = {k: float(v) for k, v in j_leaf_dots(x, y).items()}
        return {grp: sum(d[k] for k in keys if group[k] == grp) for grp in ("expert", "heads")}

    def to_phys(inputs, latent):
        return np.asarray(policy._output_transform({"state": np.asarray(inputs["state"]), "actions": np.asarray(latent)})["actions"])

    def act_error(th, ob, sm, sk, tm, tk, inputs, x0):
        ys, yt = jnp.asarray(x0), jnp.asarray(x0)
        for s_ in cfg.times():
            ys = ys - (1.0 / cfg.num_steps) * j_vs(th, ob, sm, sk, ys, s_)
            yt = yt - (1.0 / cfg.num_steps) * j_vt(t_state, ob, tm, tk, yt, s_)
        a_s, a_t = to_phys(inputs, np.asarray(ys)[0])[:5], to_phys(inputs, np.asarray(yt)[0])[:5]
        return {"xyz": float(np.sqrt(np.mean((a_s[:, :3] - a_t[:, :3]) ** 2))), "rot": float(np.sqrt(np.mean((a_s[:, 3:6] - a_t[:, 3:6]) ** 2))), "grip_agree": float(np.mean(np.sign(a_s[:, 6]) == np.sign(a_t[:, 6])))}

    def run_batch(name, spec, prompts, u, label, master, exact):
        f_out = out / name / f"u{u:03d}_{label}.npz"
        if f_out.exists():
            return json.loads(str(np.load(f_out)["summary"]))
        theta = forward(master)
        z = np.load(ROOT / spec["student"] / "batches" / f"update_{u:03d}.npz")
        sources = [str(s) for s in z["sources"]]
        phases = [str(p) for p in z["phases"]]
        chains = 2 if str(z["controller"]) == "teacher" else 1
        n_sel = len(sources)
        is_old = [s.startswith("old_") for s in sources]
        is_rep = [s.startswith("replay_") for s in sources]
        n_new = sum(not x for x in is_old)
        acc = {k: jnp.zeros(adapter0[k].shape, jnp.float32) for k in keys}
        acc_o = {k: jnp.zeros(adapter0[k].shape, jnp.float32) for k in keys} if any(is_old) else None
        meta, sks, sq_e, sq_h, kl, vrms_s, vrms_t, loss, errs = [], [], [], [], [], [], [], [], []
        t0 = time.time()
        for j in range(n_sel):
            req = {"observation/image": z[f"image_{j}"], "observation/wrist_image": z[f"wrist_{j}"], "observation/state": z[f"state_{j}"], "prompt": prompts["old" if is_old[j] else "new"]}
            obs, inputs = model_inputs(policy, req)
            ob, sm, sk = jp_s(s_full, obs)
            if is_old[j]:
                st_o, pre_o, grad_o = old_teacher(spec["old_teacher"])
                _, tm, tk = pre_o(st_o, obs)
            else:
                _, tm, tk = jp_t(t_state, obs)
            for m in range(chains):
                x0, eps = (z[f"x0_{j}"], z[f"eps_{j}"]) if m == 0 else event_noise(OT.seed_ok(450000 + 1000 * u + 10 * j + m), cfg)
                if is_old[j]:
                    (val, aux), g = grad_o(theta, st_o, ob, sm, sk, tm, tk, x0, eps)
                else:
                    (val, aux), g = j_grad(theta, ob, sm, sk, tm, tk, x0, eps)
                gh = flat(g)
                s_vec, sq = sketch_and_sq(gh)
                if is_old[j]:
                    acc_o = acc_add(acc_o, gh)
                else:
                    acc = acc_add(acc, gh)
                del g, gh
                aux = jax.tree.map(np.asarray, aux)
                sks.append(np.asarray(s_vec, np.float32))
                sq_e.append(float(sq["expert"]))
                sq_h.append(float(sq["heads"]))
                loss.append(float(val))
                kl.append(OT.kl_parts(aux["vs"], aux["vt"]))
                vrms_s.append(float(np.sqrt(np.mean(aux["vs"] ** 2))))
                vrms_t.append(float(np.sqrt(np.mean(aux["vt"] ** 2))))
                meta.append((j, m, sources[j], phases[j], is_old[j], is_rep[j]))
                if exact and m == 0 and not is_old[j]:
                    errs.append(act_error(theta, ob, sm, sk, tm, tk, inputs, z[f"x0_{j}"]))
        S = np.stack(sks).astype(np.float64)
        gram = S @ S.T
        n_traj = n_new * chains
        n_old_traj = (n_sel - n_new) * chains
        d_nn = dots(acc, acc)
        d_oo = dots(acc_o, acc_o) if acc_o is not None else {"expert": 0.0, "heads": 0.0}
        d_no = dots(acc, acc_o) if acc_o is not None else {"expert": 0.0, "heads": 0.0}
        w_ = spec["old_weight"] if n_old_traj else None
        cn, co = ((1 - w_) / n_traj, w_ / n_old_traj) if w_ is not None else (1.0 / n_traj, 0.0)
        gm_sq = {g_: cn * cn * d_nn[g_] + co * co * d_oo[g_] + 2 * cn * co * d_no[g_] for g_ in ("expert", "heads")}
        new_k = [x for x in kl]
        sel_new = [i for i, mm in enumerate(meta) if not mm[4]]
        rec = {
            "grad_norm": (gm_sq["expert"] + gm_sq["heads"]) ** 0.5, "grad_expert": gm_sq["expert"] ** 0.5, "grad_heads": gm_sq["heads"] ** 0.5,
            "kl_mean": float(np.mean([new_k[i]["full"] for i in sel_new])),
            "kl_fresh": float(np.mean([new_k[i]["full"] for i, mm in enumerate(meta) if not mm[4] and not mm[5]])) if any(not mm[4] and not mm[5] for mm in meta) else None,
            "kl_replay": float(np.mean([new_k[i]["full"] for i, mm in enumerate(meta) if mm[5]])) if any(mm[5] for mm in meta) else None,
            "kl_old": float(np.mean([new_k[i]["full"] for i, mm in enumerate(meta) if mm[4]])) if any(mm[4] for mm in meta) else None,
            **{f"kl_{p}": float(np.mean([new_k[i][p] for i in sel_new])) for p in ("executed", "unexecuted", "padded")},
            "velocity_rms_student": float(np.mean([vrms_s[i] for i in sel_new])), "velocity_rms_teacher": float(np.mean([vrms_t[i] for i in sel_new])),
            **({"act_xyz": float(np.mean([e["xyz"] for e in errs])), "act_rot": float(np.mean([e["rot"] for e in errs])), "act_grip_agree": float(np.mean([e["grip_agree"] for e in errs]))} if errs else {}),
        }
        summary = {"run": name, "u": u, "row": u + 1, "weights": label, "exact": exact, "chains": chains, "n_traj": n_traj, "n_old_traj": n_old_traj, "old_weight": w_, "seconds": time.time() - t0, "recomputed": rec,
                   "exact_sums": {"new_new": d_nn, "old_old": d_oo, "new_old": d_no}}
        (out / name).mkdir(exist_ok=True)
        np.savez_compressed(f_out, summary=json.dumps(summary), gram=gram, sq_expert=np.array(sq_e), sq_heads=np.array(sq_h), loss=np.array(loss),
                            kl_full=np.array([x["full"] for x in kl]), kl_executed=np.array([x["executed"] for x in kl]), kl_unexecuted=np.array([x["unexecuted"] for x in kl]), kl_padded=np.array([x["padded"] for x in kl]),
                            vrms_s=np.array(vrms_s), vrms_t=np.array(vrms_t), obs=np.array([mm[0] for mm in meta]), chain=np.array([mm[1] for mm in meta]), source=np.array([mm[2] for mm in meta]),
                            phase=np.array([mm[3] for mm in meta]), is_old=np.array([mm[4] for mm in meta]), is_replay=np.array([mm[5] for mm in meta]))
        print(f"[batch] {name} u={u} {label} exact={exact} traj={len(meta)} {summary['seconds']:.0f}s grad_norm={rec['grad_norm']:.1f} kl_mean={rec['kl_mean']:.2f}", flush=True)
        return summary

    plan = {}
    ckpt_masters = {}
    for name in a.runs.split(","):
        spec = RUNS[name]
        hist = history(spec["student"])
        sel = selection(hist)
        plan[name] = sel
        prompts = {"new": str(np.load(sorted((ROOT / spec["student"]).parent.glob("teacher_episodes/u000_*.npz"))[0])["prompt"])}
        if spec["old_buffer"]:
            prompts["old"] = read_meta(ROOT / spec["old_buffer"], "t035")["prompt"]
        mj = json.loads((ROOT / spec["student"] / "metrics.json").read_text())
        assert mj["identity"]["initial_adapter_sha"] == init_sha, f"{name}: the run did not start from the task-0 student"
        weights = {"init": init, "c100": load_master(f"{spec['student']}/checkpoints/adapter_100.npz"), "c200": load_master(f"{spec['student']}/checkpoints/adapter_200.npz")}
        ckpt_masters[name] = weights
        # 1) validation at the exact anchors: the recomputation must reproduce the logged row
        val = []
        for u, lab in ((0, "init"), (100, "c100"), (200, "c200")):
            s = run_batch(name, spec, prompts, u, lab, weights[lab], True)
            h = hist[u + 1]
            cmp_ = {}
            for k in COMPARE:
                if k in s["recomputed"] and s["recomputed"][k] is not None and h.get(k) not in (None, "", "None"):
                    hv, rv = float(h[k]), float(s["recomputed"][k])
                    cmp_[k] = {"logged": hv, "recomputed": rv, "rel_diff": abs(rv - hv) / max(abs(hv), 1e-12)}
            val.append({"u": u, "row": u + 1, "weights": lab, "compare": cmp_})
        (out / name).mkdir(exist_ok=True)
        (out / name / "validation.json").write_text(json.dumps(val, indent=1))
        worst = max(c["rel_diff"] for v in val for k, c in v["compare"].items() if k in ("grad_norm", "kl_mean", "kl_fresh", "kl_old"))
        print(f"[validate] {name}: worst relative difference on grad_norm/KL = {worst:.2e}", flush=True)
        if worst > 0.05:
            raise SystemExit(f"{name}: recomputation does not reproduce the logged metrics (worst {worst:.3f}); stopping before the sweep")
    (out / "plan.json").write_text(json.dumps(plan, indent=1))
    if a.only_validate:
        return
    # 2) the sweep: every selected batch at c100 and c200; Phase-1 batches (U < 100) also at the task-0 student
    for name in a.runs.split(","):
        spec, sel, weights = RUNS[name], plan[name], ckpt_masters[name]
        prompts = {"new": str(np.load(sorted((ROOT / spec["student"]).parent.glob("teacher_episodes/u000_*.npz"))[0])["prompt"])}
        if spec["old_buffer"]:
            prompts["old"] = read_meta(ROOT / spec["old_buffer"], "t035")["prompt"]
        for u in sorted(set(sel["anchors"] + sel["spikes"] + sel["normal"])):
            for lab in (["init"] if u < 100 else []) + ["c100", "c200"]:
                run_batch(name, spec, prompts, u, lab, weights[lab], (u, lab) in ((0, "init"), (100, "c100"), (200, "c200")))
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
