"""Pre-flight for the 5-task continual OPD run, GPU part. OFFLINE: no optimizer state, no training run; weights are only
perturbed in memory; every output goes to --out. Training demos demo_0..24 only (held-out demo_25..49 never read).
  CLVLA_USE_GPU=1 CLVLA_GPU=G PYTHONPATH=scripts:tests/ogpo_port scripts/py-policy scripts/cl_opd_preflight_gpu.py --out D --section e|abc

section e   E1 device memory with the student and all five teachers (+ the trainer's FP32 sums), one gradient per teacher;
            E2 the real OldTasks path on the two-task buffer: each old observation scored by its own vs the other teacher.
section abc A1 spike anatomy at the three exact spikes: full / teacher-blocked / detached gradients, per-denoising-step
               gradient norms, top singular values of dv/dx (student, teacher);
            A2 one-step lookahead from the exact spike weights along unclipped / clipped(k=3) / detached / teacher-blocked
               batch directions at 0.5x / 1x / 2x the logged step: KL, gripper agreement, action change on held-aside
               observations of neighbouring batches (and old t035 ones);
            B  all five tasks at the task-0 student: spike propensity (per-trajectory norms, chain share), teacher
               steepness, starting KL / gripper agreement to each future teacher;
            C1 5x5 cosines of per-task mean gradients (task-0 student and the no-replay t063 student);
            C2 5x5 forgetting sensitivity: one clipped-direction step of task i, KL change on task j;
            C3 t035 KL / gripper agreement on its buffer states at every saved checkpoint (success known at the ends).
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import opd_lora_train as OT  # noqa: E402  (deterministic XLA flags first)

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import yaml
from clvla.cl_opd.traj_clip import TrajClip

ROOT = Path(__file__).resolve().parents[1]
import os  # noqa: E402

SHARED = os.environ.get("PREFLIGHT_SHARED") == "1"  # a GPU other users are on: FP32 sums on the host, no big test allocations
PROC = Path(os.environ["CLVLA_PROCESSED_ROOT"]) / "libero90_aligned256_v1" / "hdf5"  # from .env (scripts/py-policy sources env.sh)
FULL = "outputs/cl_opd/full_20261108T214604Z"
NOREP = "outputs/cl_opd/t063_noreplay_20261109T150326Z"
INIT = f"{FULL}/k0_t035/student/checkpoints/adapter_400.npz"
RUNS = {"replay": {"student": f"{FULL}/k1_t063/student", "w": 0.5}, "noreplay": {"student": f"{NOREP}/k1_t063/student", "w": None}}
S1 = ROOT / "outputs/cl_opd/analysis/grad_stage1"
DEMOS = (0, 6, 12, 18)  # training demos only


def deg(c):
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def history(student):
    out = {}
    for f in ("training_metrics.csv", "training_metrics_from_u100.csv"):
        p = ROOT / student / f
        if p.exists():
            out.update({int(r["update"]): r for r in csv.DictReader(open(p))})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--section", required=True, choices=("e", "abc"))
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    T0 = time.time()

    from clvla.cl_opd.config import VARIANTS
    from clvla.config import load_experiment_config
    from clvla.data import read_instruction
    from clvla.ogpo import pi05_pathwise_kl as pw
    from clvla.ogpo import pi05_sde as sde
    from clvla.ogpo import pi05_velocity as pv
    from clvla.ogpo.pi05_sde_policy import event_noise, model_inputs, model_of
    from clvla.policy import load_official_policy
    from flax import nnx
    from opd_lora_student import build_student_model, initial_full, is_expert, is_trainable_full, load_delta_teacher

    cfg = sde.SDEConfig()
    K = cfg.num_steps
    W = np.asarray(OT.pkl.kl_coefficients(cfg)["weights"], np.float64)
    flat = OT.T.helpers()["_flat"]
    conf = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    base_model = model_of(policy)
    ccfg = yaml.safe_load((ROOT / "configs/cl_opd.yaml").read_text())
    TASKS = {t["id"]: t for t in ccfg["tasks"]}
    dev = jax.devices()[0]

    def mem():
        s = dev.memory_stats() or {}
        return {"in_use_gib": round(s.get("bytes_in_use", 0) / 2**30, 2), "peak_gib": round(s.get("peak_bytes_in_use", 0) / 2**30, 2), "limit_gib": round(s.get("bytes_limit", 0) / 2**30, 2)}

    def log(msg):
        print(f"[{time.time() - T0:7.0f}s] {msg}", flush=True)

    adapter0 = initial_full(str(Path(conf.checkpoint.local_dir) / "params"))
    keys = sorted(adapter0)

    def load_master(path):
        z = np.load(ROOT / path)
        return {k: np.asarray(z[k.replace("/", "__")], np.float32) for k in keys}

    init = load_master(INIT)
    student = build_student_model(base_model, init, True)
    trainable = nnx.All(nnx.Param, lambda path, _: is_trainable_full("/".join(str(p) for p in path)))
    vlm_f = nnx.All(nnx.Param, lambda path, _: "/".join(str(p) for p in path).startswith("PaliGemma/img") or ("/llm/" in "/".join(str(p) for p in path) and "_1" not in "/".join(str(p) for p in path)))
    gd_s, theta0, vlm, other = nnx.split(student, trainable, vlm_f, ...)
    gd_all, s_full = nnx.split(student)
    leaves, treedef = jax.tree.flatten(theta0)
    by_id = {id(v): k for k, v in flat(theta0).items()}
    order = [by_id[id(l)] for l in leaves]
    dtypes = {k: l.dtype for k, l in zip(order, leaves)}
    log(f"student built {mem()}")

    def group(k):
        return "expert" if is_expert(k) else "heads"

    def forward(master):
        return jax.tree.unflatten(treedef, [jnp.asarray(master[k]).astype(dtypes[k]) for k in order])

    def merge_s(th):
        return nnx.merge(gd_s, th, vlm, other)

    teachers = {}  # id -> state
    GD = {}

    def teacher(tid):
        if tid not in teachers:
            m_, _, _ = load_delta_teacher(base_model, ROOT / TASKS[tid]["teacher"], None, *VARIANTS[TASKS[tid]["variant"]])
            gd, st = nnx.split(m_)
            GD.setdefault("gd", gd)  # all teachers are the same LoRA variant: one graphdef, states as arguments (as the trainer)
            teachers[tid] = st
        return teachers[tid]

    def drop_teacher(tid):
        teachers.pop(tid, None)

    teacher("t063")
    gd_T = GD["gd"]

    def kl_out(th, tt, ob, sm, sk, tm, tk, x0, eps, mode, stored=None):
        return pw.pathwise_kl(merge_s(th), pv.PrefixCache(ob, sm, sk), pw.frozen_view(nnx.merge(gd_T, tt)), pv.PrefixCache(ob, tm, tk), x0, eps, cfg, mode=mode, stored_chain=stored)

    def obj(mode):
        def f(th, tt, ob, sm, sk, tm, tk, x0, eps):
            o = kl_out(th, tt, ob, sm, sk, tm, tk, x0, eps, mode)
            return jnp.sum(o["path_kl"]), {"chain": o["chain"], "vs": o["velocities_student"], "vt": o["velocities_teacher"], "kl": o["kl"]}
        return f

    g_full = jax.jit(jax.value_and_grad(obj("full"), has_aux=True))
    g_tb = jax.jit(jax.value_and_grad(obj("teacher_blocked"), has_aux=True))
    g_det = jax.jit(jax.value_and_grad(lambda th, tt, ob, sm, sk, tm, tk, stored: jnp.sum(kl_out(th, tt, ob, sm, sk, tm, tk, None, None, "detached", stored)["path_kl"])))
    f_kl = jax.jit(lambda th, tt, ob, sm, sk, tm, tk, x0, eps: kl_out(th, tt, ob, sm, sk, tm, tk, x0, eps, "full")["path_kl"])

    @jax.jit
    def step_sq(th, tt, ob, sm, sk, tm, tk, x0, eps):
        """Squared gradient norm of each denoising step's term w_k ||dv_k||^2 (one forward, K backward passes)."""
        kl, vjp = jax.vjp(lambda t: kl_out(t, tt, ob, sm, sk, tm, tk, x0, eps, "full")["kl"][0], th)

        def one(e):
            g = vjp(e)[0]
            return sum(jnp.vdot(l.reshape(-1).astype(jnp.float32), l.reshape(-1).astype(jnp.float32)) for l in jax.tree.leaves(g))
        return kl, jax.lax.map(one, jnp.eye(K, dtype=kl.dtype))

    def make_sigma(vel):
        @jax.jit
        def sigma(params, ob, m_, k_, x, s, u0):
            f = lambda xx: vel(params, ob, m_, k_, xx, s).astype(jnp.float32)  # noqa: E731

            def body(i, u):
                _, ju = jax.jvp(f, (x,), (u,))
                _, vj = jax.vjp(f, x)
                w_ = vj(ju)[0]
                return w_ / (jnp.linalg.norm(w_) + 1e-30)
            u = jax.lax.fori_loop(0, 8, body, u0 / jnp.linalg.norm(u0))
            return jnp.linalg.norm(jax.jvp(f, (x,), (u,))[1])
        return sigma

    j_vs = jax.jit(lambda th, ob, sm, sk, x, s: pv.velocity(merge_s(th), pv.PrefixCache(ob, sm, sk), x, s))
    j_vt = jax.jit(lambda tt, ob, tm, tk, x, s: pv.velocity(nnx.merge(gd_T, tt), pv.PrefixCache(ob, tm, tk), x, s))
    sig_s, sig_t = make_sigma(lambda th, ob, m_, k_, x, s: j_vs(th, ob, m_, k_, x, s)), make_sigma(lambda tt, ob, m_, k_, x, s: j_vt(tt, ob, m_, k_, x, s))
    jp_s = jax.jit(lambda st, obs: (lambda c: (c.observation, c.prefix_mask, c.kv_cache))(pv.build_prefix(nnx.merge(gd_all, st), obs)))
    jp_t = jax.jit(lambda st, obs: (lambda c: (c.observation, c.prefix_mask, c.kv_cache))(pv.build_prefix(nnx.merge(gd_T, st), obs)))
    j_leaf_stats = jax.jit(lambda g: {k: (jnp.vdot(g[k].reshape(-1).astype(jnp.float32), g[k].reshape(-1).astype(jnp.float32)), jnp.sum(g[k].astype(jnp.float32))) for k in keys})
    j_leaf_dots = jax.jit(lambda x, y: {k: jnp.vdot(x[k].reshape(-1).astype(jnp.float32), y[k].reshape(-1).astype(jnp.float32)) for k in keys})
    j_add_dev = jax.jit(lambda acc, g, c: {k: acc[k] + c * g[k].astype(jnp.float32) for k in keys})

    def j_add(acc, g, c):
        if not SHARED:
            return j_add_dev(acc, g, c)
        for k in keys:  # host FP32 sum, in place
            acc[k] += np.float32(c) * np.asarray(g[k], np.float32)
        return acc

    def zeros():
        if SHARED:
            return {k: np.zeros(adapter0[k].shape, np.float32) for k in keys}
        return {k: jnp.zeros(adapter0[k].shape, jnp.float32) for k in keys}

    def ddot(x, y):
        return sum(float(v) for v in jax.device_get(j_leaf_dots(x, y)).values())

    def gsq(g):
        _, sq = OT._leaf_check(j_leaf_stats(g), keys, group)
        return sq

    def prep(req, tid):
        obs, inputs = model_inputs(policy, req)
        ob, sm, sk = jp_s(s_full, obs)
        _, tm, tk = jp_t(teacher(tid), obs)
        return {"ob": ob, "sm": sm, "sk": sk, "tm": tm, "tk": tk, "inputs": inputs, "tid": tid}

    def to_phys(inputs, latent):
        return np.asarray(policy._output_transform({"state": np.asarray(inputs["state"]), "actions": np.asarray(latent)})["actions"])

    def ode(th_or_tt, p, x0, is_teacher):
        y = jnp.asarray(x0)
        for s_ in cfg.times():
            v = j_vt(th_or_tt, p["ob"], p["tm"], p["tk"], y, s_) if is_teacher else j_vs(th_or_tt, p["ob"], p["sm"], p["sk"], y, s_)
            y = y - (1.0 / K) * v
        return to_phys(p["inputs"], np.asarray(y)[0])[:5]

    def batch_obs(run, u, which, prompts):
        z = np.load(ROOT / RUNS[run]["student"] / "batches" / f"update_{u:03d}.npz")
        src = [str(s) for s in z["sources"]]
        out_ = []
        for j, s in enumerate(src):
            is_old = s.startswith("old_")
            kind = "old" if is_old else "replay" if s.startswith("replay_") else "fresh"
            if kind in which:
                req = {"observation/image": z[f"image_{j}"], "observation/wrist_image": z[f"wrist_{j}"], "observation/state": z[f"state_{j}"], "prompt": prompts["t035" if is_old else "t063"]}
                out_.append({"req": req, "x0": z[f"x0_{j}"], "eps": z[f"eps_{j}"], "kind": kind, "j": j, "tid": "t035" if is_old else "t063"})
        return out_, z

    def demo_obs(tid, n_per_demo=4, seed0=7_000_000):
        name = Path(TASKS[tid]["teacher"]).name.split("_", 1)[1]
        res = []
        with h5py.File(PROC / f"{name}_demo.hdf5", "r") as h:
            prompt = read_instruction(h)
            for d in DEMOS:
                g = h["data"][f"demo_{d}"]
                act = g["actions"][()]
                T = len(act)
                tog = list(np.flatnonzero(np.diff(np.sign(act[:, 6])) != 0))
                ts = [max(0, int(t) - 2) for t in tog[:2]][:n_per_demo]
                k_rem = n_per_demo - len(ts)
                ts += [int(x) for x in np.linspace(5, T - 10, k_rem + 2)[1:-1]] if k_rem > 0 else []
                for t in sorted(ts):
                    x0, eps = event_noise(OT.seed_ok(seed0 + 1000 * list(TASKS).index(tid) + 10 * d + t), cfg)
                    req = {"observation/image": np.ascontiguousarray(g["obs/agentview_rgb"][t][::-1, ::-1]), "observation/wrist_image": np.ascontiguousarray(g["obs/eye_in_hand_rgb"][t][::-1, ::-1]),
                           "observation/state": g["obs/state"][t].astype(np.float32), "prompt": prompt}
                    res.append({"req": req, "x0": x0, "eps": eps, "tid": tid, "demo": d, "t": t, "near_toggle": any(abs(t - x) <= 5 for x in tog)})
        return res

    def evaluate(th, items, cache):
        """KL (on-policy chain under th, stored noise), gripper agreement / xyz error vs the teacher ODE, and the ODE action itself."""
        kl, ga, xyz, acts = [], [], [], []
        for i, it in enumerate(items):
            p = prep(it["req"], it["tid"])
            tt = teacher(it["tid"])
            kl.append(float(f_kl(th, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], it["x0"], it["eps"])[0]))
            if ("at", i, it["tid"]) not in cache:
                cache[("at", i, it["tid"])] = ode(tt, p, it["x0"], True)
            at = cache[("at", i, it["tid"])]
            as_ = ode(th, p, it["x0"], False)
            acts.append(as_)
            ga.append(float(np.mean(np.sign(as_[:, 6]) == np.sign(at[:, 6]))))
            xyz.append(float(np.sqrt(np.mean((as_[:, :3] - at[:, :3]) ** 2))))
        return {"kl": kl, "grip_agree": ga, "xyz": xyz}, acts

    def summary(ev):
        return {k: float(np.mean(v)) for k, v in ev.items()} | {"kl_median": float(np.median(ev["kl"]))}

    # ------------------------------------------------------------------ section e
    if a.section == "e":
        R = {"after_student_and_t063_teacher": mem()}
        loaded = []
        for tid in ("t035", "t042", "t003", "t016"):
            try:
                teacher(tid)
                loaded.append(tid)
                R[f"after_teacher_{len(teachers)}"] = {"teachers": list(teachers), **mem()}
                log(f"teacher {tid} loaded {mem()}")
            except Exception as e:  # noqa: BLE001 - recorded, the point of the test
                R["teacher_load_failure"] = f"{tid}: {type(e).__name__}: {str(e)[:300]}"
                break
        # the trainer's FP32 sums (new, old, clipping correction) on top
        sums = []
        fp32_sum_gib = sum(int(np.prod(adapter0[k].shape)) for k in keys) * 4 / 2**30
        R["fp32_sum_gib_each"] = fp32_sum_gib
        if not SHARED:
            try:
                sums = [zeros() for _ in range(3)]
                R["with_three_fp32_sums"] = mem()
            except Exception as e:  # noqa: BLE001
                R["sums_failure"] = f"{type(e).__name__}: {str(e)[:300]}"
        theta = forward(init)
        grads_ok = {}
        for tid in list(teachers):
            try:
                it = demo_obs(tid, 1)[0]
                p = prep(it["req"], tid)
                (v, _), g = g_full(theta, teachers[tid], p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], it["x0"], it["eps"])
                sq = gsq(flat(g))
                grads_ok[tid] = {"loss": float(v), "grad_norm": sum(sq.values()) ** 0.5, "finite": bool(np.isfinite(float(v)) and np.isfinite(sum(sq.values())))}
                del g
            except Exception as e:  # noqa: BLE001
                grads_ok[tid] = f"{type(e).__name__}: {str(e)[:300]}"
        R["one_gradient_per_teacher"] = grads_ok
        R["after_gradients"] = mem()
        m_ = mem()
        # the trainer at task 4: everything loaded here + new/old/correction FP32 sums + the gradient's transient peak
        R["projection_task4_peak_gib"] = m_["peak_gib"] + (0 if not SHARED else 3 * fp32_sum_gib)
        R["training_cap_gib_at_0.92"] = 0.92 * 32607 / 1024
        R["shared_mode"] = SHARED
        del sums
        teachers.clear()
        OT.release("before_E2")
        # E2: the real OldTasks path on the existing two-task buffer
        tj = out / "old_teachers_e2.json"
        tj.write_text(json.dumps({t: {"delta": str(ROOT / TASKS[t]["teacher"]), "variant": TASKS[t]["variant"]} for t in ("t035", "t063")}))
        try:
            old = OT.OldTasks(ROOT / f"{FULL}/buffer", tj, base_model, 11)
            R["E2_mem_after_OldTasks"] = mem()
            plan = old.sampler.plan(40, 0)
            per = {}
            for task, ph, row in plan:
                if len(per.setdefault(task, [])) >= 6:
                    continue
                req = old.request(task, row)
                x0, eps = event_noise(OT.seed_ok(3_000_000 + len(per[task])), cfg)
                scores = {}
                for lab in ("t035", "t063"):
                    gd_o, st_o = old.teachers[lab]
                    obs, _ = model_inputs(policy, req)
                    ob, sm, sk = jp_s(s_full, obs)
                    _, tm, tk = jp_t(st_o, obs)
                    scores[lab] = float(f_kl(theta, st_o, ob, sm, sk, tm, tk, x0, eps)[0])
                per[task].append({"phase": ph, "row": int(row), "kl_own": scores[task], "kl_other": scores["t063" if task == "t035" else "t035"]})
            R["E2_labels"] = {t: {"n": len(v), "kl_own_median": float(np.median([x["kl_own"] for x in v])), "kl_other_median": float(np.median([x["kl_other"] for x in v])),
                                  "own_lower_in": int(sum(x["kl_own"] < x["kl_other"] for x in v))} for t, v in per.items()}
            R["E2_plan_counts"] = {t: sum(1 for x in plan if x[0] == t) for t in ("t035", "t063")}
        except Exception as e:  # noqa: BLE001
            R["E2_failure"] = f"{type(e).__name__}: {str(e)[:400]}"
        (out / "gpu_e.json").write_text(json.dumps(R, indent=1))
        log(json.dumps(R))
        return

    # ------------------------------------------------------------------ section abc
    prompts = {"t063": str(np.load(sorted((ROOT / FULL / "k1_t063").glob("teacher_episodes/u000_*.npz"))[0])["prompt"]),
               "t035": json.loads((ROOT / FULL / "buffer/t035/meta.json").read_text())["prompt"]}
    masters = {("init", None): init}

    def master(run, lab):
        if lab == "init":
            return init
        if (run, lab) not in masters:
            masters[(run, lab)] = load_master(f"{RUNS[run]['student']}/checkpoints/adapter_{lab[1:]}.npz")
        return masters[(run, lab)]

    def done(name):
        return (out / f"gpu_{name}.json").exists()

    def save(name, obj_):
        (out / f"gpu_{name}.json").write_text(json.dumps(obj_, indent=1))
        log(f"section {name} written {mem()}")

    # A1 ---------------------------------------------------------------
    SPIKES = [("replay", 100, "c100"), ("noreplay", 100, "c100"), ("noreplay", 200, "c200")]
    if not done("A1"):
        A1 = []
        for run, u, lab in SPIKES:
            theta = forward(master(run, lab))
            items, _ = batch_obs(run, u, ("fresh", "replay"), prompts)
            z1 = np.load(S1 / run / f"u{u:03d}_{lab}.npz")
            nrm = np.sqrt(z1["sq_expert"] + z1["sq_heads"])
            newi = [i for i in range(len(nrm)) if not z1["is_old"][i] and int(z1["chain"][i]) == 0]
            top = sorted(newi, key=lambda i: -nrm[i])[:3]
            med = float(np.median(nrm[newi]))
            ctrl = sorted(newi, key=lambda i: abs(nrm[i] - med))[:5]
            for i in top + ctrl:
                j = int(z1["obs"][i])
                it = next(x for x in items if x["j"] == j)
                p = prep(it["req"], "t063")
                tt = teacher("t063")
                (v, aux), gf = g_full(theta, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], it["x0"], it["eps"])
                gf = flat(gf)
                (_, _), gt = g_tb(theta, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], it["x0"], it["eps"])
                gt = flat(gt)
                _, gd_ = g_det(theta, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], aux["chain"])
                gd_ = flat(gd_)
                nf, nt, nd = (sum(gsq(x).values()) ** 0.5 for x in (gf, gt, gd_))
                state = {k: gf[k].astype(jnp.float32) - gd_[k].astype(jnp.float32) for k in keys}
                ns = sum(gsq(state).values()) ** 0.5
                rec = {"run": run, "u": u, "obs": j, "role": "top" if i in top else "median", "stage1_norm": float(nrm[i]), "kl": float(v), "phase": str(z1["phase"][i]),
                       "norm_full": nf, "norm_teacher_blocked": nt, "norm_detached": nd, "norm_state": ns, "full_by_group": gsq(gf),
                       "cos_full_tb": ddot(gf, gt) / (nf * nt), "cos_full_det": ddot(gf, gd_) / (nf * nd), "cos_full_state": ddot(gf, state) / (nf * ns)}
                del gt, gd_, state, gf
                kl_k, sq_k = step_sq(theta, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], it["x0"], it["eps"])
                rec["step_kl"] = np.asarray(kl_k).tolist()
                rec["step_grad_norm"] = np.sqrt(np.asarray(sq_k, np.float64)).tolist()
                dv = np.asarray(aux["vs"] - aux["vt"], np.float64)[0]
                rec["step_dv_rms"] = np.sqrt(np.mean(dv ** 2, axis=(1, 2))).tolist()
                rec["step_dv_gripper_rms"] = np.sqrt(np.mean(dv[:, :, 6] ** 2, axis=1)).tolist()
                ch = aux["chain"]
                u0 = jnp.asarray(np.random.default_rng(0).standard_normal((1, 10, 32)), jnp.float32)
                rec["sigma_student"] = [float(sig_s(theta, p["ob"], p["sm"], p["sk"], ch[:, k], s_, u0)) for k, s_ in enumerate(cfg.times())]
                rec["sigma_teacher"] = [float(sig_t(tt, p["ob"], p["tm"], p["tk"], ch[:, k], s_, u0)) for k, s_ in enumerate(cfg.times())]
                A1.append(rec)
                log(f"A1 {run} u{u} obs{j} {rec['role']}: full {nf:.0f} tb {nt:.0f} det {nd:.0f} state {ns:.0f} sigS max {max(rec['sigma_student']):.1f} sigT max {max(rec['sigma_teacher']):.1f}")
        save("A1", A1)

    # A2 ---------------------------------------------------------------
    if not done("A2"):
        A2 = []
        for run, u, lab in SPIKES + [("replay", 200, "c200")]:
            w = RUNS[run]["w"]
            M = master(run, lab)
            theta = forward(M)
            items, z = batch_obs(run, u, ("fresh", "replay", "old"), prompts)
            chains = 2 if str(z["controller"]) == "teacher" else 1
            base = TrajClip(3.0, w if any(x["kind"] == "old" for x in items) else None)

            def inputs_of(jx, m, items=items, u=u):
                it = items[jx]
                p = prep(it["req"], it["tid"])
                x0, eps = (it["x0"], it["eps"]) if m == 0 else event_noise(OT.seed_ok(450000 + 1000 * u + 10 * it["j"] + m), cfg)
                return it, p, x0, eps

            def traj_full(jx, m, theta=theta):
                it, p, x0, eps = inputs_of(jx, m)
                (_, aux), g = g_full(theta, teacher(it["tid"]), p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], x0, eps)
                return flat(g), aux

            chain_h, sums = {}, {}
            for mode_ in ("F", "T", "D"):
                pair = [zeros(), zeros()]
                for jx, it in enumerate(items):
                    for m in range(chains):
                        slot = 1 if it["kind"] == "old" else 0
                        if mode_ == "F":
                            g, aux = traj_full(jx, m)
                            fp, sq = OT._leaf_check(j_leaf_stats(g), keys, group)
                            base.record(jx, m, it["kind"], sq, fp)
                            chain_h[(jx, m)] = np.asarray(aux["chain"])
                        else:
                            it_, p, x0, eps = inputs_of(jx, m)
                            tt = teacher(it_["tid"])
                            g = flat(g_tb(theta, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], x0, eps)[1]) if mode_ == "T" else flat(g_det(theta, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], jnp.asarray(chain_h[(jx, m)]))[1])
                        pair[slot] = j_add(pair[slot], g, 1.0)
                        del g
                if mode_ == "F":
                    accF_dev = pair  # kept on the device for clip_pass2
                else:
                    sums[mode_] = [{k: np.asarray(x[k]) for k in keys} for x in pair]
                    del pair
            n_new = sum(1 for x in items if x["kind"] != "old") * chains
            n_old = sum(1 for x in items if x["kind"] == "old") * chains

            def combine(pair):
                if n_old:
                    return {k: np.asarray((1.0 - w) * pair[0][k] / n_new + w * pair[1][k] / n_old, np.float32) for k in keys}
                return {k: np.asarray(pair[0][k] / n_new, np.float32) for k in keys}

            def hdot(x, y):
                return sum(float(np.sum(np.asarray(x[k], np.float64) * np.asarray(y[k], np.float64))) for k in keys)

            def by(x):
                o = {"expert": 0.0, "heads": 0.0}
                for k in keys:
                    o[group(k)] += float(np.sum(np.asarray(x[k], np.float64) ** 2))
                return {n_: v_ ** 0.5 for n_, v_ in o.items()}

            G = {"unclipped": combine(accF_dev)}
            clip = copy.deepcopy(base)
            d, info = OT.clip_pass2(clip, lambda jx, m: traj_full(jx, m)[0], j_leaf_stats, j_leaf_dots, keys, group, not SHARED, accF_dev[0], accF_dev[1] if n_old else None)
            gn_u = hdot(G["unclipped"], G["unclipped"]) ** 0.5
            G["clipped_k3"], _, _, crow = OT.clip_finish(clip, G["unclipped"], gn_u, by(G["unclipped"]), d, info, hdot, by, keys)
            del d
            G["detached"], G["teacher_blocked"] = combine(sums["D"]), combine(sums["T"])
            del accF_dev, sums, chain_h
            norms = {n_: hdot(g_, g_) ** 0.5 for n_, g_ in G.items()}
            cos = {f"{x}|{y}": hdot(G[x], G[y]) / (norms[x] * norms[y]) for i_, x in enumerate(G) for y in list(G)[i_ + 1:]}
            # evaluation set: other batches' observations (never the batch that produced the gradient)
            ev_new, ev_old = [], []
            for uu in [uu for uu in range(u - 3, u + 4) if uu != u]:
                its, _ = batch_obs(run, uu, ("fresh", "old"), prompts)
                ev_new += [x for x in its if x["kind"] == "fresh"][:4]
                ev_old += [x for x in its if x["kind"] == "old"][:3]
            ev_new, ev_old = ev_new[:24], ev_old[:16]
            cache = {}
            base_new, acts0 = evaluate(theta, ev_new, cache)
            base_old, _ = evaluate(theta, ev_old, cache) if ev_old else ({}, None)
            step = float(history(RUNS[run]["student"])[u + 1]["master_step"])
            rec = {"run": run, "u": u, "weights": lab, "logged_grad_norm": float(history(RUNS[run]["student"])[u + 1]["grad_norm"]), "logged_master_step": step,
                   "direction_norms": norms, "direction_cos": cos, "clip_row": {k: v for k, v in crow.items() if isinstance(v, (int, float, str, bool))},
                   "n_eval_new": len(ev_new), "n_eval_old": len(ev_old), "base_new": summary(base_new), "base_old": summary(base_old) if ev_old else None, "steps": []}
            for name, g_ in G.items():
                for mult in (0.5, 1.0, 2.0):
                    eta = mult * step / norms[name]
                    th2 = forward({k: M[k] - eta * g_[k] for k in keys})
                    e_new, acts = evaluate(th2, ev_new, cache)
                    da = [float(np.sqrt(np.mean((x[:, :6] - y[:, :6]) ** 2))) for x, y in zip(acts, acts0)]
                    flips = [float(np.mean(np.sign(x[:, 6]) != np.sign(y[:, 6]))) for x, y in zip(acts, acts0)]
                    r = {"direction": name, "mult": mult, "new": summary(e_new), "dkl_new_mean": float(np.mean(np.subtract(e_new["kl"], base_new["kl"]))),
                         "frac_obs_kl_worse": float(np.mean(np.array(e_new["kl"]) > np.array(base_new["kl"]))), "action_change_rms_mean": float(np.mean(da)), "action_change_rms_max": float(np.max(da)),
                         "gripper_flip_rate": float(np.mean(flips))}
                    if ev_old:
                        e_old, _ = evaluate(th2, ev_old, cache)
                        r["old"] = summary(e_old)
                        r["dkl_old_mean"] = float(np.mean(np.subtract(e_old["kl"], base_old["kl"])))
                    rec["steps"].append(r)
                    del th2
                    log(f"A2 {run} u{u} {name} x{mult}: dKL_new {r['dkl_new_mean']:+.1f} grip {r['new']['grip_agree']:.2f} dA {r['action_change_rms_mean']:.4f} flips {r['gripper_flip_rate']:.2f}" + (f" dKL_old {r['dkl_old_mean']:+.2f}" if ev_old else ""))
            A2.append(rec)
            del G
            save("A2", A2)  # incremental

    # B + C1 + C2 --------------------------------------------------------
    order_t = list(TASKS)
    if not done("BC"):
        BC = {"students": {}}
        demo = {t: demo_obs(t) for t in order_t}
        for sname, M in (("task0_student", init), ("noreplay_t063_c400", load_master(f"{NOREP}/k1_t063/student/checkpoints/adapter_400.npz"))):
            theta = forward(M)
            S = {"tasks": {}}
            mean_full, mean_det, mean_clip = {}, {}, {}
            for tid in order_t:
                tt = teacher(tid)
                accF, accD, rows_ = zeros(), zeros(), []
                norms_ = []
                for it in demo[tid]:
                    p = prep(it["req"], tid)
                    (v, aux), gf = g_full(theta, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], it["x0"], it["eps"])
                    gf = flat(gf)
                    _, gd_ = g_det(theta, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], aux["chain"])
                    gd_ = flat(gd_)
                    nf, nd = sum(gsq(gf).values()) ** 0.5, sum(gsq(gd_).values()) ** 0.5
                    norms_.append(nf)
                    accF, accD = j_add(accF, gf, 1.0 / len(demo[tid])), j_add(accD, gd_, 1.0 / len(demo[tid]))
                    r = {"demo": it["demo"], "t": it["t"], "near_toggle": it["near_toggle"], "kl": float(v), "norm_full": nf, "norm_det": nd}
                    if sname == "task0_student":
                        at, as_ = ode(tt, p, it["x0"], True), ode(theta, p, it["x0"], False)
                        r["grip_agree"] = float(np.mean(np.sign(as_[:, 6]) == np.sign(at[:, 6])))
                        r["xyz"] = float(np.sqrt(np.mean((as_[:, :3] - at[:, :3]) ** 2)))
                        k_mid = K - 1
                        u0 = jnp.asarray(np.random.default_rng(1).standard_normal((1, 10, 32)), jnp.float32)
                        r["sigma_teacher_last_step"] = float(sig_t(tt, p["ob"], p["tm"], p["tk"], aux["chain"][:, k_mid], cfg.times()[k_mid], u0))
                        r["sigma_student_last_step"] = float(sig_s(theta, p["ob"], p["sm"], p["sk"], aux["chain"][:, k_mid], cfg.times()[k_mid], u0))
                    rows_.append(r)
                    del gf, gd_
                # clipped mean (k = 3 x the median of this task's trajectories), second pass over the clipped ones only
                med = float(np.median(norms_))
                accC = {k: (np.array(accF[k]) if SHARED else accF[k]) for k in keys}
                for idx, it in enumerate(demo[tid]):
                    if norms_[idx] > 3 * med:
                        p = prep(it["req"], tid)
                        (_, _), gf = g_full(theta, tt, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], it["x0"], it["eps"])
                        accC = j_add(accC, flat(gf), (3 * med / norms_[idx] - 1.0) / len(demo[tid]))
                        del gf
                mean_full[tid] = {k: np.asarray(accF[k]) for k in keys}
                mean_det[tid] = {k: np.asarray(accD[k]) for k in keys}
                mean_clip[tid] = {k: np.asarray(accC[k]) for k in keys}
                del accF, accD, accC
                nfa = np.array([r["norm_full"] for r in rows_])
                nda = np.array([r["norm_det"] for r in rows_])
                S["tasks"][tid] = {"rows": rows_, "kl_median": float(np.median([r["kl"] for r in rows_])), "norm_full_median": float(np.median(nfa)), "norm_full_max_over_median": float(nfa.max() / np.median(nfa)),
                                   "full_over_det_median": float(np.median(nfa / nda)), "full_over_det_max": float(np.max(nfa / nda)), "n_clipped_k3": int(np.sum(nfa > 3 * np.median(nfa))),
                                   "near_toggle_vs_not_norm_median": [float(np.median(nfa[[r["near_toggle"] for r in rows_]])) if any(r["near_toggle"] for r in rows_) else None,
                                                                      float(np.median(nfa[[not r["near_toggle"] for r in rows_]]))]}
                if sname == "task0_student":
                    S["tasks"][tid].update({"grip_agree_mean": float(np.mean([r["grip_agree"] for r in rows_])), "xyz_mean": float(np.mean([r["xyz"] for r in rows_])),
                                            "sigma_teacher_median": float(np.median([r["sigma_teacher_last_step"] for r in rows_])), "sigma_teacher_max": float(np.max([r["sigma_teacher_last_step"] for r in rows_]))})
                log(f"B {sname} {tid}: KL med {S['tasks'][tid]['kl_median']:.1f} |g| med {np.median(nfa):.0f} max/med {S['tasks'][tid]['norm_full_max_over_median']:.1f} full/det max {S['tasks'][tid]['full_over_det_max']:.1f}")
                if tid not in ("t035", "t063"):
                    drop_teacher(tid)

            def hcos(A, B_):
                num = sum(float(np.sum(A[k].astype(np.float64) * B_[k])) for k in keys)
                return num / (sum(float(np.sum(A[k].astype(np.float64) ** 2)) for k in keys) * sum(float(np.sum(B_[k].astype(np.float64) ** 2)) for k in keys)) ** 0.5

            for lab_, MM in (("full", mean_full), ("detached", mean_det), ("clipped_k3", mean_clip)):
                S[f"C1_cos_{lab_}"] = {f"{x}|{y}": hcos(MM[x], MM[y]) for i_, x in enumerate(order_t) for y in order_t[i_ + 1:]}
            if sname == "task0_student":
                # C2: one step of task i's clipped direction (logged Phase-2 median step and 4x), KL change on every task j
                step = float(np.median([float(r["master_step"]) for uu, r in history(RUNS["replay"]["student"]).items() if 101 <= uu <= 140]))
                C2 = {"step": step, "dkl": {}}
                thetas = {}
                for i in order_t:
                    nrm = sum(float(np.sum(mean_clip[i][k].astype(np.float64) ** 2)) for k in keys) ** 0.5
                    for mult in (1.0, 4.0):
                        thetas[(i, mult)] = {k: M[k] - (mult * step / nrm) * mean_clip[i][k] for k in keys}
                for j in order_t:
                    teacher(j)
                    cache = {}
                    base_j, _ = evaluate(theta, demo[j], cache)
                    for (i, mult), Mi in thetas.items():
                        e_j, _ = evaluate(forward(Mi), demo[j], cache)
                        C2["dkl"][f"{i}->{j}|x{mult}"] = {"dkl_mean": float(np.mean(np.subtract(e_j["kl"], base_j["kl"]))), "dkl_rel": float(np.mean(e_j["kl"]) / np.mean(base_j["kl"]) - 1),
                                                         "dgrip": float(np.mean(e_j["grip_agree"]) - np.mean(base_j["grip_agree"]))}
                    log(f"C2 evaluated on {j}")
                    if j not in ("t035", "t063"):
                        drop_teacher(j)
                S["C2"] = C2
                del thetas
            BC["students"][sname] = S
            del mean_full, mean_det, mean_clip
        save("BC", BC)

    # C3 ---------------------------------------------------------------
    if not done("C3"):
        from clvla.cl_opd.buffer import load_arrays
        arr = load_arrays(ROOT / f"{FULL}/buffer", "t035")
        rows_n = len(arr["state"])
        idx = np.linspace(0, rows_n - 1, 24).astype(int)
        items = []
        for n_, r in enumerate(idx):
            x0, eps = event_noise(OT.seed_ok(8_000_000 + n_), cfg)
            items.append({"req": {"observation/image": np.asarray(arr["image"][r]), "observation/wrist_image": np.asarray(arr["wrist_image"][r]), "observation/state": np.asarray(arr["state"][r]), "prompt": prompts["t035"]},
                          "x0": x0, "eps": eps, "tid": "t035"})
        cache, C3 = {}, []
        known = {("init", "task0 end"): 0.96, ("replay", "c400"): 1.00, ("noreplay", "c400"): 0.00}
        for run, lab in [("init", "task0 end")] + [(r, f"c{c}") for r in ("replay", "noreplay") for c in (100, 200, 300, 400)]:
            Mx = init if run == "init" else master(run, lab)
            e, _ = evaluate(forward(Mx), items, cache)
            C3.append({"run": run, "ckpt": lab, **summary(e), "heldout_t035_success": known.get((run, lab))})
            log(f"C3 {run} {lab}: KL {C3[-1]['kl']:.1f} (median {C3[-1]['kl_median']:.1f}) grip {C3[-1]['grip_agree']:.2f}")
            if run != "init":
                masters.pop((run, lab), None)
        save("C3", C3)
    log("[done]")


if __name__ == "__main__":
    main()
