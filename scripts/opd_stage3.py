"""Stage 3: exactly one teacher-only pathwise Flow-OPD update on a temporary FP32 master copy of the student, on stored Stage 2 observations. Pre-update gates stop before any update.
  CLVLA_USE_GPU=1 CLVLA_GPU=1 PYTHONPATH=scripts:tests/ogpo_port scripts/py-policy scripts/opd_stage3.py --stage2 D --teacher-delta T --out O
Step rule, fixed before the first run: clip the batch gradient to norm 1 (OpenPI recipe), then lr = h = 2% of batch KL / ||g||, capped at 1e-3 of the trainable parameter norm. SGD, momentum 0.9, so the first update is -h g/||g||.
Fresh-noise evaluation: same programme and noise before and after. The random-direction control is a parameter copy of the same step norm, not an optimizer update; it shows what BF16 rounding alone does to the KL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TAG", "opd_stage3")
import task10b as T  # sets the deterministic flags before JAX is imported

import jax
import jax.numpy as jnp
import numpy as np
import optax
from clvla.config import load_experiment_config
from clvla.ogpo import pi05_path_kl as pkl
from clvla.ogpo import pi05_pathwise_kl as pw
from clvla.ogpo import pi05_sde as sde
from clvla.ogpo import pi05_velocity as pv
from clvla.ogpo.pi05_sde_policy import event_noise, make_sampler, model_inputs, model_of, tree_sha
from clvla.policy import load_official_policy
from flax import nnx, traverse_util
from openpi.models import model as _model
from opd_stage2_verify import load_teacher

ROOT = Path(__file__).resolve().parents[1]
cfg = sde.SDEConfig()
EVENTS = tuple(int(round(x)) for x in np.linspace(0, 79, 8))
M, N_EVAL, TRAIN_SEED, EVAL_SEED, TARGET, CAP, MOMENTUM, CLIP = 4, 16, 60000, 62000, 0.02, 1e-3, 0.9, 1.0
FORBIDDEN = set(range(12000, 12008)) | set(range(13000, 13008)) | {15000}
STUDENT_SHA, TEACHER_SHA, DELTA_SHA = "84d98b1be89ab221c407e84146fa001959de277de7e8481c76786afa0ffade5f", "ed4a99f23d0352cb037eecdfa16ca51505d6c12cb031365a1e45850281ed7763", "d0d689ac669c94d8420c80b0149fa9901de2b0743d173f738c8f28be76193500"
w = np.asarray(pkl.kl_coefficients(cfg)["weights"], np.float64)
MASKS = {}
for name, (r, c) in pkl.GROUPS.items():
    m = np.zeros((10, 32), np.float32)
    m[r, c] = 1.0
    MASKS[name] = m
FULL = np.ones((10, 32), np.float32)
R = {}


def save(out):
    (out / "report.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))


def err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(np.abs(a - b).max()), float(np.linalg.norm(a - b) / np.linalg.norm(b))


def stats(a):
    a = np.asarray(a, np.float64)
    return {"mean": float(a.mean()), "se": float(a.std(ddof=1) / len(a) ** 0.5), "n": len(a)}


def kl_from_vel(vs, vt):
    d = (np.asarray(vs, np.float64) - np.asarray(vt, np.float64)) ** 2
    return {"full": float(np.sum(w * np.sum(d, axis=(2, 3)))), "per_step": (w * np.sum(d, axis=(2, 3)))[0].tolist(), **{n: float(np.sum(w * np.sum(d * m, axis=(2, 3)))) for n, m in MASKS.items()}}


def paired(before, after, B, Ne):
    """t interval over the 8 observation means (2.365 for 7 d.f.), the conservative unit; the pair-level standard error is also reported."""
    d = np.asarray(after, np.float64) - np.asarray(before, np.float64)
    obs = d.reshape(B, Ne).mean(axis=1)
    se = float(obs.std(ddof=1) / B**0.5)
    return {"before": float(np.mean(before)), "after": float(np.mean(after)), "diff": float(d.mean()), "relative": float(d.mean() / np.mean(before)), "se_obs_level": se, "ci95_obs_level": [float(d.mean() - 2.365 * se), float(d.mean() + 2.365 * se)], "se_pair_level": float(d.std(ddof=1) / d.size**0.5),
            "obs_means": obs.tolist(), "obs_decreased": int(np.sum(obs < 0)), "pairs_decreased": int(np.sum(d < 0)), "pairs": d.size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage2", required=True)
    ap.add_argument("--teacher-delta", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t_all = time.perf_counter()
    stage2, delta_dir, out = Path(a.stage2), Path(a.teacher_delta), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    H = T.helpers()
    flat = H["_flat"]
    R["xla_flags"] = os.environ["XLA_FLAGS"]
    conf = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    student = model_of(policy)
    report2 = json.loads((stage2 / "report.json").read_text())
    env2 = report2["environment"]
    assert env2["reset"]["demo"] == 0 and env2["reset"]["profile"] == "n25" and env2["libero_90_index"] == 35 and env2["env_seed"] == 7
    teacher, trec = load_teacher(student, delta_dir)
    gd_s0, full_s = nnx.split(student)
    gd_t, t_state = nnx.split(teacher)
    s_sha, t_sha = tree_sha(full_s), tree_sha(t_state)
    R["integrity"] = {"stage2_dir": str(stage2), "student_sha": s_sha, "teacher_sha": t_sha, "teacher": trec, "matches_stage2": s_sha == STUDENT_SHA == report2["student_sha"]["start"] and t_sha == TEACHER_SHA == report2["teacher"]["state_sha_after"] and trec["delta_npz_sha256"] == DELTA_SHA}
    assert R["integrity"]["matches_stage2"], "checkpoint identity differs from Stage 2"
    assert trec["leaves_verified"] == 71
    gd_s, theta, vlm_s, other_s = nnx.split(student, H["in_groups"](H["TRAINABLE"]), H["in_groups"](H["VLM"]), ...)
    from test_pi05_pathwise_gradient import group_of

    run_fn, _ = make_sampler(student, cfg)

    def prefix(gd, st, obs):
        c = pv.build_prefix(nnx.merge(gd, st), obs)
        return c.observation, c.prefix_mask, c.kv_cache

    jp_s, jp_t = jax.jit(lambda st, o: prefix(gd_s0, st, o)), jax.jit(lambda st, o: prefix(gd_t, st, o))

    def path_eval(th, vl, tt, ob, sm, sk, tm, tk, chain):
        o = pkl.path_kl(nnx.merge(gd_t, tt), pv.PrefixCache(ob, tm, tk), nnx.merge(gd_s, th, vl, other_s), pv.PrefixCache(ob, sm, sk), chain, cfg)
        return {k: o[k] for k in ("velocities_student", "velocities_teacher", "means_student", "path_kl_velocity", "kl_groups", "kl_velocity")}

    def sample_eval(th, vl, tt, ob, sm, sk, tm, tk, x0, eps):
        s = sde.sde_sample(nnx.merge(gd_s, th, vl, other_s), pv.PrefixCache(ob, sm, sk), x0, eps, cfg)
        return s["chain"], path_eval(th, vl, tt, ob, sm, sk, tm, tk, s["chain"])

    j_path, j_sample = jax.jit(path_eval), jax.jit(sample_eval)

    def obj(th, vl, tt, ob, sm, sk, tm, tk, x0, eps, mask):
        o = pw.pathwise_kl(nnx.merge(gd_s, th, vl, other_s), pv.PrefixCache(ob, sm, sk), pw.frozen_view(nnx.merge(gd_t, tt)), pv.PrefixCache(ob, tm, tk), x0, eps, cfg, mode="full")
        kl = jnp.asarray(w, jnp.float32) * jnp.sum(mask * (o["velocities_student"] - o["velocities_teacher"]) ** 2, axis=(2, 3))
        return jnp.sum(kl), {"chain": o["chain"], "vs": o["velocities_student"], "vt": o["velocities_teacher"]}

    def obj_det(th, vl, tt, ob, sm, sk, tm, tk, stored):
        o = pw.pathwise_kl(nnx.merge(gd_s, th, vl, other_s), pv.PrefixCache(ob, sm, sk), pw.frozen_view(nnx.merge(gd_t, tt)), pv.PrefixCache(ob, tm, tk), None, None, cfg, mode="detached", stored_chain=stored)
        return jnp.sum(o["path_kl"])

    grad_full, grad_det = jax.jit(jax.value_and_grad(obj, has_aux=True)), jax.jit(jax.grad(obj_det))
    keys = sorted(flat(theta))
    host = lambda g: {k: np.asarray(v).astype(np.float64) for k, v in flat(g).items()}  # noqa: E731
    dot = lambda x, y: sum(float(np.sum(x[k] * y[k])) for k in keys)  # noqa: E731
    norm = lambda x: dot(x, x) ** 0.5  # noqa: E731

    def group_norms(x):
        g = {}
        for k in keys:
            g[group_of(k)] = g.get(group_of(k), 0.0) + float(np.sum(x[k] ** 2))
        return {n: v**0.5 for n, v in g.items()}

    # Part A: batch
    batch = []
    for ev in EVENTS:
        e = dict(np.load(stage2 / f"event_{ev:04d}.npz"))
        request = {"observation/image": e["observation_image"], "observation/wrist_image": e["observation_wrist_image"], "observation/state": e["observation_state"], "prompt": str(e["prompt"])}
        obs, inputs = model_inputs(policy, request)
        ob, sm, sk = jp_s(full_s, obs)
        ob_t, tm, tk = jp_t(t_state, obs)
        assert tree_sha(ob) == tree_sha(ob_t), "student and teacher preprocessed observations differ"
        batch.append({"event": ev, "env_step": 10 + 5 * ev, "raw_obs": obs, "ob": ob, "sm": sm, "sk": sk, "tm": tm, "tk": tk, "e": e, "state": np.asarray(inputs["state"]), "caches_differ": not all(bool(jnp.array_equal(x, y)) for x, y in zip(jax.tree.leaves(sk), jax.tree.leaves(tk)))})
    assert all(b["caches_differ"] for b in batch)
    assert not all(np.array_equal(batch[0]["e"]["observation_image"], b["e"]["observation_image"]) for b in batch[1:])
    R["batch"] = {"events": list(EVENTS), "env_steps": [b["env_step"] for b in batch], "observations": len(batch), "trajectories_per_observation": M, "trajectories_total": len(batch) * M, "recorded_trajectories": len(batch), "extra_noise_seeds": f"{TRAIN_SEED}+", "temporal_note": "one episode, about 55 environment steps apart: temporally correlated", "prompt": str(batch[0]["e"]["prompt"]), "reset": env2["reset"], "norm_stats_sha256": hashlib.sha256((Path(conf.checkpoint.local_dir) / "assets" / conf.checkpoint.norm_stats_asset_id / "norm_stats.json").read_bytes()).hexdigest(), "separate_prefix_caches": True}
    save(out)

    def noise(b_i, m):
        b = batch[b_i]
        if m == 0:
            return b["e"]["x0"], b["e"]["eps"]
        s = TRAIN_SEED + 100 * b_i + m
        assert s not in FORBIDDEN
        return event_noise(s, cfg)

    # Parts B, D, E, F: every trajectory through both programmes; the batch gradient accumulated over microbatches of one
    g_sum, g_rec, g_det = ({k: 0.0 for k in keys} for _ in range(3))
    rows, recorded_bitwise, per_obs, s_rms, t_rms, shapes = [], True, np.zeros(len(batch)), [], [], set()
    t0 = time.perf_counter()
    args = (theta, vlm_s, t_state)
    for i, b in enumerate(batch):
        I = (b["ob"], b["sm"], b["sk"], b["tm"], b["tk"])
        for m in range(M):
            x0, eps = noise(i, m)
            x0, eps = jnp.asarray(x0), jnp.asarray(eps)
            ro = jax.tree.map(np.asarray, run_fn(b["raw_obs"], x0, eps))
            if m == 0:
                recorded_bitwise = recorded_bitwise and np.array_equal(ro["chain"], b["e"]["chain"]) and np.array_equal(ro["means"], b["e"]["means"])
            pe = jax.tree.map(np.asarray, j_path(*args, *I, jnp.asarray(ro["chain"])))
            (val, aux), g = grad_full(*args, *I, x0, eps, FULL)
            aux = jax.tree.map(np.asarray, aux)
            gh = host(g)
            del g
            finite = all(bool(np.isfinite(gh[k]).all()) for k in keys) and np.isfinite(float(val))
            gd_ = host(grad_det(*args, *I, jnp.asarray(aux["chain"])))
            for k in keys:
                g_sum[k] = g_sum[k] + gh[k]
                g_det[k] = g_det[k] + gd_[k]
                if m == 0:
                    g_rec[k] = g_rec[k] + gh[k]
            del gd_, gh
            x, vs = aux["chain"][:, :-1], aux["vs"]
            s = np.asarray(cfg.times(), np.float32)[None, :, None, None]
            mu_g = x - (1.0 / cfg.num_steps) * (vs + 0.5 * cfg.sigma**2 * (x + (1.0 - s) * vs))
            kl_r, kl_g = kl_from_vel(pe["velocities_student"], pe["velocities_teacher"]), kl_from_vel(aux["vs"], aux["vt"])
            shapes.add((tuple(aux["vs"].shape), tuple(aux["vt"].shape)))
            s_rms.append(float(np.sqrt(np.mean(aux["vs"] ** 2))))
            t_rms.append(float(np.sqrt(np.mean(aux["vt"] ** 2))))
            per_obs[i] += kl_g["full"] / M
            rows.append({"event": b["event"], "m": m, "finite": bool(finite), "objective_fp32": float(val), "chain": err(aux["chain"], ro["chain"]), "final": err(aux["chain"][0, -1], ro["chain"][0, -1]), "means": err(mu_g, ro["means"]), "stds_equal": bool(np.array_equal(ro["stds"], np.asarray(cfg.stds(), np.float32))), "rollout": kl_r, "gradient": kl_g, "same_chain_program_gap": abs(kl_r["full"] - kl_g["full"])})
        print(f"[batch] event {b['event']} {time.perf_counter() - t0:.0f} s", flush=True)
    M_all = len(batch) * M
    g = {k: g_sum[k] / M_all for k in keys}
    g_d = {k: g_det[k] / M_all for k in keys}
    g_s = {k: g[k] - g_d[k] for k in keys}
    rec = [r for r in rows if r["m"] == 0]

    def gap(rs, key):
        a, c = np.mean([r["rollout"][key] for r in rs]), np.mean([r["gradient"][key] for r in rs])
        return {"rollout": float(a), "gradient": float(c), "abs": float(abs(a - c)), "relative": float(abs(a - c) / a)}

    R["chain_consistency"] = {"recorded_rollout_replay_bitwise": bool(recorded_bitwise), "max_abs_chain": max(r["chain"][0] for r in rows), "max_rel_chain": max(r["chain"][1] for r in rows), "max_abs_final": max(r["final"][0] for r in rows), "max_rel_final": max(r["final"][1] for r in rows), "max_abs_means": max(r["means"][0] for r in rows), "max_rel_means": max(r["means"][1] for r in rows), "stds_equal": all(r["stds_equal"] for r in rows),
                              "kl_recorded": gap(rec, "full"), "kl_all": gap(rows, "full"), "executed_recorded": gap(rec, "executed"), "executed_all": gap(rows, "executed"), "largest_single_trajectory_relative": max(abs(r["rollout"]["full"] - r["gradient"]["full"]) / r["rollout"]["full"] for r in rows)}
    gate_chain = R["chain_consistency"]["kl_recorded"]["relative"] < 0.01 and R["chain_consistency"]["kl_all"]["relative"] < 0.01
    R["chain_consistency"]["gate_below_1pct"] = bool(gate_chain)

    kl = np.array([r["gradient"]["full"] for r in rows])
    R["initial_kl"] = {"mean": float(kl.mean()), "se_pair": float(kl.std(ddof=1) / len(kl) ** 0.5), "per_flow_step": np.mean([r["gradient"]["per_step"] for r in rows], 0).tolist(), "per_observation": per_obs.tolist(), "velocity_rms_student": float(np.mean(s_rms)), "velocity_rms_teacher": float(np.mean(t_rms)), "velocity_shapes": [str(x) for x in shapes]}
    R["coordinates"] = {n: {"coords": int(MASKS[n].sum()), "kl": float(np.mean([r["gradient"][n] for r in rows])), "share": float(np.mean([r["gradient"][n] for r in rows]) / kl.mean()), "kl_per_flow_step": float(np.mean([r["gradient"][n] for r in rows]) / cfg.num_steps)} for n in MASKS}
    print("[gate]", json.dumps(R["chain_consistency"]), flush=True)
    save(out)

    # Coordinate-group gradients on the 8 recorded trajectories (same programme, mask changed)
    g_groups = {}
    for n, mask in MASKS.items():
        acc = {k: 0.0 for k in keys}
        for i, b in enumerate(batch):
            x0, eps = noise(i, 0)
            (_, _), gg = grad_full(*args, b["ob"], b["sm"], b["sk"], b["tm"], b["tk"], jnp.asarray(x0), jnp.asarray(eps), mask)
            gh = host(gg)
            del gg
            for k in keys:
                acc[k] = acc[k] + gh[k]
        g_groups[n] = {k: acc[k] / len(batch) for k in keys}
    grec = {k: g_rec[k] / len(batch) for k in keys}
    sum3 = {k: sum(g_groups[n][k] for n in MASKS) for k in keys}
    res = {k: sum3[k] - grec[k] for k in keys}
    for n in MASKS:
        R["coordinates"][n].update({"gradient_norm_recorded_8": norm(g_groups[n]), "cos_with_recorded_gradient": dot(g_groups[n], grec) / (norm(g_groups[n]) * norm(grec))})
    R["coordinate_gradient_sum_rel_error"] = norm(res) / norm(grec)
    del sum3, res, g_groups
    save(out)

    # Part G: routing, one group per derivative
    b0 = batch[0]
    x0, eps = noise(0, 0)
    routing = {}
    for name, argn, tree in (("student_vlm", 1, vlm_s), ("teacher_all", 2, t_state)):
        try:
            gz = jax.jit(jax.grad(lambda *q: obj(*q)[0], argnums=argn))(theta, vlm_s, t_state, b0["ob"], b0["sm"], b0["sk"], b0["tm"], b0["tk"], jnp.asarray(x0), jnp.asarray(eps), FULL)
            lv = [v for v in jax.tree.leaves(gz) if jnp.issubdtype(v.dtype, jnp.floating)]
            routing[name] = {"nonzero": int(sum(int(jnp.count_nonzero(v)) for v in lv)), "size": int(sum(v.size for v in lv)), "finite": all(bool(jnp.isfinite(v.astype(jnp.float32)).all()) for v in lv)}
            del gz, lv
        except Exception as err_:  # noqa: BLE001 - reported, and it stops the update below
            routing[name] = {"error": f"{type(err_).__name__}: {str(err_)[:200]}"}
        jax.clear_caches()
    frozen_t = pw.frozen_view(teacher)
    tan = jax.jvp(lambda y: pv.velocity(frozen_t, pv.PrefixCache(b0["ob"], b0["tm"], b0["tk"]), y, cfg.times()[5]), (jnp.asarray(b0["e"]["chain"][:, 5]),), (jnp.asarray(np.random.default_rng(60999).standard_normal((1, 10, 32)), jnp.float32),))[1]
    routing["teacher_input_derivative"] = {"finite": bool(jnp.isfinite(tan).all()), "norm": float(jnp.linalg.norm(tan))}
    gn, gdn, gsn = norm(g), norm(g_d), norm(g_s)
    gr = group_norms(g)
    routing.update({"gradient_norm": gn, "group_norms": gr, "direct_norm": gdn, "state_norm": gsn, "state_over_full": gsn / gn, "cos_full_direct": dot(g, g_d) / (gn * gdn), "cos_full_state": dot(g, g_s) / (gn * gsn), "state_group_norms": group_norms(g_s), "all_finite": all(r["finite"] for r in rows)})
    th_norm = {}
    for k in keys:
        th_norm[group_of(k)] = th_norm.get(group_of(k), 0.0) + float(np.sum(np.asarray(flat(theta)[k]).astype(np.float64) ** 2))
    routing["bf16_parameter_norms"] = {n: v**0.5 for n, v in th_norm.items()}
    R["routing"] = routing
    print("[routing]", json.dumps(routing), flush=True)
    save(out)
    gates = {"chain_kl_gap_below_1pct": bool(gate_chain), "recorded_replay_bitwise": bool(recorded_bitwise), "gradients_finite": routing["all_finite"], "expert_nonzero": gr.get("action_expert", 0) > 0, "heads_nonzero": gr.get("action_heads", 0) > 0, "student_vlm_zero": routing["student_vlm"].get("nonzero") == 0 and routing["student_vlm"].get("finite") is True, "teacher_zero": routing["teacher_all"].get("nonzero") == 0 and routing["teacher_all"].get("finite") is True, "teacher_input_derivative": routing["teacher_input_derivative"]["finite"] and routing["teacher_input_derivative"]["norm"] > 0, "state_connected": gsn > 0 and np.isfinite(gsn), "velocity_shapes": R["initial_kl"]["velocity_shapes"] == [str(((1, 10, 10, 32), (1, 10, 10, 32)))]}
    R["pre_update_gates"] = gates
    save(out)
    if not all(gates.values()):
        R["update"] = {"performed": False, "reason": [k for k, v in gates.items() if not v]}
        save(out)
        print("[stop] pre-update gate failed", R["update"], flush=True)
        return

    # Part H: FP32 master from the checkpoint's own float32 values, never from the BF16 cast
    base = traverse_util.flatten_dict(_model.restore_params(Path(conf.checkpoint.local_dir) / "params", restore_type=np.ndarray), sep="/")
    master = {k: np.asarray(base[k.removesuffix("/value")] if k.removesuffix("/value") in base else base[k]) for k in keys}
    assert all(v.dtype == np.float32 for v in master.values())
    th_flat = flat(theta)
    assert all(np.array_equal(np.asarray(jnp.asarray(master[k]).astype(th_flat[k].dtype)), np.asarray(th_flat[k])) for k in keys), "master does not cast to the forward parameters"
    del base
    jax.clear_caches()
    pn = sum(float(np.sum(v.astype(np.float64) ** 2)) for v in master.values()) ** 0.5
    h = min(TARGET * float(kl.mean()) / gn, CAP * pn)
    R["optimizer"] = {"type": "optax chain(clip_by_global_norm, sgd momentum)", "clip": CLIP, "lr": h, "momentum": MOMENTUM, "rule": f"h = min({TARGET} KL / ||g||, {CAP} ||theta||)", "fp32_master_norm": pn, "predicted_first_order_delta_kl": -h * gn, "predicted_relative_update": h / pn, "master_precision": "float32 from the checkpoint", "optimizer_state_precision": "float32"}
    save(out)
    print("[optimizer]", json.dumps(R["optimizer"]), flush=True)

    # Exactly one update
    opt = optax.chain(optax.clip_by_global_norm(CLIP), optax.sgd(h, momentum=MOMENTUM))
    cpu = jax.devices("cpu")[0]
    mast = {k: jax.device_put(master[k], cpu) for k in keys}
    grads32 = {k: jax.device_put(np.asarray(g[k], np.float32), cpu) for k in keys}
    with jax.default_device(cpu):
        state = opt.init(mast)
        updates, state = opt.update(grads32, state, mast)
        new_master = optax.apply_updates(mast, updates)
    updates_np = {k: np.asarray(updates[k]) for k in keys}
    new_np = {k: np.asarray(new_master[k]) for k in keys}
    state_finite = all(bool(jnp.isfinite(x).all()) for x in jax.tree.leaves(state))
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat(theta).items()}
    order = [by_id[id(l)] for l in leaves]
    th_new = jax.tree.unflatten(treedef, [jnp.asarray(new_np[k]).astype(l.dtype) for k, l in zip(order, leaves)])
    tn = flat(th_new)
    fw_new = {k: np.asarray(tn[k]).astype(np.float64) for k in keys}
    fw_old = {k: np.asarray(th_flat[k]).astype(np.float64) for k in keys}
    eff = {k: fw_new[k] - fw_old[k] for k in keys}
    intended = {k: new_np[k].astype(np.float64) - master[k].astype(np.float64) for k in keys}
    changed = {}
    for k in keys:
        c = changed.setdefault(group_of(k), {"master": 0, "forward": 0, "size": 0})
        c["master"] += int(np.count_nonzero(new_np[k] != master[k]))
        c["forward"] += int(np.count_nonzero(fw_new[k] != fw_old[k]))
        c["size"] += int(new_np[k].size)
    R["update"] = {"performed": True, "count": 1, "optimizer_state_finite": state_finite, "master_step_norm": norm(intended), "forward_step_norm": norm(eff), "cos_forward_with_master_step": dot(eff, intended) / (norm(eff) * norm(intended)), "cos_forward_with_negative_gradient": -dot(eff, g) / (norm(eff) * gn), "changed_by_group": changed, "master_norm_after": sum(float(np.sum(v.astype(np.float64) ** 2)) for v in new_np.values()) ** 0.5, "forward_first_order_delta_kl": dot(eff, g)}
    tmp = out / "temporary_update"
    tmp.mkdir(exist_ok=True)
    np.savez(tmp / "master_after_one_step.npz", **{k.replace("/", "__"): new_np[k] for k in keys})
    z = np.load(tmp / "master_after_one_step.npz")
    R["update"]["temporary_checkpoint"] = {"path": str(tmp / "master_after_one_step.npz"), "round_trip_bitwise": all(np.array_equal(z[k.replace("/", "__")], new_np[k]) for k in keys)}
    z.close()
    save(out)
    print("[update]", json.dumps(R["update"]), flush=True)

    # Part I: fresh paired noise. The control has the same forward step norm in a random direction.
    r = np.random.default_rng(62999)
    ctrl = {k: r.standard_normal(new_np[k].shape).astype(np.float32) for k in keys}
    cn = sum(float(np.sum(v.astype(np.float64) ** 2)) for v in ctrl.values()) ** 0.5
    ctrl_step = R["update"]["master_step_norm"] / cn
    th_ctrl = jax.tree.unflatten(treedef, [jnp.asarray(master[k] + np.float32(ctrl_step) * ctrl[k]).astype(l.dtype) for k, l in zip(order, leaves)])
    del ctrl
    jax.clear_caches()
    j_sample = jax.jit(sample_eval)

    def evaluate(th):
        out_, phys = [], []
        for i, b in enumerate(batch):
            for m in range(N_EVAL):
                s = EVAL_SEED + 100 * i + m
                assert s not in FORBIDDEN
                x0, eps = event_noise(s, cfg)
                chain, o = j_sample(th, vlm_s, t_state, b["ob"], b["sm"], b["sk"], b["tm"], b["tk"], jnp.asarray(x0), jnp.asarray(eps))
                kg = np.asarray(o["kl_groups"])[0]
                out_.append([float(o["path_kl_velocity"][0]), kg[:, 0].sum(), kg[:, 1].sum(), kg[:, 2].sum()])
                phys.append(np.asarray(policy._output_transform({"state": b["state"], "actions": np.asarray(chain)[0, cfg.num_steps]})["actions"]))
        return np.array(out_, np.float64), np.stack(phys)

    t0 = time.perf_counter()
    before, _ = evaluate(theta)
    repeat, _ = evaluate(theta)
    after, phys = evaluate(th_new)
    control, _ = evaluate(th_ctrl)
    B = len(batch)
    R["evaluation"] = {"seconds": time.perf_counter() - t0, "pairs": B * N_EVAL, "per_observation": N_EVAL, "noise_seed": f"{EVAL_SEED}+", "repeat_same_parameters_bitwise": bool(np.array_equal(before, repeat)), "full": paired(before[:, 0], after[:, 0], B, N_EVAL), "control_random_direction_full": paired(before[:, 0], control[:, 0], B, N_EVAL)}
    for j, n in enumerate(MASKS, 1):
        R["evaluation"][n] = paired(before[:, j], after[:, j], B, N_EVAL)
    pa = phys[:, :5]
    R["actions_after_update"] = {"shape": list(phys.shape), "finite": bool(np.isfinite(phys).all()), "arm_abs_max_executed": float(np.abs(pa[..., :6]).max()), "outside_arm_range_fraction": float(np.mean(np.abs(pa[..., :6]) > 1)), "gripper_min": float(pa[..., 6].min()), "gripper_max": float(pa[..., 6].max())}
    f = R["evaluation"]["full"]
    R["chain_gap_vs_eval_se"] = {"abs_gap_all": R["chain_consistency"]["kl_all"]["abs"], "eval_se_obs_level": f["se_obs_level"], "eval_se_pair_level": f["se_pair_level"], "abs_diff": abs(f["diff"])}
    th_after = flat(theta)
    R["immutability"] = {"student_forward_unchanged": tree_sha(nnx.split(student)[1]) == s_sha, "teacher_unchanged": tree_sha(nnx.split(teacher)[1]) == t_sha, "vlm_unchanged_in_after_model": H["sha"](vlm_s) == R.get("vlm_sha", H["sha"](vlm_s)), "theta_arrays_unchanged": all(np.array_equal(np.asarray(th_after[k]), fw_old[k].astype(np.asarray(th_after[k]).dtype)) for k in keys)}
    R["resources"] = {"peak_gpu_gib": jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30, "rss_gib": H["rss_gib"](), "seconds": time.perf_counter() - t_all}
    decreased = f["ci95_obs_level"][1] < 0
    R["decision_inputs"] = {"ci_excludes_zero_below": bool(decreased), "beyond_control": bool(f["after"] < R["evaluation"]["control_random_direction_full"]["after"])}
    save(out)
    print("[evaluation]", json.dumps(R["evaluation"]), flush=True)
    print("[end]", json.dumps({k: R[k] for k in ("actions_after_update", "immutability", "resources", "decision_inputs", "chain_gap_vs_eval_se")}), flush=True)


if __name__ == "__main__":
    main()
