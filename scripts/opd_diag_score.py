"""t35 diagnostics, Experiments 2a and 3, no optimizer. Stored observations from the Experiment 1 rollouts: set (b) the teacher's, set (a) the 500-update student's.
2a: every set (b) observation; ODE actions of teacher, student checkpoint 500 and checkpoint 0 (the base function), from the noise the teacher used there, unnormalized by the policy's own output transform.
3: fixed subsets of 240 per set (chosen by the preregistered rule, seed 0); for every checkpoint, the training objective split by coordinate group, action-dimension group and denoising step on the student's own SDE chain, and the ODE action error to the teacher.
  CLVLA_USE_GPU=1 PYTHONPATH=scripts scripts/py-policy scripts/opd_diag_score.py --diag DIR --teacher-delta DIR --checkpoints DIR [--limit N]"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"

import jax
import jax.numpy as jnp
import numpy as np
from clvla.config import load_experiment_config
from clvla.ogpo import pi05_path_kl as pkl, pi05_sde as sde, pi05_velocity as pv
from clvla.ogpo.pi05_sde_policy import event_noise, model_inputs, model_of
from clvla.policy import load_official_policy
from flax import nnx
from opd_lora_student import build_student_model, load_delta_teacher

ROOT = Path(__file__).resolve().parents[1]
cfg = sde.SDEConfig()
SCORE_SEED0, WAIT, PROMPT = 800000, 10, "open the microwave"
w = np.asarray(pkl.kl_coefficients(cfg)["weights"], np.float64)
MASKS = {}
for name, (r_, c_) in {**pkl.GROUPS, "exec_xyz": (slice(0, 5), slice(0, 3)), "exec_rot": (slice(0, 5), slice(3, 6)), "exec_grip": (slice(0, 5), slice(6, 7))}.items():
    m_ = np.zeros((10, 32))
    m_[r_, c_] = 1.0
    MASKS[name] = m_


def phase(snap):
    """Preregistered: opening once the door is at or below -0.05 rad, else near when the grip site is within 0.12 m of the handle, else approach."""
    return "opening" if snap[6] <= -0.05 else ("near" if snap[7] <= 0.12 else "approach")


def kl_parts(vs, vt):
    d = (np.asarray(vs, np.float64) - np.asarray(vt, np.float64)) ** 2
    return {"full": float(np.sum(w * np.sum(d, axis=(2, 3)))), "per_step": (w * np.sum(d, axis=(2, 3)))[0].tolist(), **{n: float(np.sum(w * np.sum(d * m, axis=(2, 3)))) for n, m in MASKS.items()}}


def act_err(a, t):
    """Executed rows 0-4 in physical units: arm RMSE over xyz and rotation, gripper sign agreement."""
    a, t = a[:5], t[:5]
    return {"rmse_all": float(np.sqrt(np.mean((a - t) ** 2))), "rmse_xyz": float(np.sqrt(np.mean((a[:, :3] - t[:, :3]) ** 2))), "rmse_rot": float(np.sqrt(np.mean((a[:, 3:6] - t[:, 3:6]) ** 2))), "grip_sign_agree": float(np.mean(np.sign(a[:, 6]) == np.sign(t[:, 6])))}


def load_set(d, policy_id):
    obs = []
    for l in (d / "episodes.jsonl").read_text().splitlines():
        e = json.loads(l)
        z = np.load(d / ("ep_%02d.npz" % e["k"]))
        for i in range(len(z["ev_t_env"])):
            obs.append({"set_policy": policy_id, "noise_seed": e["noise_seed"], "k": e["k"], "event": i, "t_exec": int(z["ev_t_env"][i]) - WAIT, "episode_success": e["success"], "phase": phase(z["ev_snap"][i]), "snap": z["ev_snap"][i], "chunk": z["ev_chunk"][i], "image": z["ev_image"][i], "wrist": z["ev_wrist"][i], "state": z["ev_state"][i]})
    return obs


def subset(obs, n, stratified):
    """Preregistered: seed 0; set (b) up to n/3 per phase with the remainder filled at random, set (a) uniform at random."""
    r = np.random.default_rng(0)
    if not stratified:
        return sorted(r.choice(len(obs), min(n, len(obs)), replace=False).tolist())
    pick = []
    for p in ("approach", "near", "opening"):
        ids = [i for i, o in enumerate(obs) if o["phase"] == p]
        pick += r.choice(ids, min(n // 3, len(ids)), replace=False).tolist()
    rest = [i for i in range(len(obs)) if i not in set(pick)]
    pick += r.choice(rest, min(n - len(pick), len(rest)), replace=False).tolist()
    return sorted(pick)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag", required=True)
    ap.add_argument("--teacher-delta", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--subset", type=int, default=240)
    ap.add_argument("--limit", type=int, default=None, help="smoke test only")
    ap.add_argument("--ckpts", default="0,25,50,75,100,200,300,400,500")
    ap.add_argument("--teacher-set", default="teacher10d", help="exp1/<name>: set (b)")
    ap.add_argument("--student-set", default="student500", help="exp1/<name>: set (a)")
    ap.add_argument("--compare-ckpt", type=int, default=500, help="checkpoint compared with checkpoint 0 in Experiment 2a")
    ap.add_argument("--out-name", default="scores")
    a = ap.parse_args()
    diag = Path(a.diag)
    out = diag / a.out_name
    CKPTS = tuple(int(c) for c in a.ckpts.split(","))
    out.mkdir(parents=True, exist_ok=True)
    conf = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    base = model_of(policy)
    teacher, _, _ = load_delta_teacher(base, Path(a.teacher_delta))
    gd_t, st_t = nnx.split(teacher)
    students = {}
    for c in CKPTS:
        z = np.load(Path(a.checkpoints) / ("adapter_%03d.npz" % c))
        students[c] = nnx.split(build_student_model(base, {k.replace("__", "/"): z[k] for k in z.files}))
    gd_s = students[CKPTS[0]][0]
    st_s = {c: s for c, (_, s) in students.items()}

    def prefix_fn(gd):
        def f(st, ob):
            pc = pv.build_prefix(nnx.merge(gd, st), ob)
            return pc.observation, pc.prefix_mask, pc.kv_cache
        return jax.jit(f)

    def vel_fn(gd):
        return jax.jit(lambda st, ob, mk, kv, x, s: pv.velocity(nnx.merge(gd, st), pv.PrefixCache(ob, mk, kv), x, s))

    pre_s, pre_t, vel_s, vel_t = prefix_fn(gd_s), prefix_fn(gd_t), vel_fn(gd_s), vel_fn(gd_t)
    sample_s = jax.jit(lambda st, ob, mk, kv, x0, eps: sde.sde_sample(nnx.merge(gd_s, st), pv.PrefixCache(ob, mk, kv), x0, eps, cfg)["chain"])
    times = [jnp.float32(s) for s in cfg.times()]

    def ode(vel, st, c, x0, inputs):
        y = jnp.asarray(x0)
        for s in times:
            y = y - (1.0 / cfg.num_steps) * vel(st, *c, y, s)
        return np.asarray(policy._output_transform({"state": np.asarray(inputs["state"]), "actions": np.asarray(y)[0]})["actions"], np.float64)

    def prepare(o):
        obs, inputs = model_inputs(policy, {"observation/image": o["image"], "observation/wrist_image": o["wrist"], "observation/state": o["state"], "prompt": PROMPT})
        return inputs, pre_s(st_s[0], obs), pre_t(st_t, obs), obs

    teacher_obs = load_set(diag / "exp1" / a.teacher_set, a.teacher_set)
    student_obs = load_set(diag / "exp1" / a.student_set, a.student_set)
    if a.limit:
        teacher_obs, student_obs = teacher_obs[: a.limit], student_obs[: a.limit]
    meta = lambda o: {k: o[k] for k in ("set_policy", "k", "event", "t_exec", "episode_success", "phase")}

    # Experiment 2a: every teacher observation, the teacher's own rollout noise.
    rows, rec_dev, cache_same = [], [], []
    for i, o in enumerate(teacher_obs):
        inputs, cs, ct, obs_ = prepare(o)
        if i < 3:
            cache_same.append(all(bool(jnp.array_equal(x, y)) for x, y in zip(jax.tree.leaves(pre_s(st_s[a.compare_ckpt], obs_)), jax.tree.leaves(cs))))
        x0, _ = event_noise(o["noise_seed"] + o["event"], cfg)
        at = ode(vel_t, st_t, ct, x0, inputs)
        rec_dev.append(float(np.abs(at - o["chunk"]).max()))
        r = {**meta(o), "teacher_vs_recorded_max_abs": rec_dev[-1]}
        for c in (0, a.compare_ckpt):
            r.update({"ckpt%d_%s" % (c, k): v for k, v in act_err(ode(vel_s, st_s[c], cs, x0, inputs), at).items()})
        rows.append(r)
        if i % 200 == 0:
            print("[2a]", i, len(teacher_obs), json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in r.items()}), flush=True)
    with (out / "exp2a_actions.jsonl").open("w") as f:
        f.writelines(json.dumps(r) + "\n" for r in rows)
    print("[2a-check]", json.dumps({"teacher_vs_recorded_max_abs": max(rec_dev), "student_prefix_same_across_checkpoints": cache_same}), flush=True)

    # Experiment 3: fixed subsets, every checkpoint.
    for set_name, obs, strat in (("b_teacher_states", teacher_obs, True), ("a_student_states", student_obs, False)):
        pick = subset(obs, a.subset, strat)
        (out / ("exp3_subset_%s.json" % set_name)).write_text(json.dumps([meta(obs[i]) for i in pick]))
        with (out / ("exp3_%s.jsonl" % set_name)).open("w") as f:
            for n_done, i in enumerate(pick):
                inputs, cs, ct, _ = prepare(obs[i])
                x0, eps = event_noise(SCORE_SEED0 + i, cfg)
                for c in CKPTS:
                    chain = sample_s(st_s[c], *cs, jnp.asarray(x0), jnp.asarray(eps))
                    vs = np.stack([np.asarray(vel_s(st_s[c], *cs, chain[:, k], s)) for k, s in enumerate(times)], axis=1)
                    vt = np.stack([np.asarray(vel_t(st_t, *ct, chain[:, k], s)) for k, s in enumerate(times)], axis=1)
                    at = ode(vel_t, st_t, ct, x0, inputs)
                    r = {**meta(obs[i]), "ckpt": c, "kl": kl_parts(vs, vt), "act": act_err(ode(vel_s, st_s[c], cs, x0, inputs), at)}
                    f.write(json.dumps(r) + "\n")
                if n_done % 40 == 0:
                    print("[3]", set_name, n_done, len(pick), flush=True)
    print("SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
