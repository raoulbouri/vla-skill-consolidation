"""Checks the evaluated student is the trained one, and how close its executed actions are to the teacher's. Same stored observations and noise for base, final student and teacher; ODE and SDE final latents through the policy's own output transform. No optimizer.
  CLVLA_USE_GPU=1 CLVLA_GPU=1 PYTHONPATH=scripts scripts/py-policy scripts/opd_t35_check.py <run_dir> <teacher_delta>"""
import json, os, sys
from pathlib import Path
os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"
import jax, jax.numpy as jnp, numpy as np
from clvla.config import load_experiment_config
from clvla.ogpo import pi05_path_kl as pkl, pi05_sde as sde, pi05_velocity as pv
from clvla.ogpo.pi05_sde_policy import event_noise, model_inputs, model_of
from clvla.policy import load_official_policy
from flax import nnx
from opd_lora_student import build_student_model, load_delta_teacher
cfg = sde.SDEConfig(); run = Path(sys.argv[1])
conf = load_experiment_config("configs/experiment.yaml")
policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
base = model_of(policy)
z = np.load(run / "student10d/checkpoints/adapter_100.npz"); student = build_student_model(base, {k.replace("__", "/"): z[k] for k in z.files})
teacher, _, _ = load_delta_teacher(base, Path(sys.argv[2]))
w = np.asarray(pkl.kl_coefficients(cfg)["weights"], np.float64)
b = np.load(run / "student10d/batches/update_099.npz"); prompt = "open the microwave"
M = {"base": base, "student": student, "teacher": teacher}
def prep(m, obs):
    return pv.build_prefix(m, obs)
vel = {n: jax.jit(lambda st, ob, mk, kv, x, s, gd=nnx.split(m)[0]: pv.velocity(nnx.merge(gd, st), pv.PrefixCache(ob, mk, kv), x, s)) for n, m in M.items()}
states = {n: nnx.split(m)[1] for n, m in M.items()}
rows = []
for j in range(len(b["events"])):
    req = {"observation/image": b[f"image_{j}"], "observation/wrist_image": b[f"wrist_{j}"], "observation/state": b[f"state_{j}"], "prompt": prompt}
    obs, inp = model_inputs(policy, req)
    caches = {n: prep(m, obs) for n, m in M.items()}
    for s_i in range(4):
        x0, eps = event_noise(700000 + 100 * j + s_i, cfg)
        out = {}
        for n in M:
            c = caches[n]; x = jnp.asarray(x0); y = jnp.asarray(x0)
            for k, s in enumerate(cfg.times()):
                y = y - 0.1 * vel[n](states[n], c.observation, c.prefix_mask, c.kv_cache, y, s)
            out[n] = np.asarray(policy._output_transform({"state": np.asarray(inp["state"]), "actions": np.asarray(y)[0]})["actions"])
        vs = {n: [] for n in ("student", "teacher", "base")}
        x = jnp.asarray(b[f"chain_{j}"])
        for k, s in enumerate(cfg.times()):
            for n in vs:
                c = caches[n]; vs[n].append(np.asarray(vel[n](states[n], c.observation, c.prefix_mask, c.kv_cache, x[:, k], s), np.float64))
        d = lambda a, e: float(np.sum(w * np.sum((np.stack(vs[a], 1) - np.stack(vs[e], 1)) ** 2, axis=(2, 3))))
        rows.append({"j": j, "rmse_exec_student_teacher": float(np.sqrt(np.mean((out["student"][:5] - out["teacher"][:5]) ** 2))), "rmse_exec_base_teacher": float(np.sqrt(np.mean((out["base"][:5] - out["teacher"][:5]) ** 2))), "gripper_student_teacher_sign_agree": float(np.mean(np.sign(out["student"][:5, 6]) == np.sign(out["teacher"][:5, 6]))), "gripper_base_teacher_sign_agree": float(np.mean(np.sign(out["base"][:5, 6]) == np.sign(out["teacher"][:5, 6]))), "kl_student_teacher": d("student", "teacher"), "kl_base_teacher": d("base", "teacher")})
res = {k: float(np.mean([r[k] for r in rows])) for k in rows[0] if k != "j"}
res["n"] = len(rows); res["adapter_b_max"] = max(float(np.abs(z[k]).max()) for k in z.files if "lora_b" in k)
(run / "action_check.json").write_text(json.dumps(res, indent=1)); print(json.dumps(res, indent=0))
