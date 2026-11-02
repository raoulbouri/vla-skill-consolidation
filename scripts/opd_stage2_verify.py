"""Stage 2: fresh-process replay of a recorded student episode, and evaluation of the fixed teacher on the student's states. No optimiser: gradients are computed once to measure memory and are discarded.
  CLVLA_USE_GPU=1 CLVLA_GPU=1 scripts/py-policy scripts/opd_stage2_verify.py --run D --teacher-delta T
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ["XLA_FLAGS"] = "--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0"

import jax
import jax.numpy as jnp
import numpy as np
from clvla.config import load_experiment_config
from clvla.ogpo import pi05_path_kl as pkl
from clvla.ogpo import pi05_pathwise_kl as pw
from clvla.ogpo import pi05_sde as sde
from clvla.ogpo import pi05_velocity as pv
from clvla.ogpo.pi05_sde_policy import KEEP, event_noise, make_sampler, model_inputs, model_of, tree_sha
from clvla.policy import load_official_policy
from flax import nnx, traverse_util
from openpi.models import pi0_config

ROOT = Path(__file__).resolve().parents[1]
cfg = sde.SDEConfig()


def err(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return {"max_abs": float(np.abs(a - b).max()), "rel_l2": float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-300)), "bitwise": bool(np.array_equal(a, b))}


def worst(rows):
    return {k: {"max_abs": max(r[k]["max_abs"] for r in rows), "rel_l2": max(r[k]["rel_l2"] for r in rows), "all_bitwise": all(r[k]["bitwise"] for r in rows)} for k in rows[0]}


def leaf_sha(v):
    a = np.asarray(v)
    return hashlib.sha256(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()).hexdigest()


def load_teacher(student, delta_dir):
    """Teacher as create_trained_policy builds it (every leaf BF16), sharing unchanged arrays with the student. Refuses to continue unless every leaf hash matches the original checkpoint's."""
    sha = json.loads((delta_dir / "leaf_sha.json").read_text())
    z = np.load(delta_dir / "delta.npz")
    flat = traverse_util.flatten_dict(nnx.split(student)[1].to_pure_dict(), sep="/")
    for k, v in flat.items():
        if k in sha["base_bf16"]:
            assert leaf_sha(v) == sha["base_bf16"][k], f"student leaf differs from the teacher's base: {k}"
    params = {k: v for k, v in flat.items() if k not in sha["delta_keys"]}
    params.update({k: jnp.asarray(z[k.replace("/", "__")]).astype(jnp.bfloat16) for k in sha["delta_keys"]})
    assert set(params) == set(sha["teacher_bf16"]), "teacher leaf set differs"
    bad = [k for k, v in params.items() if leaf_sha(v) != sha["teacher_bf16"][k]]
    assert not bad, f"rebuilt teacher differs from the checkpoint: {bad[:5]}"
    conf = pi0_config.Pi0Config(pi05=True, action_horizon=10, discrete_state_input=False, paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
    teacher = conf.load(traverse_util.unflatten_dict(params, sep="/"))
    return teacher, {"leaves_verified": len(params), "shared_with_student": sum(1 for k in params if k not in sha["delta_keys"]), "own_leaves": len(sha["delta_keys"]), "delta_npz_sha256": hashlib.sha256((delta_dir / "delta.npz").read_bytes()).hexdigest()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--teacher-delta", required=True)
    a = ap.parse_args()
    run, delta_dir = Path(a.run), Path(a.teacher_delta)
    t_start = time.perf_counter()
    conf = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    student = model_of(policy)
    env_meta = json.loads((run / "environment.json").read_text())
    z_env = np.load(run / "environment.npz")
    files = sorted(run.glob("event_*.npz"))
    R = {"xla_flags": os.environ["XLA_FLAGS"], "environment": {k: env_meta[k] for k in ("libero_90_index", "task", "instruction", "reset", "env_seed", "success", "error", "policy_steps", "policy_calls", "env_steps", "reward_sum", "wall_s")}, "events": len(files)}
    run_fn, st = make_sampler(student, cfg)
    R["student_sha"] = {"start": (run / "student_sha_start.txt").read_text(), "end_of_rollout": (run / "student_sha_end.txt").read_text(), "replay_process_start": tree_sha(st)}

    # Replay in this fresh process, same programme: all events
    rows, keep_events, ready_ok, seed_ok = [], [], True, True
    for i, f in enumerate(files):
        e = dict(np.load(f))
        request = {"observation/image": e["observation_image"], "observation/wrist_image": e["observation_wrist_image"], "observation/state": e["observation_state"], "prompt": str(e["prompt"])}
        obs, inputs = model_inputs(policy, request)
        ready_ok = ready_ok and tree_sha({k: inputs[k] for k in KEEP}) == str(e["model_ready_sha"])
        x0, eps = event_noise(int(e["seed"]), cfg)
        seed_ok = seed_ok and np.array_equal(x0, e["x0"]) and np.array_equal(eps, e["eps"])
        out = jax.block_until_ready(run_fn(obs, jnp.asarray(e["x0"]), jnp.asarray(e["eps"])))
        phys = np.asarray(policy._output_transform({"state": np.asarray(inputs["state"]), "actions": np.asarray(out["chain"])[0, cfg.num_steps]})["actions"])
        rows.append({"chain": err(out["chain"], e["chain"]), "means": err(out["means"], e["means"]), "final_latent": err(np.asarray(out["chain"])[0, -1], e["chain"][0, -1]), "physical": err(phys, e["physical"]), "stds": err(out["stds"], e["stds"])})
        keep_events.append(e)
    R["replay_same_programme"] = worst(rows)
    R["replay_model_inputs_identical"], R["replay_noise_regenerated_from_seed"] = bool(ready_ok), bool(seed_ok)

    # Action path and environment consistency
    K, wait, replan = cfg.num_steps, env_meta["num_steps_wait"], env_meta["replan_steps"]
    chunks, actions, at = z_env["chunks"], z_env["actions"], z_env["step_at_call"]
    states = z_env["states"]
    ok_exec, ok_state, ok_req = True, True, True
    for i, e in enumerate(keep_events):
        ok_req = ok_req and env_meta["requests"][i]["image_sha"] == hashlib.sha256(np.ascontiguousarray(e["observation_image"]).tobytes()).hexdigest() and env_meta["requests"][i]["wrist_sha"] == hashlib.sha256(np.ascontiguousarray(e["observation_wrist_image"]).tobytes()).hexdigest() and np.array_equal(np.asarray(env_meta["requests"][i]["state"]), e["observation_state"])
        ok_exec = ok_exec and np.array_equal(chunks[i], e["physical"])
        n = actions[at[i]: at[i] + replan]
        ok_exec = ok_exec and np.array_equal(n, chunks[i][: len(n)].astype(np.float64))
        pos, grip = np.asarray(env_meta["requests"][i]["state"])[:3], np.asarray(env_meta["requests"][i]["state"])[-2:]
        ok_state = ok_state and np.array_equal(pos, states[at[i]][:3]) and np.array_equal(grip, states[at[i]][-2:])
    pa = actions[wait:]
    shas = z_env["image_sha"]
    R["actions"] = {"chunk_shape": list(chunks.shape), "latent_shape": list(keep_events[0]["chain"].shape), "all_finite": bool(np.isfinite(chunks).all()), "executed_equal_recorded_chunk": bool(ok_exec), "sim_request_equal_server_record": bool(ok_req), "request_state_from_latest_env_obs": bool(ok_state),
                    "calls_times_replan_covers_policy_steps": bool(env_meta["policy_steps"] <= env_meta["policy_calls"] * replan), "executed_per_call": sorted({int(x) for x in np.diff(at)}), "outside_arm_input_range_fraction": float(np.mean(np.abs(pa[:, :6]) > 1.0)), "arm_abs_max": float(np.abs(pa[:, :6]).max()), "gripper_min": float(pa[:, 6].min()), "gripper_max": float(pa[:, 6].max()),
                    "image_changed_on_step_fraction": float(np.mean(shas[wait + 1:] != shas[wait:-1])), "state_changed_on_step_fraction": float(np.mean(np.any(states[wait + 1:] != states[wait:-1], axis=1)))}

    # Teacher, frozen, on the student's stored states
    teacher, rec = load_teacher(student, delta_dir)
    R["teacher"] = rec
    gd_t, t_state = nnx.split(teacher)
    R["teacher"]["state_sha_before"] = tree_sha(t_state)
    gd_s, s_state = nnx.split(student)

    def prefix_t(st, obs):
        c = pv.build_prefix(nnx.merge(gd_t, st), obs)
        return c.observation, c.prefix_mask, c.kv_cache

    def prefix_s(st, obs):
        c = pv.build_prefix(nnx.merge(gd_s, st), obs)
        return c.observation, c.prefix_mask, c.kv_cache

    def kl_eval(ss, tt, ob, sm, sk, tm, tk, chain):
        return pkl.path_kl(nnx.merge(gd_t, tt), pv.PrefixCache(ob, tm, tk), nnx.merge(gd_s, ss), pv.PrefixCache(ob, sm, sk), chain, cfg)

    jt, js, jk = jax.jit(prefix_t), jax.jit(prefix_s), jax.jit(kl_eval)
    t0 = time.perf_counter()
    kls, ratio, execd, prefix_diff, finite = [], [], [], [], True
    prefixes = {}
    shape = None
    for i, e in enumerate(keep_events):
        request = {"observation/image": e["observation_image"], "observation/wrist_image": e["observation_wrist_image"], "observation/state": e["observation_state"], "prompt": str(e["prompt"])}
        obs, _ = model_inputs(policy, request)
        ob, sm, sk = js(s_state, obs)
        _, tm, tk = jt(t_state, obs)
        o = jk(s_state, t_state, ob, sm, sk, tm, tk, jnp.asarray(e["chain"]))
        vt, vs = np.asarray(o["velocities_teacher"]), np.asarray(o["velocities_student"])
        finite = finite and np.isfinite(vt).all() and np.isfinite(float(o["path_kl_velocity"][0]))
        shape = list(vt.shape)
        kls.append(float(o["path_kl_velocity"][0]))
        ratio.append(float(o["log_ratio"][0]))
        execd.append(float(o["kl_groups"][0, :, 0].sum()))
        if i == 0:
            prefix_diff = {"kv_cache_equal": all(bool(jnp.array_equal(x, y)) for x, y in zip(jax.tree.leaves(sk), jax.tree.leaves(tk)))}
        if i in (0, len(keep_events) // 2, len(keep_events) - 1):
            prefixes[i] = (ob, sm, sk, tm, tk, e)
    R["teacher"].update({"seconds_all_events": time.perf_counter() - t0, "velocity_shape": shape, "all_finite": bool(finite), "first_event_prefix": prefix_diff, "path_kl_per_event": kls, "path_kl_mean": float(np.mean(kls)), "path_kl_min": float(np.min(kls)), "path_kl_max": float(np.max(kls)), "log_ratio_mean": float(np.mean(ratio)), "executed_group_kl_mean": float(np.mean(execd)), "generates_environment_actions": False, "note": "teacher exists only in this process, after the rollout has ended; the rollout server never loaded it"})

    # Teacher input derivative through frozen parameters
    ob, sm, sk, tm, tk, e = prefixes[0]
    frozen = pw.frozen_view(teacher)
    x, u = jnp.asarray(e["chain"][:, 5]), jnp.asarray(np.random.default_rng(50999).standard_normal(e["chain"][:, 5].shape), jnp.float32)
    _, tangent = jax.jvp(lambda y: pv.velocity(frozen, pv.PrefixCache(ob, tm, tk), y, cfg.times()[5]), (x,), (u,))
    R["teacher"]["input_derivative"] = {"finite": bool(jnp.isfinite(tangent).all()), "norm": float(jnp.linalg.norm(tangent))}

    # The training programme on the same noise: how far its chain is from the executed one. One gradient, discarded, to measure memory.
    gd2, theta, other = nnx.split(student, nnx.All(nnx.Param, lambda path, _: "/".join(str(p) for p in path).startswith(("action_", "time_mlp")) or ("/llm/" in "/".join(str(p) for p in path) and "_1" in "/".join(str(p) for p in path))), ...)

    def f(th, ob, sm, sk, tm, tk, x0, eps):
        o = pw.pathwise_kl(nnx.merge(gd2, th, other), pv.PrefixCache(ob, sm, sk), nnx.merge(gd_t, t_state), pv.PrefixCache(ob, tm, tk), x0, eps, cfg, mode="full")
        return jnp.sum(o["path_kl"]), o["chain"]

    grad = jax.jit(jax.value_and_grad(f, has_aux=True))
    rows = []
    t0 = time.perf_counter()
    for i, (ob, sm, sk, tm, tk, e) in prefixes.items():
        (val, chain), g = grad(theta, ob, sm, sk, tm, tk, jnp.asarray(e["x0"]), jnp.asarray(e["eps"]))
        rows.append({"event": i, "kl": float(val), "chain": err(chain, e["chain"]), "final_latent": err(np.asarray(chain)[0, -1], e["chain"][0, -1]), "grad_finite": all(bool(jnp.isfinite(v.astype(jnp.float32)).all()) for v in jax.tree.leaves(g))})
        del g
    R["replay_training_programme"] = {"rows": rows, "seconds_per_gradient": (time.perf_counter() - t0) / len(rows), "n_trainable": int(sum(v.size for v in jax.tree.leaves(theta)))}

    R["teacher"]["state_sha_after"] = tree_sha(nnx.split(teacher)[1])
    R["student_sha"]["replay_process_end"] = tree_sha(nnx.split(student)[1])
    status = json.loads((run / "server_status.json").read_text())
    R["sde_sample_s"] = {"first_including_compile": status["sample_s"][0], "mean_after_first": float(np.mean(status["sample_s"][1:])), "max_after_first": float(np.max(status["sample_s"][1:]))}
    R["peak_gpu_gib"] = {"rollout_server": status["peak_gpu_bytes"] / 2**30, "replay_teacher_gradient": jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30}
    R["verify_seconds"] = time.perf_counter() - t_start
    s, rs = R["student_sha"], R["replay_same_programme"]
    R["acceptance"] = {"1_task6_sampler_used": status["events"] == len(files) == env_meta["policy_calls"], "2_actions_finite_and_valid": R["actions"]["all_finite"] and R["actions"]["chunk_shape"][1:] == [10, 7], "3_env_executed_student_actions": R["actions"]["executed_equal_recorded_chunk"] and R["actions"]["sim_request_equal_server_record"] and not env_meta["error"],
                       "4_observation_updated": R["actions"]["request_state_from_latest_env_obs"] and R["actions"]["image_changed_on_step_fraction"] > 0.9, "5_teacher_evaluates_student_states": R["teacher"]["all_finite"] and R["teacher"]["velocity_shape"] == [1, 10, 10, 32], "6_noise_reproduces_chain": rs["chain"]["all_bitwise"] and rs["final_latent"]["all_bitwise"] and rs["means"]["all_bitwise"] and rs["physical"]["all_bitwise"],
                       "7_no_parameters_updated": s["start"] == s["end_of_rollout"] == s["replay_process_start"] == s["replay_process_end"], "8_teacher_unchanged_in_process": R["teacher"]["state_sha_before"] == R["teacher"]["state_sha_after"]}
    (run / "report.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print(json.dumps({k: R[k] for k in ("events", "environment", "student_sha", "replay_same_programme", "replay_model_inputs_identical", "replay_noise_regenerated_from_seed", "actions", "replay_training_programme", "sde_sample_s", "peak_gpu_gib", "acceptance", "verify_seconds")}, indent=0), flush=True)
    print(json.dumps({k: v for k, v in R["teacher"].items() if k != "path_kl_per_event"}), flush=True)


if __name__ == "__main__":
    main()
