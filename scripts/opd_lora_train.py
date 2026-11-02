"""Single-task teacher-only pathwise OPD for one LoRA student. Fresh on-policy episode before every update; reward logged, never used. Stops at a gate rather than continuing on a finite loss.
  CLVLA_USE_GPU=1 CLVLA_GPU=G PYTHONPATH=scripts:tests/ogpo_port scripts/py-policy scripts/opd_lora_train.py --teacher-run D --name student10d --out O --updates 100 --position 4 --adapter-seed 7
Fixed before the first run: AdamW as in the repository's recipe (lr 5e-5, b1 .9, b2 .95, eps 1e-8, wd 1e-10, clip 1.0); 8 evenly spaced observations and 2 trajectories per observation per update (recorded noise + fresh noise); training states demo_0..N-1 in order (--train-demos).
Gate after update 1: forward parameters changed, and fresh-noise paired KL 95% interval lies below zero. Later stops: non-finite values, or diagnostic KL above 1.5x its initial value.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gc
import hashlib
import json
import os
import subprocess
import threading
import time
from pathlib import Path

os.environ.setdefault("TAG", "opd_lora")
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
from clvla.ogpo.pi05_sde_policy import event_noise, model_inputs, model_of, tree_sha
from clvla.openpi_training import load_trained_policy
from clvla.policy import load_official_policy
from flax import nnx, traverse_util
from openpi.serving import websocket_policy_server
from opd_lora_student import HEADS, build_student_model, initial_adapter, is_trainable, load_delta_teacher

ROOT = Path(__file__).resolve().parents[1]
cfg = sde.SDEConfig()
B_OBS, M, N_DIAG, DIAG_EVERY, DIRECT_EVERY = 8, 2, 8, 10, 10
LR, B1, B2, EPS, WD, CLIP = 5e-5, 0.9, 0.95, 1e-8, 1e-10, 1.0
CKPT_AT = (0, 25, 50, 75, 100, 200, 300, 400, 500)
w = np.asarray(pkl.kl_coefficients(cfg)["weights"], np.float64)
MASKS = {}
for name, (r_, c_) in pkl.GROUPS.items():
    m_ = np.zeros((10, 32))
    m_[r_, c_] = 1.0
    MASKS[name] = m_


def release(tag):
    """Return freed host memory to the OS and log what is left: aims has 61 GB and the pathwise compile is large."""
    gc.collect()
    ctypes.CDLL("libc.so.6").malloc_trim(0)
    hwm = next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) / 2**20
    print(f"[mem] {tag} rss {T_rss():.1f} GiB peak {hwm:.1f} GiB t {time.time():.0f}", flush=True)


def T_rss():
    return int(open("/proc/self/statm").read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 2**30


def seed_ok(s):
    assert not (12000 <= s <= 12007 or 13000 <= s <= 13007 or s == 15000)
    return s


def kl_parts(vs, vt):
    d = (np.asarray(vs, np.float64) - np.asarray(vt, np.float64)) ** 2
    return {"full": float(np.sum(w * np.sum(d, axis=(2, 3)))), "per_step": (w * np.sum(d, axis=(2, 3)))[0].tolist(), **{n: float(np.sum(w * np.sum(d * m, axis=(2, 3)))) for n, m in MASKS.items()}}


def paired(a, b, B):
    d = np.asarray(b, np.float64) - np.asarray(a, np.float64)
    o = d.reshape(B, -1).mean(axis=1)
    se = float(o.std(ddof=1) / B**0.5)
    tv = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365}.get(B - 1, 2.365)
    return {"before": float(np.mean(a)), "after": float(np.mean(b)), "diff": float(d.mean()), "relative": float(d.mean() / np.mean(a)), "se_obs": se, "ci95_obs": [float(d.mean() - tv * se), float(d.mean() + tv * se)], "obs_decreased": int(np.sum(o < 0)), "pairs_decreased": int(np.sum(d < 0)), "pairs": int(d.size)}


class Server:
    """Hands the trainer's control messages to the simulator and samples with the current forward parameters. Never blocks the socket while training."""

    metadata = {"sampler": "task6_ogpo_discrete"}

    def __init__(self, sample):
        self.sample, self.lock, self.pending, self.events, self.result, self.seed, self.stop = sample, threading.Lock(), None, [], None, 0, False

    def infer(self, request):
        ctl = request.get("_ctl")
        if ctl == "next":
            with self.lock:
                if self.stop:
                    return {"actions": 0, "stop": True}
                if self.pending is None:
                    return {"actions": 0, "wait": True}
                d, self.pending = self.pending, None
            return {"actions": 0, "demo": d}
        if ctl == "done":
            with self.lock:
                self.result = {k: request[k] for k in ("success", "steps", "calls", "reward", "error", "seconds", "door_min", "start_sha", "start_min_l2_to_held_out", "arm_offset_max")}
            return {"actions": 0}
        s = seed_ok(self.seed + len(self.events))
        phys, rec = self.sample(request, s)
        self.events.append(rec)
        return {"actions": phys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher-run", required=True)
    ap.add_argument("--teacher-delta")
    ap.add_argument("--train-demos", type=int, default=25, help="rollout states demo_0..N-1")
    ap.add_argument("--profile", default="n25")
    ap.add_argument("--random-starts", action="store_true", help="fresh environment reset for every rollout, seeded by update")
    ap.add_argument("--arm-noise", type=float, default=0.0, help="uniform half-width in rad added to the reset arm joints, random starts only")
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--updates", type=int, default=100)
    ap.add_argument("--position", type=int, default=4)
    ap.add_argument("--adapter-seed", type=int, default=7)
    ap.add_argument("--sanity-only", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "batches").mkdir(exist_ok=True)
    R = {"name": a.name, "teacher_run": a.teacher_run, "xla_flags": os.environ["XLA_FLAGS"], "optimizer": {"type": "adamw", "lr": LR, "b1": B1, "b2": B2, "eps": EPS, "weight_decay": WD, "clip": CLIP}, "batch": {"observations": B_OBS, "trajectories_per_observation": M}, "updates_requested": a.updates}

    def save():
        (out / "metrics.json").write_text(json.dumps(R, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

    H = T.helpers()
    flat = H["_flat"]
    conf = load_experiment_config(ROOT / "configs/experiment.yaml")
    policy, _ = load_official_policy(conf.checkpoint.openpi_config, conf.checkpoint.local_dir, expected_asset_id=conf.checkpoint.norm_stats_asset_id, state_dim=conf.policy.state_dim, action_dim=conf.policy.action_dim, action_horizon=conf.policy.action_horizon)
    base_model = model_of(policy)
    base_sha = tree_sha(nnx.split(base_model)[1])
    release("base_loaded")
    if a.teacher_delta:
        teacher, tstep, tck = load_delta_teacher(base_model, Path(a.teacher_delta), R)
    else:
        tpolicy, tstep, tck = load_trained_policy(Path(a.teacher_run), conf.checkpoint.local_dir)
        teacher = model_of(tpolicy)
    gd_t, t_state = nnx.split(teacher)
    t_sha = tree_sha(t_state)
    release("teacher_loaded")
    t_summary = json.loads((Path(a.teacher_run) / "summary.json").read_text())
    adapter0 = initial_adapter(str(Path(conf.checkpoint.local_dir) / "params"), a.adapter_seed)
    adapter_sha = hashlib.sha256(b"".join(adapter0[k].tobytes() for k in sorted(adapter0))).hexdigest()
    student = build_student_model(base_model, adapter0)
    keys = sorted(adapter0)
    trainable = nnx.All(nnx.Param, lambda path, _: is_trainable("/".join(str(p) for p in path)))
    vlm_f = nnx.All(nnx.Param, lambda path, _: "/".join(str(p) for p in path).startswith("PaliGemma/img") or ("/llm/" in "/".join(str(p) for p in path) and "_1" not in "/".join(str(p) for p in path)))
    gd_s, theta0, vlm, other = nnx.split(student, trainable, vlm_f, ...)
    assert sorted(flat(theta0)) == keys, "trainable leaves differ from the adapter"
    gd_all, s_full = nnx.split(student)
    frozen_sha = tree_sha((vlm, other))
    release("student_built")
    R["identity"] = {"base_sha": base_sha, "teacher_sha": t_sha, "teacher_checkpoint": str(tck), "teacher_step": tstep, "teacher_fingerprint": t_summary.get("trained_checkpoint_fingerprint"), "adapter_seed": a.adapter_seed, "initial_adapter_sha": adapter_sha, "frozen_student_sha": frozen_sha, "n_trainable": int(sum(v.size for v in adapter0.values())), "n_lora_b": int(sum(v.size for k, v in adapter0.items() if "lora_b" in k)), "leaves": len(keys)}
    save()
    np.savez(out / "checkpoints" / "adapter_000.npz", **{k.replace("/", "__"): adapter0[k] for k in keys})

    leaves, treedef = jax.tree.flatten(theta0)
    by_id = {id(v): k for k, v in flat(theta0).items()}
    order = [by_id[id(l)] for l in leaves]
    dtypes = {k: l.dtype for k, l in zip(order, leaves)}

    def forward(master):
        return jax.tree.unflatten(treedef, [jnp.asarray(master[k]).astype(dtypes[k]) for k in order])

    def s_prefix(st, obs):
        c = pv.build_prefix(nnx.merge(gd_all, st), obs)
        return c.observation, c.prefix_mask, c.kv_cache

    def t_prefix(st, obs):
        c = pv.build_prefix(nnx.merge(gd_t, st), obs)
        return c.observation, c.prefix_mask, c.kv_cache

    def merge_s(th):
        return nnx.merge(gd_s, th, vlm, other)

    def sample_(th, ob, sm, sk, x0, eps):
        o = sde.sde_sample(merge_s(th), pv.PrefixCache(ob, sm, sk), x0, eps, cfg)
        return {k: o[k] for k in ("chain", "means")}

    def vel_s_(th, ob, sm, sk, x, s_):
        return pv.velocity(merge_s(th), pv.PrefixCache(ob, sm, sk), x, s_)

    def vel_t_(tt, ob, tm, tk, x, s_):
        return pv.velocity(nnx.merge(gd_t, tt), pv.PrefixCache(ob, tm, tk), x, s_)

    def obj(th, ob, sm, sk, tm, tk, x0, eps):
        o = pw.pathwise_kl(merge_s(th), pv.PrefixCache(ob, sm, sk), pw.frozen_view(nnx.merge(gd_t, t_state)), pv.PrefixCache(ob, tm, tk), x0, eps, cfg, mode="full")
        return jnp.sum(o["path_kl"]), {"chain": o["chain"], "vs": o["velocities_student"], "vt": o["velocities_teacher"]}

    def obj_det(th, ob, sm, sk, tm, tk, stored):
        o = pw.pathwise_kl(merge_s(th), pv.PrefixCache(ob, sm, sk), pw.frozen_view(nnx.merge(gd_t, t_state)), pv.PrefixCache(ob, tm, tk), None, None, cfg, mode="detached", stored_chain=stored)
        return jnp.sum(o["path_kl"])

    jp_s, jp_t, j_sample, j_vs, j_vt = jax.jit(s_prefix), jax.jit(t_prefix), jax.jit(sample_), jax.jit(vel_s_), jax.jit(vel_t_)
    j_grad, j_det = jax.jit(jax.value_and_grad(obj, has_aux=True)), jax.jit(jax.grad(obj_det))
    master = dict(adapter0)
    theta = forward(master)
    state_box = {"theta": theta}
    sample_seconds = []

    def serve_sample(request, seed):
        x0, eps = event_noise(seed, cfg)
        obs, inputs = model_inputs(policy, request)
        ob, sm, sk = jp_s(s_full, obs)
        t0 = time.perf_counter()
        o = jax.block_until_ready(j_sample(state_box["theta"], ob, sm, sk, x0, eps))
        sample_seconds.append(time.perf_counter() - t0)
        chain = np.asarray(o["chain"])
        phys = np.asarray(policy._output_transform({"state": np.asarray(inputs["state"]), "actions": chain[0, cfg.num_steps]})["actions"])
        return phys, {"seed": seed, "x0": x0, "eps": eps, "chain": chain, "means": np.asarray(o["means"]), "request": {k: np.asarray(request[k]) if k != "prompt" else str(request[k]) for k in ("observation/image", "observation/wrist_image", "observation/state", "prompt")}, "physical": phys}

    release("before_rollout")
    srv = Server(serve_sample)
    port = int(subprocess.check_output([os.environ["CLVLA_SIM_PY"], "-c", "import socket; s=socket.socket(); s.bind(('127.0.0.1', 0)); print(s.getsockname()[1])"]).strip())
    th_srv = threading.Thread(target=websocket_policy_server.WebsocketPolicyServer(policy=srv, host="127.0.0.1", port=port, metadata=srv.metadata).serve_forever, daemon=True)
    th_srv.start()
    sim_log = open(out / "simulator.log", "w")
    sim = subprocess.Popen(["scripts/py-sim", "-W", "ignore", "scripts/opd_lora_collect.py", "--port", str(port), "--position", str(a.position), "--profile", a.profile, *(["--random-starts", "--arm-noise", str(a.arm_noise)] if a.random_starts else [])], stdout=sim_log, stderr=subprocess.STDOUT, cwd=ROOT, env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1"})

    def collect(demo, seed):
        with srv.lock:
            srv.events, srv.result, srv.seed = [], None, seed
            srv.pending = demo
        while True:
            if sim.poll() is not None:
                raise RuntimeError("simulator exited: see simulator.log")
            with srv.lock:
                if srv.result is not None:
                    return srv.events, srv.result
            time.sleep(0.2)

    def prepare(ev):
        obs, _ = model_inputs(policy, ev["request"])
        ob, sm, sk = jp_s(s_full, obs)
        ob_t, tm, tk = jp_t(t_state, obs)
        assert tree_sha(ob) == tree_sha(ob_t)
        return {"ob": ob, "sm": sm, "sk": sk, "tm": tm, "tk": tk, "same_cache": all(bool(jnp.array_equal(x, y)) for x, y in zip(jax.tree.leaves(sk), jax.tree.leaves(tk)))}

    def diag_eval(th, prepared, base_seed, n):
        rows = []
        for i, p in enumerate(prepared):
            for m in range(n):
                x0, eps = event_noise(seed_ok(base_seed + 100 * i + m), cfg)
                chain = j_sample(th, p["ob"], p["sm"], p["sk"], x0, eps)["chain"]
                vs = np.stack([np.asarray(j_vs(th, p["ob"], p["sm"], p["sk"], chain[:, k], s_)) for k, s_ in enumerate(cfg.times())], axis=1)
                vt = np.stack([np.asarray(j_vt(t_state, p["ob"], p["tm"], p["tk"], chain[:, k], s_)) for k, s_ in enumerate(cfg.times())], axis=1)
                q = kl_parts(vs, vt)
                rows.append([q["full"], q["executed"], q["unexecuted"], q["padded"]])
        return np.array(rows, np.float64)

    opt = optax.chain(optax.clip_by_global_norm(CLIP), optax.adamw(LR, b1=B1, b2=B2, eps=EPS, weight_decay=WD))
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        mast_jax = {k: jax.device_put(master[k], cpu) for k in keys}
        opt_state = opt.init(mast_jax)
    metrics, rollouts = [], []
    csv_path, roll_path = out / "training_metrics.csv", out / "rollout_records.jsonl"
    diag_prep, diag0 = None, None
    R["diagnostic"] = {"note": "fixed observations from the first rollout, fixed noise, every 10 updates", "history": []}
    R["direct_vs_state"] = []
    stop_reason = None
    t_train = time.perf_counter()
    u = 0
    try:
        while u < a.updates:
            t_u = time.perf_counter()
            demo = 900000 + u if a.random_starts else u % a.train_demos
            events, res = collect(demo, 400000 + 1000 * u)
            if u == 0:
                release("after_first_rollout")
            rollouts.append({"update": u, "demo": demo, **res, "events": len(events)})
            with roll_path.open("a") as f:
                f.write(json.dumps(rollouts[-1], default=str) + "\n")
            if res["error"]:
                stop_reason = f"rollout error: {res['error']}"
                break
            idx = sorted({int(round(x)) for x in np.linspace(0, len(events) - 1, B_OBS)})
            sel = [events[i] for i in idx]
            prep = [prepare(e) for e in sel]
            if diag_prep is None:
                diag_prep = prep
                diag0 = diag_eval(theta, diag_prep, 500000, N_DIAG)
                release("after_first_diagnostic")
                R["diagnostic"]["events"] = idx
                R["diagnostic"]["history"].append({"update": 0, "mean": float(diag0[:, 0].mean())})
                R["identity"]["student_teacher_prefix_cache_identical"] = bool(prep[0]["same_cache"])
            if u == 0:
                release("before_gradient")
            acc = {k: 0.0 for k in keys}
            acc_det = {k: 0.0 for k in keys}
            do_det = u % DIRECT_EVERY == 0
            kls, gaps, chain_rel, finite, s_rms, t_rms = [], [], [], True, [], []
            np.savez_compressed(out / "batches" / f"update_{u:03d}.npz", events=np.array(idx), demo=demo, **{f"image_{j}": s["request"]["observation/image"] for j, s in enumerate(sel)}, **{f"wrist_{j}": s["request"]["observation/wrist_image"] for j, s in enumerate(sel)}, **{f"state_{j}": s["request"]["observation/state"] for j, s in enumerate(sel)}, **{f"x0_{j}": s["x0"] for j, s in enumerate(sel)}, **{f"eps_{j}": s["eps"] for j, s in enumerate(sel)}, **{f"chain_{j}": s["chain"] for j, s in enumerate(sel)})
            for j, (ev, p) in enumerate(zip(sel, prep)):
                for m in range(M):
                    x0, eps = (ev["x0"], ev["eps"]) if m == 0 else event_noise(seed_ok(450000 + 1000 * u + 10 * j + m), cfg)
                    (val, aux), g = j_grad(theta, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], x0, eps)
                    aux = jax.tree.map(np.asarray, aux)
                    gh = flat(g)
                    finite = finite and np.isfinite(float(val)) and all(bool(jnp.isfinite(gh[k].astype(jnp.float32)).all()) for k in keys)
                    for k in keys:
                        acc[k] = acc[k] + np.asarray(gh[k]).astype(np.float32)
                    del g, gh
                    parts = kl_parts(aux["vs"], aux["vt"])
                    kls.append(parts)
                    s_rms.append(float(np.sqrt(np.mean(aux["vs"] ** 2))))
                    t_rms.append(float(np.sqrt(np.mean(aux["vt"] ** 2))))
                    if m == 0:
                        c = np.asarray(ev["chain"], np.float64)
                        chain_rel.append(float(np.linalg.norm(np.asarray(aux["chain"], np.float64) - c) / np.linalg.norm(c)))
                    if do_det:
                        gdt = flat(j_det(theta, p["ob"], p["sm"], p["sk"], p["tm"], p["tk"], jnp.asarray(aux["chain"])))
                        for k in keys:
                            acc_det[k] = acc_det[k] + np.asarray(gdt[k]).astype(np.float32)
                        del gdt
            if u == 0:
                release("after_first_gradients")
            n_traj = len(sel) * M
            g_m = {k: acc[k] / n_traj for k in keys}
            dot = lambda x, y: sum(float(np.sum(np.asarray(x[k], np.float64) * np.asarray(y[k], np.float64))) for k in keys)  # noqa: E731
            gn = dot(g_m, g_m) ** 0.5
            def by(x):
                out_ = {"lora_a": 0.0, "lora_b": 0.0, "heads": 0.0}
                for k in keys:
                    out_["lora_a" if "lora_a" in k else "lora_b" if "lora_b" in k else "heads"] += float(np.sum(np.asarray(x[k], np.float64) ** 2))
                return {n: v**0.5 for n, v in out_.items()}
            gr = by(g_m)
            if do_det:
                g_d = {k: acc_det[k] / n_traj for k in keys}
                g_s = {k: g_m[k] - g_d[k] for k in keys}
                dn, sn = dot(g_d, g_d) ** 0.5, dot(g_s, g_s) ** 0.5
                R["direct_vs_state"].append({"update": u, "full": gn, "direct": dn, "state": sn, "state_over_full": sn / gn, "cos_full_direct": dot(g_m, g_d) / (gn * dn), "cos_full_state": dot(g_m, g_s) / (gn * sn)})
            if u == 0:
                release("after_detached_gradients")
                # the gate's pre-update fresh-noise baseline on this batch, and VLM derivative routing
                sanity_before = diag_eval(theta, prep, 510000, N_DIAG * 2)
                try:
                    def obj_vlm(v, p_, x0_, eps_):
                        o = pw.pathwise_kl(nnx.merge(gd_s, theta, v, other), pv.PrefixCache(p_["ob"], p_["sm"], p_["sk"]), pw.frozen_view(nnx.merge(gd_t, t_state)), pv.PrefixCache(p_["ob"], p_["tm"], p_["tk"]), x0_, eps_, cfg, mode="full")
                        return jnp.sum(o["path_kl"])

                    gv = jax.jit(jax.grad(obj_vlm))(vlm, {k: prep[0][k] for k in ("ob", "sm", "sk", "tm", "tk")}, sel[0]["x0"], sel[0]["eps"])
                    R["routing_vlm_nonzero"] = int(sum(int(jnp.count_nonzero(v)) for v in jax.tree.leaves(gv)))
                    del gv
                except Exception as e_:  # noqa: BLE001 - recorded; the update gate below requires it
                    R["routing_vlm_nonzero"] = f"{type(e_).__name__}: {str(e_)[:160]}"
                release("after_vlm_routing")
            # the update: FP32 master on CPU
            with jax.default_device(cpu):
                gr32 = {k: jax.device_put(np.asarray(g_m[k], np.float32), cpu) for k in keys}
                mast_jax = {k: jax.device_put(master[k], cpu) for k in keys}
                updates, opt_state = opt.update(gr32, opt_state, mast_jax)
                new = optax.apply_updates(mast_jax, updates)
            new_np = {k: np.asarray(new[k]) for k in keys}
            del new, updates, gr32
            theta_new = forward(new_np)
            fn, fo = flat(theta_new), flat(theta)
            eff = {k: np.asarray(fn[k]).astype(np.float64) - np.asarray(fo[k]).astype(np.float64) for k in keys}
            intended = {k: new_np[k].astype(np.float64) - master[k].astype(np.float64) for k in keys}
            ein, iin = dot(eff, eff) ** 0.5, dot(intended, intended) ** 0.5
            ch = {"lora_a": [0, 0], "lora_b": [0, 0], "heads": [0, 0]}
            for k in keys:
                c = ch["lora_a" if "lora_a" in k else "lora_b" if "lora_b" in k else "heads"]
                c[0] += int(np.count_nonzero(eff[k]))
                c[1] += int(eff[k].size)
            state_finite = all(bool(np.isfinite(np.asarray(x)).all()) for x in jax.tree.leaves(opt_state) if hasattr(x, "shape") and jnp.issubdtype(jnp.asarray(x).dtype, jnp.floating))
            row = {"update": u + 1, "demo": demo, "rollout_success": res["success"], "rollout_steps": res["steps"], "rollout_reward": res["reward"], "kl_mean": float(np.mean([x["full"] for x in kls])), "kl_executed": float(np.mean([x["executed"] for x in kls])), "kl_unexecuted": float(np.mean([x["unexecuted"] for x in kls])), "kl_padded": float(np.mean([x["padded"] for x in kls])), "grad_norm": gn, "grad_lora_a": gr["lora_a"], "grad_lora_b": gr["lora_b"], "grad_heads": gr["heads"], "clip_scale": min(1.0, CLIP / gn), "master_step": iin, "forward_step": ein, "forward_over_master": ein / iin if iin else None, "cos_forward_master": dot(eff, intended) / (ein * iin) if ein and iin else None, "changed_lora_a": ch["lora_a"][0] / ch["lora_a"][1], "changed_lora_b": ch["lora_b"][0] / ch["lora_b"][1], "changed_heads": ch["heads"][0] / ch["heads"][1], "master_norm": sum(float(np.sum(v.astype(np.float64) ** 2)) for v in new_np.values()) ** 0.5, "chain_rel_max": max(chain_rel), "velocity_rms_student": float(np.mean(s_rms)), "velocity_rms_teacher": float(np.mean(t_rms)), "finite": bool(finite and state_finite), "seconds": time.perf_counter() - t_u}
            metrics.append(row)
            with csv_path.open("a", newline="") as f:
                wr = csv.DictWriter(f, fieldnames=list(row))
                if u == 0:
                    wr.writeheader()
                wr.writerow(row)
            print("[update]", json.dumps(row), flush=True)
            if not row["finite"]:
                stop_reason = "non-finite gradient or optimizer state"
                break
            del eff, intended
            master, theta = new_np, theta_new
            state_box["theta"] = theta
            u += 1
            if u == 1:
                release("after_first_update")
                after = diag_eval(theta, prep, 510000, N_DIAG * 2)
                R["sanity"] = {"fresh_noise_paired": paired(sanity_before[:, 0], after[:, 0], len(prep)), "groups": {n: paired(sanity_before[:, i], after[:, i], len(prep)) for i, n in enumerate(MASKS, 1)}, "forward_changed_lora_b": row["changed_lora_b"], "grad_lora_b": row["grad_lora_b"], "grad_heads": row["grad_heads"], "routing_vlm_nonzero": R["routing_vlm_nonzero"], "teacher_unchanged": tree_sha(nnx.split(teacher)[1]) == t_sha, "base_unchanged": tree_sha(nnx.split(base_model)[1]) == base_sha, "frozen_student_unchanged": tree_sha((vlm, other)) == frozen_sha, "cos_forward_master": row["cos_forward_master"]}
                s_ = R["sanity"]
                s_["gates"] = {"forward_changed": s_["forward_changed_lora_b"] > 0, "lora_b_gradient": s_["grad_lora_b"] > 0, "vlm_zero": s_["routing_vlm_nonzero"] == 0, "immutability": s_["teacher_unchanged"] and s_["base_unchanged"] and s_["frozen_student_unchanged"], "fresh_noise_kl_below_zero": s_["fresh_noise_paired"]["ci95_obs"][1] < 0, "direct_and_state_finite": all(np.isfinite(v) for v in (R["direct_vs_state"][0]["direct"], R["direct_vs_state"][0]["state"])) and R["direct_vs_state"][0]["state"] > 0}
                save()
                print("[sanity]", json.dumps(s_), flush=True)
                if not all(s_["gates"].values()):
                    stop_reason = "one-batch gate failed: " + ",".join(k for k, v in s_["gates"].items() if not v)
                    break
                if a.sanity_only:
                    stop_reason = "sanity only"
                    break
            if u % DIAG_EVERY == 0:
                d = diag_eval(theta, diag_prep, 500000, N_DIAG)
                R["diagnostic"]["history"].append({"update": u, "mean": float(d[:, 0].mean())})
                if d[:, 0].mean() > 1.5 * diag0[:, 0].mean():
                    stop_reason = "diagnostic KL above 1.5x initial"
                    break
            if u in CKPT_AT:
                np.savez(out / "checkpoints" / f"adapter_{u:03d}.npz", **{k.replace("/", "__"): master[k] for k in keys})
            save()
    finally:
        with srv.lock:
            srv.stop = True
        try:
            sim.wait(timeout=30)
        except subprocess.TimeoutExpired:
            sim.kill()
    if u and not (out / "checkpoints" / f"adapter_{u:03d}.npz").exists():
        np.savez(out / "checkpoints" / f"adapter_{u:03d}.npz", **{k.replace("/", "__"): master[k] for k in keys})
    if u:
        d = diag_eval(theta, diag_prep, 500000, N_DIAG)
        R["diagnostic"]["final"] = {"updates": u, "full": paired(diag0[:, 0], d[:, 0], len(diag_prep)), **{n: paired(diag0[:, i], d[:, i], len(diag_prep)) for i, n in enumerate(MASKS, 1)}}
    R["completed_updates"], R["stop_reason"], R["training_seconds"] = u, stop_reason, time.perf_counter() - t_train
    R["end"] = {"teacher_unchanged": tree_sha(nnx.split(teacher)[1]) == t_sha, "base_unchanged": tree_sha(nnx.split(base_model)[1]) == base_sha, "frozen_student_unchanged": tree_sha((vlm, other)) == frozen_sha, "peak_gpu_gib": jax.devices()[0].memory_stats()["peak_bytes_in_use"] / 2**30, "rss_gib": T_rss(), "host_peak_gib": next(int(l.split()[1]) for l in open("/proc/self/status") if l.startswith("VmHWM")) / 2**20, "sample_seconds_mean": float(np.mean(sample_seconds[1:])) if len(sample_seconds) > 1 else None}
    save()
    print("[end]", json.dumps({k: R[k] for k in ("completed_updates", "stop_reason", "training_seconds", "end")}), json.dumps(R["diagnostic"].get("final", {}).get("full")), flush=True)


if __name__ == "__main__":
    main()
