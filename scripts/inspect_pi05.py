#!/usr/bin/env python
"""Inspect pi0.5-LIBERO: tensor shapes at every stage, the timestep conditioning, and the flow-matching behaviour.

Designed for the IRL-project policy env. Copy it to IRL-project/scripts/inspect_pi05.py and run:

    scripts/py-policy scripts/inspect_pi05.py --quick                    # CPU (py-policy forces CPU), small settings
    CLVLA_USE_GPU=1 scripts/py-policy scripts/inspect_pi05.py            # GPU 1; refuses if another user is on it

It loads the exact official pi05_libero checkpoint (clvla.policy.load_official_policy, with its compatibility gate),
takes one real demo frame of manifest task 0 (LIBERO-90 #78), and produces, under
$CLVLA_OUTPUT_ROOT/inspect/<stamp>_pi05_flow/ :

  shapes.md / shapes.json   every tensor from the client request to the final (10, 7) action
  fig1_inputs.png           the three camera slots, masks and prompt tokens as the model sees them
  fig2_time_embedding.png   how a timestep t becomes the conditioning vector (sinusoid -> MLP -> adaRMS)
  fig3_denoising.png        x_t, predicted velocity and implied clean action at each of the 10 Euler steps
  fig4_flow_curves.png      per-dimension noise->action paths and the one-step clean-action estimate vs t
  fig5_timestep_sweep.png   TRAINING VIEW: feed x_t = t*eps + (1-t)*A for many t, compare with the true velocity
  fig6_sensitivity.png      what changes the output: noise seed, prompt, images, state
  fig7_num_steps.png        (--steps-study) Euler step count vs result
  tensors.npz, summary.json raw arrays and key numbers

Important facts this script lets you verify (all from the pinned OpenPI source, models/pi0.py):
  * there are NO vocabulary logits in the action path: the network outputs a VELOCITY field v(x_t, t) of shape (B,10,32)
  * training: x_t = t*eps + (1-t)*A, target u = eps - A, loss = mean((v - u)^2), t ~ 0.001 + 0.999*Beta(1.5, 1)
  * inference: Euler steps from t=1 (noise) to t=0 (action), dt = -1/10, x <- x + dt * v
  * pi0.5 injects t ONLY through adaRMS conditioning of the action expert; the 8-D state is not consumed (checked here)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(os.environ.get("CLVLA_ROOT") or Path(__file__).resolve().parents[1])
IMG_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
REAL = 7  # LIBERO uses the first 7 of the 32 padded action dimensions
D_NAMES = ["dx", "dy", "dz", "droll", "dpitch", "dyaw", "gripper"]


# ---------------------------------------------------------------------------------------------------------------------
# bookkeeping


class Recorder:
    """Collects (stage, name, shape, dtype, min/max/mean) rows and prints them as they arrive."""

    def __init__(self):
        self.rows = []

    def add_shape(self, stage, name, shape, dtype):
        self._push({"stage": stage, "name": name, "shape": list(shape), "dtype": str(dtype)})

    def add(self, stage, name, x):
        if isinstance(x, (str, bytes)):
            self._push({"stage": stage, "name": name, "shape": [], "dtype": "str",
                        "value": x if isinstance(x, str) else x.decode(errors="replace")})
            return
        arr = np.asarray(x)
        dtype = arr.dtype.name
        if dtype == "bfloat16":
            arr = arr.astype(np.float32)
        row = {"stage": stage, "name": name, "shape": list(arr.shape), "dtype": dtype}
        if arr.dtype.kind in "fiu" and arr.size:
            row.update(min=float(arr.min()), max=float(arr.max()), mean=float(arr.astype(np.float64).mean()))
        elif arr.dtype.kind == "b":
            row["true_count"] = int(arr.sum())
        self._push(row)

    def _push(self, row):
        self.rows.append(row)
        s = f"[{row['stage']}] {row['name']:<46} {str(tuple(row['shape'])):<20} {row['dtype']:<9}"
        if "min" in row:
            s += f" range [{row['min']:.3g}, {row['max']:.3g}] mean {row['mean']:.3g}"
        if "true_count" in row:
            s += f" true={row['true_count']}"
        if "value" in row:
            s += f" {row['value']!r}"
        print(s, flush=True)

    def to_markdown(self):
        lines = ["| stage | tensor | shape | dtype | min | max | mean |", "|---|---|---|---|---|---|---|"]
        for r in self.rows:
            f = lambda k: f"{r[k]:.4g}" if k in r else ""  # noqa: E731
            shape = str(tuple(r["shape"])) if r["shape"] or r["dtype"] != "str" else r.get("value", "")
            lines.append(f"| {r['stage']} | `{r['name']}` | {shape} | {r['dtype']} | {f('min')} | {f('max')} | {f('mean')} |")
        return "\n".join(lines)


def flatten(tree, prefix=""):
    if isinstance(tree, dict):
        for k, v in tree.items():
            yield from flatten(v, f"{prefix}/{k}" if prefix else str(k))
    else:
        yield prefix, tree


def signature(x):
    if isinstance(x, (str, bytes)):
        return ("str", x)
    a = np.asarray(x)
    return (a.shape, a.dtype.name)


def section(name, fn, results):
    print(f"\n{'=' * 100}\n=== {name}\n{'=' * 100}", flush=True)
    t0 = time.time()
    try:
        results[name] = fn()
        print(f"--- {name}: done in {time.time() - t0:.1f}s", flush=True)
    except Exception:  # noqa: BLE001 - one failing section must not lose the others
        traceback.print_exc()
        results[name] = {"error": traceback.format_exc()[-2000:]}


# ---------------------------------------------------------------------------------------------------------------------
# model access


def find_model(policy):
    for v in vars(policy).values():
        if hasattr(v, "embed_prefix") and hasattr(v, "embed_suffix"):
            return v
    raise RuntimeError("could not find the Pi0 module inside the OpenPI Policy object")


def obs_from(d, batch=1):
    """Dict from OpenPI's transforms -> model Observation with a leading batch dim (identical copies if batch > 1)."""
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model

    keep = ("image", "image_mask", "state", "tokenized_prompt", "tokenized_prompt_mask")
    d = {k: d[k] for k in keep if k in d}
    d = jax.tree.map(lambda x: jnp.repeat(jnp.asarray(x)[None], batch, axis=0), d)
    return _model.Observation.from_dict(d)


class Flow:
    """OpenPI's Pi0.sample_actions, unrolled eagerly so every step can be observed.

    Mirrors models/pi0.py (pinned commit): the prefix (images + prompt) is processed ONCE into a KV cache; each step only
    runs the action expert on the 10 noisy-action tokens, conditioned on t through adaRMS.
    """

    def __init__(self, model, obs, rec=None):
        import jax
        import jax.numpy as jnp

        from openpi.models import model as _model
        from openpi.models.pi0 import make_attn_mask

        self.m, self.jnp, self.make_attn_mask = model, jnp, make_attn_mask
        self.obs = _model.preprocess_observation(None, obs, train=False)
        self.B = self.obs.state.shape[0]
        prefix_tokens, self.prefix_mask, prefix_ar = model.embed_prefix(self.obs)
        attn = make_attn_mask(self.prefix_mask, prefix_ar)
        positions = jnp.cumsum(self.prefix_mask, axis=1) - 1
        _, self.kv = model.PaliGemma.llm([prefix_tokens, None], mask=attn, positions=positions)
        if rec is not None:
            for name in IMG_KEYS:
                rec.add("3 preprocess", f"images[{name}] (float, [-1,1])", self.obs.images[name])
            rec.add("4 embed_prefix", "prefix_tokens (images -> SigLIP, prompt -> embed)", prefix_tokens)
            rec.add("4 embed_prefix", "prefix_mask (False = padding / masked camera)", self.prefix_mask)
            rec.add("4 embed_prefix", "prefix_ar_mask (all False = full attention)", prefix_ar)
            rec.add("4 embed_prefix", "prefix attention mask", attn)
            for i, leaf in enumerate(jax.tree_util.tree_leaves(self.kv)):
                rec.add_shape("5 prefix pass", f"kv_cache leaf {i} (layers, batch, prefix_len, kv_heads, head_dim)",
                              leaf.shape, leaf.dtype)

    def velocity(self, x_t, t, rec=None):
        """v_theta(obs, x_t, t): x_t (B, 10, 32), t scalar or (B,). Returns float32 (B, 10, 32)."""
        jnp, m = self.jnp, self.m
        tvec = jnp.broadcast_to(jnp.asarray(t, jnp.float32), (self.B,))
        suffix_tokens, suffix_mask, suffix_ar, cond = m.embed_suffix(self.obs, x_t, tvec)
        suffix_attn = self.make_attn_mask(suffix_mask, suffix_ar)
        prefix_attn = jnp.broadcast_to(self.prefix_mask[:, None, :],
                                       (self.B, suffix_tokens.shape[1], self.prefix_mask.shape[1]))
        full = jnp.concatenate([prefix_attn, suffix_attn], axis=-1)
        positions = jnp.sum(self.prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1
        (_, suffix_out), _ = m.PaliGemma.llm([None, suffix_tokens], mask=full, positions=positions, kv_cache=self.kv,
                                             adarms_cond=[None, cond])
        v = m.action_out_proj(suffix_out[:, -m.action_horizon:]).astype(jnp.float32)
        if rec is not None:
            rec.add("6 denoise step", "x_t  (noisy action chunk in)", x_t)
            rec.add("6 denoise step", "t    (timestep)", tvec)
            rec.add("6 denoise step", "suffix_tokens = action_in_proj(x_t)  (B,10,1024)", suffix_tokens)
            rec.add("6 denoise step", "adarms_cond = swish(MLP(sincos(t)))  (B,1024)", cond)
            rec.add("6 denoise step", "suffix attention mask (10 x 10, bidirectional)", suffix_attn)
            rec.add("6 denoise step", "full attention mask (10 x [prefix+10])", full)
            rec.add("6 denoise step", "action-expert output hidden state (B,10,1024)", suffix_out)
            rec.add("6 denoise step", "v_t = action_out_proj(hidden)  (B,10,32)", v)
        return v

    def sample(self, noise, num_steps=10, record_first=None):
        dt = -1.0 / num_steps
        x, t = self.jnp.asarray(noise, self.jnp.float32), 1.0
        xs, vs, ts = [x], [], []
        for i in range(num_steps):
            v = self.velocity(x, t, rec=record_first if i == 0 else None)
            ts.append(t)
            vs.append(v)
            x = x + dt * v
            t = t + dt
            xs.append(x)
        return xs, vs, ts


def to_real(policy, state_norm, x0):
    """Normalised, padded chunk (10, 32) -> real LIBERO action units (10, 7), via OpenPI's own output transforms."""
    out = policy._output_transform({"state": np.array(state_norm, dtype=np.float32), "actions": np.array(x0, dtype=np.float32)})
    return np.asarray(out["actions"])


# ---------------------------------------------------------------------------------------------------------------------
# data


def make_request(ep, t, instruction, resize=224):
    """The request OpenPI's LIBERO client sends: rotate 180, resize_with_pad 224, 8-D state, prompt."""
    from openpi_client import image_tools

    from clvla.data import episode_state, rotate180

    def prep(img):
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rotate180(img), resize, resize))

    return {"observation/image": prep(ep["obs/agentview_rgb"][t]),
            "observation/wrist_image": prep(ep["obs/eye_in_hand_rgb"][t]),
            "observation/state": episode_state(ep)[t].astype(np.float64), "prompt": instruction}


# ---------------------------------------------------------------------------------------------------------------------
# plotting


def heat(ax, M, title, vmax=None):
    vmax = float(vmax) if vmax else max(float(np.abs(M).max()), 1e-6)
    ax.imshow(M, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.axvline(REAL - 0.5, color="k", lw=0.8)
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])


def save(ctx, fig, name):
    if ctx.plt is None:
        return
    path = ctx.out / name
    fig.savefig(path, dpi=130, bbox_inches="tight")
    ctx.plt.close(fig)
    print(f"    saved {path}", flush=True)


# ---------------------------------------------------------------------------------------------------------------------
# sections


def sec_inputs(ctx):
    """Client request -> each OpenPI input transform -> what the model receives."""
    import jax

    rec, tf = ctx.rec, ctx.policy._input_transform
    for name, leaf in flatten(ctx.request):
        rec.add("0 client request", name, leaf)
    data = jax.tree.map(lambda x: x, ctx.request)
    prev = {k: signature(v) for k, v in flatten(data)}
    for i, t in enumerate(tf.transforms):
        data = t(data)
        cur = {k: v for k, v in flatten(data)}
        gone = sorted(set(prev) - set(cur))
        if gone:
            print(f"    [1.{i} {type(t).__name__}] removed: {gone}")
        for k, v in cur.items():
            if prev.get(k) != signature(v):
                rec.add(f"1.{i} {type(t).__name__}", k, v)
        prev = {k: signature(v) for k, v in cur.items()}
    ctx.infer_d = data

    pieces = None
    for t in tf.transforms:
        sp = getattr(getattr(t, "tokenizer", None), "_tokenizer", None)
        if sp is not None:
            ids = np.asarray(data["tokenized_prompt"])[np.asarray(data["tokenized_prompt_mask"])]
            pieces = [sp.id_to_piece(int(i)) for i in ids]
    if ctx.plt is not None:
        plt = ctx.plt
        fig, ax = plt.subplots(1, 4, figsize=(17, 4.4), gridspec_kw={"width_ratios": [1, 1, 1, 1.4]})
        for a, k in zip(ax[:3], IMG_KEYS):
            img = np.asarray(data["image"][k])
            img = img / 255.0 if img.dtype == np.uint8 else (img + 1.0) / 2.0
            a.imshow(np.clip(img, 0, 1))
            a.set_title(f"{k}\nmask={bool(np.asarray(data['image_mask'][k]))}  shape={img.shape}", fontsize=9)
            a.axis("off")
        m = np.asarray(data["tokenized_prompt_mask"])
        ax[3].imshow(m[None, :], aspect="auto", cmap="Greys", vmin=0, vmax=1)
        ax[3].set_yticks([])
        ax[3].set_xlabel("token position (0..199)")
        ax[3].set_title(f"tokenized_prompt_mask: {int(m.sum())} real tokens / {m.size}\n{pieces}", fontsize=8)
        fig.suptitle("Model inputs for one query (after OpenPI's input transforms; right wrist camera is empty and masked)")
        save(ctx, fig, "fig1_inputs.png")
    return {"prompt_tokens": pieces, "n_prompt_tokens": int(np.asarray(data["tokenized_prompt_mask"]).sum())}


def sec_time_embedding(ctx):
    """How a scalar t is turned into the adaRMS conditioning vector."""
    import flax.nnx as nnx
    import jax.numpy as jnp

    from openpi.models.pi0 import posemb_sincos

    m = ctx.model
    ts = jnp.linspace(0.0, 1.0, 101)
    pe = posemb_sincos(ts, m.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
    cond = nnx.swish(m.time_mlp_out(nnx.swish(m.time_mlp_in(pe)))).astype(jnp.float32)
    pe, cond = np.asarray(pe), np.asarray(cond)
    ctx.rec.add("2 time embedding", "t grid", np.asarray(ts))
    ctx.rec.add("2 time embedding", "sincos(t)  (101, 1024)  periods 4e-3 .. 4.0", pe)
    ctx.rec.add("2 time embedding", "adarms_cond(t)  (101, 1024)", cond)
    c = cond / (np.linalg.norm(cond, axis=1, keepdims=True) + 1e-9)
    sim = c @ c.T
    ctx.npz.update(time_grid=np.asarray(ts), time_sincos=pe, time_adarms=cond, time_cos_sim=sim)
    if ctx.plt is not None:
        plt = ctx.plt
        fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
        im0 = ax[0].imshow(pe.T, aspect="auto", cmap="coolwarm", extent=[0, 1, pe.shape[1], 0])
        ax[0].set_title("sincos(t): 1024 dims (sin | cos halves)")
        ax[0].set_xlabel("t")
        plt.colorbar(im0, ax=ax[0])
        im1 = ax[1].imshow(cond.T, aspect="auto", cmap="coolwarm", extent=[0, 1, cond.shape[1], 0],
                           vmin=-np.abs(cond).max(), vmax=np.abs(cond).max())
        ax[1].set_title("adaRMS conditioning = swish(MLP(sincos(t)))")
        ax[1].set_xlabel("t")
        plt.colorbar(im1, ax=ax[1])
        im2 = ax[2].imshow(sim, cmap="viridis", extent=[0, 1, 1, 0])
        ax[2].set_title("cosine similarity of conditioning vectors\n(how distinguishable are two noise levels?)")
        ax[2].set_xlabel("t")
        ax[2].set_ylabel("t")
        plt.colorbar(im2, ax=ax[2])
        fig.suptitle("How the model is told the noise level: t -> sinusoid -> 2-layer MLP -> adaRMS scale/shift in every action-expert layer")
        save(ctx, fig, "fig2_time_embedding.png")
    return {"cos_sim_t0.0_vs_1.0": float(sim[0, -1]), "cos_sim_t0.5_vs_0.6": float(sim[50, 60])}


def sec_denoise(ctx):
    """The 10-step Euler integration from noise (t=1) to the action chunk (t=0)."""
    import jax

    obs = obs_from(ctx.infer_d)
    flow = Flow(ctx.model, obs, rec=ctx.rec)
    H, D = ctx.model.action_horizon, ctx.model.action_dim
    noise = jax.random.normal(jax.random.key(ctx.args.seed), (1, H, D))
    xs, vs, ts = flow.sample(noise, 10, record_first=ctx.rec)
    x0 = np.asarray(xs[-1][0])
    official = np.asarray(ctx.policy._sample_actions(jax.random.key(0), obs, noise=noise)[0])
    diff = float(np.abs(official - x0).max())
    print(f"    manual eager loop vs OpenPI's jitted sample_actions (same noise): max|diff| = {diff:.3g} "
          "(bf16 + different kernels => small non-zero values are expected)")
    X = np.stack([np.asarray(x[0]) for x in xs])  # (11, 10, 32)
    V = np.stack([np.asarray(v[0]) for v in vs])  # (10, 10, 32)
    T = np.array(ts)
    X0HAT = X[:-1] - T[:, None, None] * V  # clean-action estimate implied by each step: x_t - t * v
    ctx.noise, ctx.flow, ctx.x0 = noise, flow, x0
    ctx.npz.update(den_x=X, den_v=V, den_t=T, den_x0hat=X0HAT, den_noise=np.asarray(noise[0]))
    ctx.real0 = to_real(ctx.policy, ctx.infer_d["state"], x0)
    ctx.rec.add("7 output", "x_0 final chunk, normalised, padded (10,32)", x0)
    ctx.rec.add("7 output", "after Unnormalize + LiberoOutputs: real actions (10,7)", ctx.real0)

    if ctx.plt is not None:
        plt = ctx.plt
        # fig3: heatmaps
        fig, ax = plt.subplots(3, 11, figsize=(24, 6.2))
        vx, vv, vh = np.abs(X).max(), np.abs(V).max(), np.abs(X0HAT).max()
        for k in range(11):
            heat(ax[0, k], X[k], f"x_t, t={1 - 0.1 * k:.1f}", vx)
        for k in range(10):
            heat(ax[1, k], V[k], f"v (t={T[k]:.1f})", vv)
            heat(ax[2, k], X0HAT[k], f"x_t - t*v (t={T[k]:.1f})", vh)
        ax[1, 10].axis("off")
        ax[2, 10].axis("off")
        ax[0, 0].set_ylabel("rows: 10 chunk steps\ncols: 32 dims | black line = LIBERO's 7 dims", fontsize=7)
        fig.suptitle("Denoising. Row 1: noisy chunk x_t (t=1 pure noise -> t=0 action). Row 2: predicted velocity v. "
                     "Row 3: one-step estimate of the CLEAN action = x_t - t*v. Right of the black line = 25 padded dims -> ~0")
        save(ctx, fig, "fig3_denoising.png")
        # fig4: curves
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
        tt = np.array([1 - 0.1 * k for k in range(11)])
        for d in range(REAL):
            ax[0].plot(tt, X[:, 0, d], marker="o", ms=3, label=D_NAMES[d])
            ax[1].plot(T, X0HAT[:, 0, d], marker="o", ms=3, label=D_NAMES[d])
            ax[1].axhline(x0[0, d], color=f"C{d}", ls=":", lw=0.8)
        ax[0].set_title("path of x_t for the FIRST chunk step (the one executed first)")
        ax[1].set_title("clean-action estimate x_t - t*v at each step (dotted = final x_0)")
        for a in ax[:2]:
            a.set_xlabel("t   (1 = noise, 0 = action)")
            a.invert_xaxis()
        ax[0].legend(fontsize=7, ncol=2)
        nv = np.linalg.norm(V[:, :, :REAL].reshape(10, -1), axis=1)
        ne = np.linalg.norm((X0HAT - x0[None])[:, :, :REAL].reshape(10, -1), axis=1)
        ax[2].semilogy(T, nv, "o-", label="||v||  (velocity magnitude)")
        ax[2].semilogy(T, ne + 1e-9, "s-", label="||clean estimate - final x_0||")
        ax[2].set_xlabel("t")
        ax[2].invert_xaxis()
        ax[2].legend(fontsize=8)
        ax[2].set_title("how fast the estimate settles (7 real dims)")
        save(ctx, fig, "fig4_flow_curves.png")
    return {"manual_vs_official_max_abs_diff": diff, "final_real_action_step0": ctx.real0[0].tolist(),
            "padded_dims_abs_mean_in_x0": float(np.abs(x0[:, REAL:]).mean())}


def sec_timestep_sweep(ctx):
    """Training view: for a KNOWN clean chunk A, build x_t for many t, query the model, compare with u = eps - A."""
    import jax
    import jax.numpy as jnp

    H, D = ctx.model.action_horizon, ctx.model.action_dim
    K = ctx.args.draws
    if ctx.train_d is not None:
        A = jnp.asarray(ctx.train_d["actions"])[None]  # (1,10,32) normalised + padded demo chunk
        src = "demo action chunk (normalised, padded)"
        obs_d = ctx.train_d
    else:
        A = jnp.asarray(ctx.x0)[None]
        src = "the model's own sampled chunk (no demo data)"
        obs_d = ctx.infer_d
    flowK = Flow(ctx.model, obs_from(obs_d, batch=K))
    eps = jax.random.normal(jax.random.key(ctx.args.seed + 1), (K, H, D))
    grid = ctx.t_grid
    rows, cache = [], {}
    for t in grid:
        x_t = t * eps + (1 - t) * A
        v = flowK.velocity(x_t, float(t))
        u = eps - A
        x0hat = x_t - t * v
        se = np.asarray((v - u) ** 2)
        row = {"t": float(t), "mse_all32": float(se.mean()), "mse_real7": float(se[..., :REAL].mean()),
               "mse_pad25": float(se[..., REAL:].mean()), "zero_velocity_mse": float(np.asarray(u ** 2).mean()),
               "x0hat_abs_err_real7": float(np.abs(np.asarray(x0hat - A))[..., :REAL].mean())}
        rows.append(row)
        cache[float(t)] = (np.asarray(x_t[0]), np.asarray(u[0]), np.asarray(v[0]), np.asarray(x0hat[0]))
        print(f"    t={t:5.3f}  loss(all32)={row['mse_all32']:.4f}  real7={row['mse_real7']:.4f}  "
              f"pad25={row['mse_pad25']:.5f}  predict-zero-velocity={row['zero_velocity_mse']:.3f}  "
              f"|x0hat-A|(real7)={row['x0hat_abs_err_real7']:.3f}", flush=True)
    tg = np.array([r["t"] for r in rows])
    mse = np.array([r["mse_all32"] for r in rows])
    s = (tg - 0.001) / 0.999
    pdf = 1.5 * np.sqrt(np.clip(s, 0, None)) / 0.999  # density of t = 0.001 + 0.999*Beta(1.5, 1)
    trapz = getattr(np, "trapezoid", None) or np.trapz
    mc = float(trapz(mse * pdf, tg) / trapz(pdf, tg))
    out = {"source_of_A": src, "mse_by_t": rows, "beta_weighted_loss_estimate": mc}
    try:
        n = 32
        loss = ctx.model.compute_loss(jax.random.key(3), obs_from(obs_d, batch=n), jnp.repeat(A, n, axis=0), train=False)
        out["official_compute_loss_mean(32 random t, eval mode)"] = float(np.asarray(loss).mean())
        print(f"    OpenPI compute_loss (random t~Beta, 32 copies): {out['official_compute_loss_mean(32 random t, eval mode)']:.4f}"
              f"   vs this sweep weighted by the Beta density: {mc:.4f}   (should be the same order; sweep uses {K} noise draws)")
    except Exception as e:  # noqa: BLE001
        out["official_compute_loss_error"] = f"{type(e).__name__}: {e}"
    ctx.npz.update(sweep_t=tg, sweep_mse=mse, sweep_A=np.asarray(A[0]))
    if ctx.plt is not None:
        plt = ctx.plt
        fig, ax = plt.subplots(1, 3, figsize=(18, 4.6))
        ax[0].semilogy(tg, mse, "o-", label="all 32 dims (= training loss at this t)")
        ax[0].semilogy(tg, [r["mse_real7"] for r in rows], "s-", label="7 real dims")
        ax[0].semilogy(tg, [r["mse_pad25"] + 1e-8 for r in rows], "^-", label="25 padded dims")
        ax[0].semilogy(tg, [r["zero_velocity_mse"] for r in rows], "k--", label="baseline: predict v = 0")
        tw = ax[0].twinx()
        tw.fill_between(tg, 0, pdf, color="grey", alpha=0.15)
        tw.set_ylabel("sampling density of t in training (Beta(1.5,1))", fontsize=7)
        ax[0].set_xlabel("t   (0.001 = almost clean, 1 = pure noise)")
        ax[0].set_ylabel("MSE(v_pred, eps - A)")
        ax[0].legend(fontsize=7)
        ax[0].set_title("flow-matching loss as a function of the timestep")
        ax[1].plot(tg, [r["x0hat_abs_err_real7"] for r in rows], "o-")
        ax[1].set_xlabel("t")
        ax[1].set_ylabel("mean |x_t - t*v_pred - A|  (7 dims, normalised units)")
        ax[1].set_title("error of the one-step clean-action estimate")
        ax[2].axis("off")
        ax[2].text(0, 0.95, "Read it like this:\n"
                   "- small t: x_t is almost A, v = eps - A is hard to infer from x_t\n  (eps is almost invisible), so the loss is large\n"
                   "- large t: x_t is almost noise; A must come from the images+prompt,\n  so error = how well the policy knows the action\n"
                   "- the padded dims have target ~0: they are learned to be 0\n"
                   f"- Beta-weighted estimate {mc:.4f}\n  vs OpenPI compute_loss {out.get('official_compute_loss_mean(32 random t, eval mode)', float('nan')):.4f}",
                   va="top", fontsize=9)
        save(ctx, fig, "fig5_timestep_sweep.png")
        # panel: x_t, u, v_pred, x0hat for three noise levels
        picks = [min(cache, key=lambda g: abs(g - w)) for w in (0.9, 0.5, 0.1)]
        fig, ax = plt.subplots(3, 5, figsize=(17, 6.2))
        vmax = max(np.abs(np.asarray(A)).max(), 1.0)
        for r, t in enumerate(picks):
            x_t, u, v, x0h = cache[t]
            heat(ax[r, 0], np.asarray(A[0]), "A (true clean chunk)", vmax)
            heat(ax[r, 1], x_t, f"x_t at t={t:.2f}", vmax)
            heat(ax[r, 2], u, "true velocity u = eps - A", np.abs(u).max())
            heat(ax[r, 3], v, "predicted v", np.abs(u).max())
            heat(ax[r, 4], x0h, "x_t - t*v (clean estimate)", vmax)
        fig.suptitle("Same demo chunk A at three noise levels: top t~0.9 (mostly noise), middle t~0.5, bottom t~0.1")
        save(ctx, fig, "fig5b_noise_levels.png")
    return out


def sec_sensitivity(ctx):
    """What moves the output: noise seed vs prompt vs images vs state. Same noise for every variant."""
    import jax

    H, D = ctx.model.action_horizon, ctx.model.action_dim
    std = np.array(ctx.norm_stats["actions"]["std"], dtype=np.float64)
    req = ctx.request

    def run(request, noise):
        d = ctx.policy._input_transform(jax.tree.map(lambda x: x, request))
        flow = Flow(ctx.model, obs_from(d))
        xs, _, _ = flow.sample(noise, 10)
        return to_real(ctx.policy, d["state"], np.asarray(xs[-1][0]))

    base = run(req, ctx.noise)
    img, wr = req["observation/image"], req["observation/wrist_image"]
    zero = lambda a: np.zeros_like(a)  # noqa: E731
    variants = {
        f"prompt -> '{ctx.args.alt_prompt}'": {**req, "prompt": ctx.args.alt_prompt},
        "prompt -> '' (empty)": {**req, "prompt": ""},
        "base image blanked": {**req, "observation/image": zero(img)},
        "wrist image blanked": {**req, "observation/wrist_image": zero(wr)},
        "both images blanked": {**req, "observation/image": zero(img), "observation/wrist_image": zero(wr)},
        "base image shifted 24 px": {**req, "observation/image": np.roll(img, 24, axis=1)},
        "state + 0.5 (all 8 dims)": {**req, "observation/state": req["observation/state"] + 0.5},
        "state zeroed": {**req, "observation/state": np.zeros_like(req["observation/state"])},
    }
    results = {}
    for name, r in variants.items():
        a = run(r, ctx.noise)
        results[name] = a
        print(f"    {name:<44} mean |delta| = {np.mean(np.abs(a - base) / std):.4f} std-units "
              f"(step 0 only: {np.mean(np.abs(a[0] - base[0]) / std):.4f})", flush=True)
    seeds = []
    for sd in range(ctx.args.seeds):
        n = jax.random.normal(jax.random.key(1000 + sd), (1, H, D))
        seeds.append(run(req, n))
    seeds = np.stack(seeds)
    pair = [np.mean(np.abs(seeds[i] - seeds[j]) / std) for i in range(len(seeds)) for j in range(i + 1, len(seeds))]
    spread = float(np.mean(pair)) if pair else float("nan")
    print(f"    different NOISE seeds, same input ({len(seeds)} seeds): mean pairwise |delta| = {spread:.4f} std-units")
    ctx.npz.update(sens_base=base, sens_seeds=seeds, **{f"sens_{i}": v for i, v in enumerate(results.values())})
    table = {k: float(np.mean(np.abs(v - base) / std)) for k, v in results.items()}
    if ctx.plt is not None:
        plt = ctx.plt
        fig = plt.figure(figsize=(18, 7))
        gs = fig.add_gridspec(2, 5, width_ratios=[1.6, 1, 1, 1, 1])
        axb = fig.add_subplot(gs[:, 0])
        names = list(table)[::-1]
        axb.barh(names, [table[n] for n in names], color="C0")
        axb.axvline(spread, color="r", ls="--", label=f"noise-seed spread = {spread:.3f}")
        axb.set_xlabel("mean |change in action| / action std  (same noise)")
        axb.legend(fontsize=8)
        axb.set_title("what changes the action chunk", fontsize=10)
        for d in range(REAL):
            a = fig.add_subplot(gs[d // 4, 1 + d % 4])
            for s_ in seeds:
                a.plot(s_[:, d], color="grey", lw=0.6, alpha=0.6)
            a.plot(base[:, d], "k", lw=2, label="baseline")
            first = list(results)[0]
            a.plot(results[first][:, d], "r", lw=1.2, label="alt prompt")
            a.plot(results["base image blanked"][:, d], "b", lw=1.2, label="base blank")
            a.plot(results["wrist image blanked"][:, d], "g", lw=1.2, label="wrist blank")
            a.set_title(D_NAMES[d], fontsize=9)
            a.set_xlabel("chunk step")
            if d == 0:
                a.legend(fontsize=6)
        fig.suptitle("Sensitivity. grey = other noise seeds (genuine multimodality); coloured = input changes with the SAME noise")
        save(ctx, fig, "fig6_sensitivity.png")
    return {"mean_abs_change_in_std_units": table, "noise_seed_pairwise_spread": spread}


def sec_steps(ctx):
    """Euler step count vs result (reference = 50 steps)."""
    flow = ctx.flow
    std = np.array(ctx.norm_stats["actions"]["std"], dtype=np.float64)
    outs = {}
    for n in (1, 2, 4, 5, 10, 20, 50):
        xs, _, _ = flow.sample(ctx.noise, n)
        outs[n] = to_real(ctx.policy, ctx.infer_d["state"], np.asarray(xs[-1][0]))
        print(f"    num_steps={n:<3} done", flush=True)
    ref = outs[50]
    err = {n: float(np.mean(np.abs(a - ref) / std)) for n, a in outs.items()}
    print("    mean |delta| to the 50-step result (std-units):", {k: round(v, 4) for k, v in err.items()})
    if ctx.plt is not None:
        plt = ctx.plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
        ax[0].loglog(list(err), [max(v, 1e-6) for v in err.values()], "o-")
        ax[0].axvline(10, color="r", ls=":", label="OpenPI default")
        ax[0].set_xlabel("Euler steps")
        ax[0].set_ylabel("mean |delta| vs 50 steps (std units)")
        ax[0].legend()
        for n, a in outs.items():
            ax[1].plot(a[:, 0], label=f"{n} steps")
        ax[1].set_title("dx over the chunk")
        ax[1].legend(fontsize=7)
        save(ctx, fig, "fig7_num_steps.png")
    return {"error_vs_50_steps": err}


# ---------------------------------------------------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--task-position", type=int, default=0, help="manifest position (0 = LIBERO-90 #78)")
    ap.add_argument("--demo", default="demo_0")
    ap.add_argument("--frame", type=int, default=40)
    ap.add_argument("--alt-prompt", default="open the microwave")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=8, help="noise seeds for the multimodality check")
    ap.add_argument("--draws", type=int, default=4, help="noise draws per timestep in the sweep")
    ap.add_argument("--quick", action="store_true", help="fewer seeds/timesteps/draws (use this on CPU)")
    ap.add_argument("--steps-study", action="store_true")
    ap.add_argument("--synthetic", action="store_true", help="random inputs instead of a demo frame (no HDF5 needed)")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.draws = 3, 2

    import jax

    if jax.default_backend() == "gpu":
        r = subprocess.run([str(ROOT / "scripts/gpu_guard.sh")], capture_output=True, text=True)
        if r.returncode != 0:
            sys.exit(f"GPU guard refused (nothing was touched): {r.stdout.strip()}")
    print(f"jax backend: {jax.default_backend()}  devices: {jax.devices()}")

    plt = None
    if not args.no_plots:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt  # noqa: F811
        except ImportError:
            print("matplotlib is not installed in this env: continuing without figures (tables/npz/json only)")

    from clvla.config import load_experiment_config
    from clvla.experiment import utc_stamp
    from clvla.policy import load_official_policy, openpi_training_transforms
    from clvla.tasks import load_manifest

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    manifest = load_manifest(cfg.resolve(cfg.task_manifest))
    task = manifest.by_position(args.task_position)
    out = Path(args.out_dir) if args.out_dir else Path(cfg.output_root) / "inspect" / f"{utc_stamp()}_pi05_flow"
    out.mkdir(parents=True, exist_ok=False)

    print(f"loading pi05_libero from {cfg.checkpoint.local_dir} ...")
    policy, report = load_official_policy(
        cfg.checkpoint.openpi_config, cfg.checkpoint.local_dir, expected_asset_id=cfg.checkpoint.norm_stats_asset_id,
        state_dim=cfg.policy.state_dim, action_dim=cfg.policy.action_dim, action_horizon=cfg.policy.action_horizon)
    print(report)
    model = find_model(policy)
    norm_stats = json.loads((Path(cfg.checkpoint.local_dir) / "assets" / cfg.checkpoint.norm_stats_asset_id
                             / "norm_stats.json").read_text())["norm_stats"]

    train_d = None
    if args.synthetic:
        from openpi.policies.libero_policy import make_libero_example

        request = make_libero_example()
        request["prompt"] = task.instruction
    else:
        from clvla.data import episode_to_openpi_frames, load_episode

        ep = load_episode(Path(cfg.data.root) / task.demo_file, args.demo)
        request = make_request(ep, args.frame, task.instruction)
        transform, _, _ = openpi_training_transforms(cfg.checkpoint.openpi_config, cfg.checkpoint.local_dir)
        frame = episode_to_openpi_frames(ep, task.instruction, cfg.policy.action_horizon, timesteps=[args.frame])[0]
        train_d = transform(dict(frame))

    ctx = SimpleNamespace(args=args, cfg=cfg, policy=policy, model=model, out=out, plt=plt, rec=Recorder(), npz={},
                          request=request, train_d=train_d, norm_stats=norm_stats,
                          t_grid=([0.001, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0] if args.quick else
                                  [0.001, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]))
    results = {}
    section("1. inputs: client request -> OpenPI input transforms", lambda: sec_inputs(ctx), results)
    if "error" in results.get("1. inputs: client request -> OpenPI input transforms", {}):
        sys.exit("input stage failed; later sections depend on it")
    if train_d is not None:  # the training chain and the inference chain must agree on the model inputs
        a = np.asarray(ctx.infer_d["image"]["base_0_rgb"]).astype(np.float32)
        b = np.asarray(train_d["image"]["base_0_rgb"]).astype(np.float32)
        print(f"    check: inference-chain vs training-chain base image max|diff| = {np.abs(a - b).max():.3g}")
    section("2. timestep -> conditioning vector", lambda: sec_time_embedding(ctx), results)
    section("3. denoising: noise -> action in 10 Euler steps", lambda: sec_denoise(ctx), results)
    if "x0" in vars(ctx):
        section("4. timestep sweep (the training objective)", lambda: sec_timestep_sweep(ctx), results)
        section("5. sensitivity: noise vs prompt vs images vs state", lambda: sec_sensitivity(ctx), results)
        if args.steps_study:
            section("6. Euler step count", lambda: sec_steps(ctx), results)

    (out / "shapes.md").write_text(ctx.rec.to_markdown())
    (out / "shapes.json").write_text(json.dumps(ctx.rec.rows, indent=1))
    np.savez_compressed(out / "tensors.npz", **ctx.npz)
    (out / "summary.json").write_text(json.dumps(
        {"task": task.task_id, "demo": args.demo, "frame": args.frame, "synthetic": args.synthetic,
         "backend": jax.default_backend(), "checkpoint_dir": cfg.checkpoint.local_dir,
         "checkpoint_fingerprint_sha256": json.loads((ROOT / cfg.checkpoint.lock_file).read_text())["fingerprint_sha256"],
         "results": results},
        indent=1, default=str))
    print(f"\nall outputs in {out}")


if __name__ == "__main__":
    main()
