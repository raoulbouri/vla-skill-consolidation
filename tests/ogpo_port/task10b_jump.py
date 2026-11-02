"""Task 10B: where does the kernel-only jump enter? Plain eager evaluations on both sides; FP32 control B. Diagnostic only."""
from __future__ import annotations
import gc, json
import task10b as T
import jax, jax.numpy as jnp, numpy as np
pv = T.pv

def main():
    import einops
    from openpi.models.pi0 import make_attn_mask
    from task10a_derivatives import promote_models
    H = T.helpers(); nnx, flat = H["nnx"], H["_flat"]
    e = H["load_real_env"]()
    teacher = e["model"]
    student, _ = H["make_student"](teacher, H["ALPHA"], H["PERT_SEED"])
    obs = e["observation"](H["OBS_IDS"])
    t32, s32 = promote_models(e["cfg"].checkpoint.openpi_config, teacher, student)
    sc = pv.build_prefix(s32, obs)
    x0, _ = H["injected"](H["NOISE_SEED"], 1); x = jnp.asarray(x0)
    gd, theta, rest = nnx.split(s32, H["in_groups"](H["TRAINABLE"]), ...)
    print("[dtypes]", {k: str(flat(theta)[k].dtype) for k in ("action_out_proj/kernel", "action_out_proj/bias")}, "out_features", s32.action_out_proj.out_features, "module dtype", s32.action_out_proj.dtype, s32.action_out_proj.param_dtype, flush=True)
    def stages(th):
        m = nnx.merge(gd, th, rest)
        tok, smask, ar, cond = m.embed_suffix(sc.observation, x, jnp.ones((1,), jnp.float32))
        pa = einops.repeat(sc.prefix_mask, "b p -> b s p", s=tok.shape[1])
        full = jnp.concatenate([pa, make_attn_mask(smask, ar)], axis=-1)
        pos = jnp.sum(sc.prefix_mask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1
        (_, out), _ = m.PaliGemma.llm([None, tok], mask=full, positions=pos, kv_cache=sc.kv_cache, adarms_cond=[None, cond])
        a = out[:, -m.action_horizon:]
        return {"tok": tok, "cond": cond, "a": a, "v": m.action_out_proj(a)}
    leaves, treedef = jax.tree.flatten(theta)
    by_id = {id(v): k for k, v in flat(theta).items()}; names = [by_id[id(l)] for l in leaves]
    th = {k: np.asarray(v, np.float32) for k, v in flat(theta).items()}
    def build(arrs): return jax.tree.unflatten(treedef, [jnp.asarray(arrs[k], jnp.float32) for k in names])
    g = jax.grad(lambda t: jnp.sum(stages(t)["v"] ** 2))(theta)
    gk = np.asarray(flat(g)["action_out_proj/kernel"], np.float64); gk /= np.linalg.norm(gk)
    base = {k: np.asarray(v, np.float64) for k, v in stages(theta).items()}
    again = {k: np.asarray(v, np.float64) for k, v in stages(build(th)).items()}
    print("[repeat base bitwise]", {k: bool(np.array_equal(base[k], again[k])) for k in base}, flush=True)
    K = th["action_out_proj/kernel"].astype(np.float64); b = th["action_out_proj/bias"].astype(np.float64)
    print("[hand projection base] max|v - a@K - b| float64", float(np.abs(base["a"] @ K + b - base["v"]).max()), "a rms", float(np.sqrt(np.mean(base["a"] ** 2))), "a max", float(np.abs(base["a"]).max()), flush=True)
    raw = stages(theta)
    print("[stage dtypes]", {k: str(v.dtype) for k, v in raw.items()}, flush=True)
    np.savez(H["ROOT"] / "outputs/ogpo_port/task10b/dot_case.npz", a=np.asarray(raw["a"]), v=np.asarray(raw["v"]), K=th["action_out_proj/kernel"], b=th["action_out_proj/bias"], gk=gk.astype(np.float32))
    res = []
    for h in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3):
        p = dict(th); p["action_out_proj/kernel"] = (K + h * gk).astype(np.float32)
        dK = p["action_out_proj/kernel"].astype(np.float64) - K
        s = {k: np.asarray(v, np.float64) for k, v in stages(build(p)).items()}
        hand = base["a"] @ dK
        row = {"h": h, "dK_norm": float(np.linalg.norm(dK)), "dK_max": float(np.abs(dK).max()), "changed": int(np.count_nonzero(dK)), "d_tok": float(np.abs(s["tok"] - base["tok"]).max()), "d_cond": float(np.abs(s["cond"] - base["cond"]).max()),
               "d_a_max": float(np.abs(s["a"] - base["a"]).max()), "d_a_rel": float(np.linalg.norm(s["a"] - base["a"]) / np.linalg.norm(base["a"])), "d_v_norm": float(np.linalg.norm(s["v"] - base["v"])), "hand_dv_norm_on_unchanged_a": float(np.linalg.norm(hand)),
               "v_vs_hand_with_new_a_max": float(np.abs(s["a"] @ p["action_out_proj/kernel"].astype(np.float64) + b - s["v"]).max())}
        res.append(row); print("[kernel step]", json.dumps(row), flush=True)
    (H["ROOT"] / "outputs/ogpo_port/task10b/jump.json").write_text(json.dumps(res, indent=1)); print("[end]", flush=True)

if __name__ == "__main__":
    main()
