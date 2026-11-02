"""Task 10A, Phase 2c: first diverging tensor inside the real execution structure: lax.scan over checkpoint(nothing_saveable) of the
expert block, as gemma.Module does, with each block's intermediates returned as scan outputs. k = 0 inputs. Diagnostic only.
Returning intermediates changes the compiled programme, so this locates the mechanism; it does not reproduce the exact production values.
"""

from __future__ import annotations

import json
import os

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx, traverse_util

from pi05_real_env import ROOT, load_real_env
from task10a_primal import diff, re_expert
from test_pi05_path_kl import ALPHA, PERT_SEED, _flat, make_student, injected
from test_pi05_pathwise_gradient import NOISE_SEED, OBS_IDS

from clvla.ogpo import pi05_velocity as pv

TAG = os.environ.get("TAG", "default")
OUT = ROOT / "outputs" / "ogpo_port" / "task10a"


def main():
    import einops
    from openpi.models import gemma as _gemma
    from openpi.models.pi0 import make_attn_mask

    e = load_real_env()
    student, _ = make_student(e["model"], ALPHA, PERT_SEED)
    obs = e["observation"](OBS_IDS)
    x = jnp.asarray(injected(NOISE_SEED, 1)[0])
    sc = pv.build_prefix(student, obs)
    block = _gemma.Block(configs=(_gemma.get_config("gemma_2b"), _gemma.get_config("gemma_300m")))
    flat = _flat(nnx.split(student)[1])
    lf = {k[len("PaliGemma/llm/layers/"):]: v for k, v in flat.items() if k.startswith("PaliGemma/llm/layers/")}
    ex = {k: v for k, v in lf.items() if re_expert(k)}
    fr = {k: v for k, v in lf.items() if not re_expert(k)}
    tok, smask, ar, cond = student.embed_suffix(sc.observation, x, jnp.ones((1,), jnp.float32))
    pa = einops.repeat(sc.prefix_mask, "b p -> b s p", s=tok.shape[1])
    mask = jnp.concatenate([pa, make_attn_mask(smask, ar)], axis=-1)[:, None, :, :]
    pos = jnp.sum(sc.prefix_mask, axis=-1)[:, None] + jnp.cumsum(smask, axis=-1) - 1

    def body(carry, slice_):
        ex_i, fr_i, kv_i = slice_
        params = traverse_util.unflatten_dict({**ex_i, **fr_i}, sep="/")
        (out, _), st = block.apply({"params": params}, [None, carry], kv_i, pos, mask, [None, cond], True, capture_intermediates=True, mutable=["intermediates"])
        inter = {}
        for name, val in traverse_util.flatten_dict(st["intermediates"], sep="/").items():
            if name.startswith(("attn/__call__",)):       # k, v of the concatenated cache are large and unchanged by the expert pass
                continue
            for i, a in enumerate(jax.tree.leaves(val)):
                inter[f"{name}[{i}]"] = a
        return out[1], inter

    wrapped = jax.checkpoint(body, policy=jax.checkpoint_policies.nothing_saveable, prevent_cse=False)

    def stack(th, b=wrapped):
        return jax.lax.scan(b, tok.astype(jnp.bfloat16), (th, fr, sc.kv_cache))

    def aux(th, b=wrapped):
        c, ys = stack(th, b)
        return jnp.sum(c.astype(jnp.float32)), {"out": c, "ys": ys}

    base = aux(ex)[1]
    zero = jax.tree.map(jnp.zeros_like, ex)
    modes = {"plain_repeat": aux(ex)[1], "jvp": jax.jvp(aux, (ex,), (zero,))[0][1], "vjp": jax.vjp(aux, ex, has_aux=True)[2], "jit": jax.jit(aux)(ex)[1],
             "no_checkpoint_plain": aux(ex, body)[1], "no_checkpoint_jvp": jax.jvp(lambda t: aux(t, body), (ex,), (zero,))[0][1],
             "no_checkpoint_vjp": jax.vjp(lambda t: aux(t, body), ex, has_aux=True)[2]}
    names = list(base["ys"])
    depth = next(iter(base["ys"].values())).shape[0]
    res = {"tag": TAG, "xla_flags": os.environ.get("XLA_FLAGS", ""), "tensor_order": names, "dtypes": {n: str(base["ys"][n].dtype) for n in names}, "modes": {}}
    for m, o in modes.items():
        first, per_layer = None, []
        for layer in range(depth):
            row = {}
            for n in names:
                d = diff(np.asarray(o["ys"][n][layer]).astype(np.float64), np.asarray(base["ys"][n][layer]).astype(np.float64))
                row[n] = d
                if first is None and d["max_abs"] != 0:
                    first = {"layer": layer, "tensor": n, "dtype": res["dtypes"][n], **d}
            per_layer.append(row)
        final = diff(np.asarray(o["out"]).astype(np.float64), np.asarray(base["out"]).astype(np.float64))
        res["modes"][m] = {"first": first, "final": final, "layers_with_any_difference": [i for i, r in enumerate(per_layer) if any(v["max_abs"] != 0 for v in r.values())],
                           "layer_output_rel_l2": [r[names[-1]]["rel_l2"] for r in per_layer]}
        print(f"[scan] {m}: first {json.dumps(first)}; final max {final['max_abs']:.3e} rel_l2 {final['rel_l2']:.2e}", flush=True)
    (OUT / f"scan_probe_{TAG}.json").write_text(json.dumps(res, indent=1))
    print("[end]", flush=True)


if __name__ == "__main__":
    main()
