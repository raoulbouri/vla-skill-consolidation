"""Task 10B: the final FP32 matrix product in isolation, real a and K. TENB_FLAGS varies by process; GPU and CPU. Diagnostic only."""
import os
os.environ["XLA_FLAGS"] = os.environ.get("TENB_FLAGS", "")
import json
import jax, jax.numpy as jnp, numpy as np
from flax import nnx
z = np.load("outputs/ogpo_port/task10b/dot_case.npz")
a, K, b, gk = z["a"], z["K"], z["b"], z["gk"]
exact = lambda KK: a.astype(np.float64) @ KK.astype(np.float64) + b.astype(np.float64)
print("[case]", a.dtype, K.dtype, a.shape, K.shape, flush=True)
out = {"flags": os.environ["XLA_FLAGS"]}
for name, dev in (("gpu", jax.devices()[0]), ("cpu", jax.devices("cpu")[0])):
    A, B = jax.device_put(jnp.asarray(a), dev), jax.device_put(jnp.asarray(b), dev)
    def run(KK, how):
        K2 = jax.device_put(jnp.asarray(KK), dev)
        if how == "dot_general": return np.asarray(jax.lax.dot_general(A, K2, (((A.ndim - 1,), (0,)), ((), ()))) + B)
        if how == "jit_dot": return np.asarray(jax.jit(lambda x, y: jnp.dot(x, y))(A, K2) + B)
        if how == "einsum": return np.asarray(jnp.einsum("btd,de->bte", A, K2) + B)
    for how in ("dot_general", "jit_dot", "einsum"):
        v0 = run(K, how)
        rows = {"max_err_vs_float64": float(np.abs(v0.astype(np.float64) - exact(K)).max()), "out_dtype": str(v0.dtype), "v_abs_max": float(np.abs(v0).max())}
        for h in (1e-7, 1e-5):
            K1 = (K.astype(np.float64) + h * gk.astype(np.float64)).astype(np.float32)
            v1 = run(K1, how)
            rows[f"h{h}_dv_norm"] = float(np.linalg.norm(v1.astype(np.float64) - v0.astype(np.float64)))
            rows[f"h{h}_exact_dv_norm"] = float(np.linalg.norm(exact(K1) - exact(K)))
        out[f"{name}_{how}"] = rows
        print(f"[{name} {how}]", json.dumps(rows), flush=True)
