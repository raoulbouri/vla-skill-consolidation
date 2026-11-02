"""Task 10A: decompose the objective's sensitivity to velocity rounding, from saved primal outputs. Host only."""
import json, sys
import numpy as np
W = None
d = "outputs/ogpo_port/task10a/"
out = {}
for tag in sys.argv[1:]:
    z = np.load(d + f"derivs_primal_{tag}.npz")
    p = lambda m, k: z[f"{m}__{k}"].astype(np.float64)
    vs, vt, kl = p("plain", "vs")[0], p("plain", "vt")[0], p("plain", "kl")[0]
    w = kl / np.sum((vs - vt) ** 2, axis=(1, 2))
    rows = {}
    for m in ("value_and_grad", "jit_forward", "jit_value_and_grad"):
        if f"{m}__vs" not in z:
            continue
        vs2, vt2, kl2 = p(m, "vs")[0], p(m, "vt")[0], p(m, "kl")[0]
        dv, dv2 = vs - vt, vs2 - vt2
        e = dv2 - dv
        cross = 2 * w * np.sum(dv * e, axis=(1, 2))
        quad = w * np.sum(e**2, axis=(1, 2))
        rows[m] = {"dKL_total": float((kl2 - kl).sum()), "cross_sum": float(cross.sum()), "quad_sum": float(quad.sum()),
                   "same_x_only_k0": {"vS_rel_l2": float(np.linalg.norm(vs2[0] - vs[0]) / np.linalg.norm(vs[0])), "vT_rel_l2": float(np.linalg.norm(vt2[0] - vt[0]) / np.linalg.norm(vt[0])),
                                      "vT_0_max_abs": float(np.abs(vt2[0] - vt[0]).max()), "dv_error_over_dv_norm": float(np.linalg.norm(e[0]) / np.linalg.norm(dv[0]))},
                   "dv_error_over_dv_norm_per_step": (np.linalg.norm(e, axis=(1, 2)) / np.linalg.norm(dv, axis=(1, 2))).tolist(),
                   "kl_plain_per_step": kl.tolist(), "kl_mode_per_step": kl2.tolist()}
    out[tag] = {"objective_plain": float(kl.sum()), "v_rms": float(np.sqrt(np.mean(vs**2))), "dv_rms_per_step": np.sqrt(np.mean((vs - vt) ** 2, axis=(1, 2))).tolist(), "modes": rows}
    print(tag, json.dumps(out[tag]))
open(d + "sensitivity.json", "w").write(json.dumps(out, indent=1))
