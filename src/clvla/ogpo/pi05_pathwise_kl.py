"""Differentiable teacher-student path KL for the real pi0.5: one Monte Carlo sample of the expected forward path KL, with the
student trajectory attached to the graph (the Task 9 pathwise estimator). Not a training loop: no optimiser, no parameter updates.

Frozen configuration as Tasks 6-8. The chain comes from the frozen Task 6 sampler and velocities from the frozen Task 5 adapter:

    K_theta(X_theta, o) = sum_{k=0}^{K-1} w_k || v_S(x_k, s_k, o; theta) - v_T(x_k, s_k, o) ||^2,   w_k from pi05_path_kl.kl_coefficients
    d/dtheta K = sum_k 2 w_k dv_k^T [ d_theta v_S + (J_S,k - J_T,k) dx_k/dtheta ]

Task 8's `path_kl` stops every gradient, so it is never the objective here. Modes:
  "full"             the estimator
  "detached"         states are fixed data (`stored_chain`, generated outside the differentiated function): a control, NOT the estimator
  "teacher_blocked"  teacher state input stopped, omits J_T dx/dtheta: a control, NOT the estimator
Teacher parameters are frozen by not differentiating them, or explicitly with `frozen_view`; its state input stays differentiable.
The prefix caches are built by the caller, once, outside the differentiated function.
"""

from __future__ import annotations

from typing import Any, Optional

from clvla.ogpo import pi05_path_kl as pkl
from clvla.ogpo import pi05_sde as sde
from clvla.ogpo import pi05_velocity as pv

MODES = ("full", "detached", "teacher_blocked")


def frozen_view(model: Any) -> Any:
    """The same model with stop_gradient on every parameter: its input derivative is untouched. Allocates no parameter copy."""
    import jax
    from flax import nnx

    graphdef, state = nnx.split(model)
    return nnx.merge(graphdef, jax.tree.map(jax.lax.stop_gradient, state))


def pathwise_kl(student: Any, student_cache: "pv.PrefixCache", teacher: Any, teacher_cache: "pv.PrefixCache", x0: Any, eps: Any,
                cfg: "sde.SDEConfig" = sde.SDEConfig(), mode: str = "full", stored_chain: Optional[Any] = None) -> dict:
    """x0 (B, 10, 32), eps (K, B, 10, 32), fixed data. Returns differentiable fp32 arrays:
      path_kl (B,), kl (B, K), chain (B, K+1, 10, 32), velocities_student, velocities_teacher (B, K, 10, 32).
    `stored_chain` is required by "detached" and refused otherwise."""
    import jax
    import jax.numpy as jnp

    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if (mode == "detached") != (stored_chain is not None):
        raise ValueError('stored_chain is required by mode "detached" and only by it')
    if mode == "detached":
        chain = jax.lax.stop_gradient(jnp.asarray(stored_chain, jnp.float32))
        vs = jnp.stack([pv.velocity(student, student_cache, chain[:, k], s) for k, s in enumerate(cfg.times())], axis=1)
    else:
        samp = sde.sde_sample(student, student_cache, x0, eps, cfg)     # FROZEN Task 6, attached to the graph
        chain, vs = samp["chain"], samp["velocities"]
    pkl._check_inputs(teacher_cache, student_cache, chain, cfg)
    vt = []
    for k, s in enumerate(cfg.times()):
        x = jax.lax.stop_gradient(chain[:, k]) if mode == "teacher_blocked" else chain[:, k]
        vt.append(pv.velocity(teacher, teacher_cache, x, s))            # FROZEN Task 5, at the student's state
    vt = jnp.stack(vt, axis=1)
    w = jnp.asarray(pkl.kl_coefficients(cfg)["weights"], jnp.float32)
    kl = w * jnp.sum((vs - vt) ** 2, axis=(2, 3))
    return {"path_kl": jnp.sum(kl, axis=1), "kl": kl, "chain": chain, "velocities_student": vs, "velocities_teacher": vt}
