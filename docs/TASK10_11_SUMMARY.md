# Task 10 and Task 11: historical summary

Compact record, written 2026-11-01. The detailed ledger is `tests/ogpo_port/README.md`; results are under `outputs/ogpo_port/` (git-ignored). Cleanup record: `docs/CLEANUP_MANIFEST_20261101.md`.

## Standing state

- Task 10, the independent validation of the real-model state-dependent pathwise gradient, is **not closed**. Its latest status is inconclusive. No historical failure has been relabelled.
- Held-out seeds 12000-12007 (observation 0), 13000-13007 (observation 1) and reserved seed 15000 are **unused**. They appear only as reservations in `task10d/preregistration.json` and `preregistration_v2.json`; no result JSON uses them, and no Phase 2 C3 or C4A run exists.
- No v3 preregistration was written.
- No OPD training has run. Task 11's one-step updates touched temporary parameter copies on one fixture observation only.
- Tasks 5-9 are frozen; hashes are in `tests/ogpo_port/frozen_task5_9.sha256` (copy of `outputs/ogpo_port/frozen_task5_9.sha256`).

## Task 10 variants

| Variant | Purpose | Result | Key numbers | Record |
|---|---|---|---|---|
| Task 10 | Real-model pathwise gradient plumbing | **FAILED** one check of 11 | `g_full - g_detached` forward vs reverse 23% vs a 5% bound; `K` varied about 2% by execution path | `task10_pi05_pathwise_gradient.json` |
| 10A | Diagnose the primal inconsistency | Diagnostic; not passed | FP32 eager scaled AD error <= 3.5e-5; BF16 objective spread about 1%, gradient 3.3% between programs | `task10a/` |
| 10B | Validate the canonical execution path | P2 and P3 **pass**, P4 **fail** | Bitwise across 4 processes (objective 42.5960807800293); P4 failed 8 of 8 with Triton on; Triton GEMM FP32 error 1.1e-3 vs 5.9e-7 off | `task10b/analysis.json`, `fd_audit.json`, `jump.json` |
| 10C | Compare canonical BF16 with the FP32 reference | Report only, no threshold | C3 vs C4A full cosine 0.99964, rel 2.7%; state cosine 0.99855, rel 5.4%; objective 0.47% | `task10c/comparison.json` |
| 10D | Preregistered FD, then held-out comparison | **FAILED** Phase 1, seed 10000; Phase 2 not run | Full 0.0051%, full-minus-detached 0.059%, random heads 0.0090%, random all 1.24% vs 0.1% | `task10d/preregistration.json`, `run_fd.json` |
| 10D-v2 | Amended criteria on development seed 14000 | **FAILED** Phase 1; Phase 2 not run | Full 0.0017%, full-minus-detached 0.119% vs 0.1%, heads 0.0010%, random all scaled 5.1e-8 | `task10d/preregistration_v2.json`, `v2/run_fd.json` |
| 10D-v3 calibration 1 | Fixed six-point grid, uncounted, seed 14000 | Method insufficient | Full 0.0089%, Check A 0.053%, Check B state 0.18%, heads 62%, random all scaled 6.9e-5 | `task10d/v3/run_fd_calibration.json` |
| 10D-v3 calibration 2 | 36-point grid, M2 and M4, block omission, uncounted, seeds 10000 and 14000 | **INCONCLUSIVE** | See below | `task10d/v3/calibration2/` |

No variant named 10E exists in the repository, logs or ledger.

### Revised finite-difference validation (calibration 2)

Errors are M2/M4, relative to AD, for seed 10000; seed 14000.

| Curve | Error | M2-M4 | Status |
|---|---|---|---|
| Full objective | 0.010/0.020%; 0.008/0.024% | 0.010%; 0.016% | Within 0.1% |
| Check A | 0.017/0.033%; 0.042/0.094% | 0.016%; 0.052% | Within 0.1% |
| Check B state contribution | 0.122/0.242%; 0.099/0.186% | 0.119%; 0.087% | Inconclusive; block shift up to 0.281%, residual lag-1 +0.82; +0.74 |
| Detached | 0.270/0.532%; 0.041/0.039% | 0.262%; 0.002% | Reported |
| Random heads | 0.36/0.50%; 1.46/1.98% | 0.142%; 0.520% | Above 0.1%, secondary |
| Random all, scaled | 6.4e-8/6.6e-8; 3.3e-8/2.8e-8 | n/a | Within 1e-5 |

**Unresolved:** the state contribution's error changes sign between seeds, so there is no sign of constant bias, but the method cannot resolve it at 0.1%. Omitting the smallest-step block removes most of the M2 error on both seeds; that was seen after the fact and selects nothing.

## Task 11: readiness audit

All results are in `outputs/ogpo_port/task11/` (`synthetic.json`, `real.json`, `followup.json`). Scripts are `tests/ogpo_port/task11_synthetic.py`, `task11_real.py` and `task11_followup.py`.

**Fixture-level, not environment training.** The real-model tests used `pi05_libero` as teacher and a copy with a perturbed `action_out_proj/kernel` as student, on stored dataset frames. No LIBERO rollout, environment reward or fine-tuned teacher was involved. Fine-tuned teachers exist only on aims.

### Density ratios

- **Synthetic** (`d=3`, `K=5`, 8 seeds, up to 1e6 samples per seed): mean ratio 1.80312 vs closed form 1.80308, and 19.127 vs 19.133. Pathwise, score function, leave-one-out baseline, reward-to-go and analytic-KL score are unbiased. Two deliberately wrong estimators were detected as biased.
- **Real, 2048 fresh chains per observation:** ratio vs analytic KL 47.93 vs 47.83, 56.68 vs 56.46, 53.00 vs 52.71, with paired z 0.47, 0.95, 1.27 (limit 3 fixed before that run). An earlier 256 chains gave z -2.60 on observation 0, which did not recur.
- Identical parameters give exactly 0.
- Routing: student trainable gradients nonzero; student VLM, teacher trainable and teacher VLM gradients exactly zero, each from its own derivative.

### Score-function variance

One observation, 32 chains:

| Estimator | Norm per chain | Split-half cosine |
|---|---|---|
| Score function | 143,595 | -0.069 |
| Leave-one-out baseline | 25,574 | -0.019 |
| Reward-to-go | 84,955 | -0.012 |
| Analytic-KL score | 67,103 | 0.022 |
| Pathwise | 1,810 | 0.874 |

Rough extrapolation, using the pathwise gradient as the signal scale: 10% relative error needs about 230 pathwise chains or about 65,000 leave-one-out score-function chains per observation.

### Pathwise and temporary updates

One step on a temporary copy, 16 unseen chains, one observation, KL before 48.65:

| Update | Nominal step | Actual step | KL change |
|---|---|---|---|
| Leave-one-out score function | 0.003 | 0.00221 | +0.037 +- 0.054 |
| Pathwise | 0.003 | 0.00209 | -0.948 +- 0.068 |

This is not validation of the Task 10 state term and not evidence of learning.

### FP32 master requirement

Casting updated parameters back to BF16 kept only about 0.01% of 430M entries changed. Any optimizer needs FP32 master parameters or an equivalent explicit mechanism.

### Unresolved LIBERO integration

The eval loop has seeding, the action path, the 5-of-10 horizon and success logging. Missing: SDE sampling in rollouts, environment reward (`run_episode` discards it), storage of observations, chains, rewards and old log-probabilities, and a PPO or REINFORCE update.

### Decisions at the end of Task 11

KL-only pathwise: NO-GO today, pursue first. Score-function OPD: NO-GO at feasible batch sizes. Environment reward plus KL: NO-GO.
