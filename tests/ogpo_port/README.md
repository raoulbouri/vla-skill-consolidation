# OGPO -> OpenPI port: verification ledger

CPU-only checks that OGPO's flow math (`simchowitzlabpublic/OGPO_public` @ `0b3be413cde766a41257c6b19c0c2b06393a557f`) maps
onto OpenPI pi0.5 conventions. Run: `CUDA_VISIBLE_DEVICES= scripts/py-policy -m pytest tests/ogpo_port -q -p no:cacheprovider`
(OGPO clone: `$OGPO_SRC`, else `~/IRL-project-refs/OGPO_public`, else the xulab refs path).

| Task | File | Status | Verified |
|---|---|---|---|
| 1 | `test_euler_convention.py` | closed 2026-10-30 | s = 1 - t, v_O = -v_pi, Euler steps and trajectories equal; score and tapered corrected mean in OpenPI terms |
| 2 | `test_sde_transition.py` | closed 2026-10-30 | tapered SDE: means, per-step std, sampled states and trajectories equal with injected noise; sigma = 0 -> ODE; final tapered step stochastic; OGPO std sigma*sqrt(s) != Euler-Maruyama sigma*sqrt(s*Delta) (ratio sqrt(K)); mean clip and noise clip are distinct |
| 3 | `test_flow_log_prob.py` | closed 2026-10-30 (35 pass) | path log-probs: sampling == recompute == independent Gaussian sum (tapered and constant, unclipped, unnormalised); ratio 1 at identical params; ft_flow_steps and init term; normalisation; clipping/divisor effects (see table in the review) |
| 4 | `test_gaussian_path_kl.py` | closed 2026-10-30 (8 pass) | teacher-student path KL (frozen config, sigma = 0.05): direct Gaussian KL = closed form = w_k ||v_S - v_T||^2; teacher evaluated only at student states (bitwise); MC log-ratio vs analytic path KL consistent (paired z = -0.86, N = 40000); identical params -> 0; init term cancels; 35/35/250 groups sum exactly |

## Recorded caveats (check these first when something breaks later)

1. **`distrax.Normal` is substituted (Tasks 2-3).** distrax is not installed in our env. OGPO's functions run with `distrax.Normal`
   bound to a stub: `sample` = loc + scale * (injected eps), `log_prob` = `jax.scipy.stats.norm.logpdf` (Task 3). This verifies
   the transition and likelihood EQUATIONS with identical noise; it does NOT test OGPO's actual random-number sampling or distrax's
   own `log_prob`. If results ever differ from a real OGPO run, check this first.
2. **Synthetic velocity only.** All tests use a synthetic nonlinear v_pi(x, s, o). They verify the mathematical adapter, not the
   real pi0.5 action expert; that validation belongs to the integration stage (sigma -> 0 vs OpenPI `sample_actions`).
3. **Clipping is disabled in the equivalence tests on purpose.** OGPO clips the mean (`pg_helper.py:210-211`), the sampled noise
   to mu +- 3 sigma (239-241) and the final action (246-247). These change the process and/or the likelihood and are tested
   separately; investigate them before interpreting any log-probability.
4. **Only the tapered (+ score correction) and, from Task 3, constant schedules are covered;** the learned `noise_fn` branch is not.
5. **Function extraction.** OGPO functions are ast-extracted and exec'd with real `jax.numpy`; only `jax.random` (for injection)
   and `distrax` are rebound. A change in OGPO's source at another commit invalidates these tests (they pin the commit).

## Likelihood status by configuration (Task 3)

| Configuration | What the number is |
|---|---|
| tapered, no clipping, unnormalised | A: exact path density log p(x0..xK) |
| constant, no clipping, unnormalised | A: exact density of the stochastic prefix (x0..x_{K-1}); x_K is deterministic |
| + intermediate mean clip | A: exact density of the modified (clipped-mean) process |
| + 3-sigma noise clip (when it binds) | B: unclipped-Gaussian density at an atom of the clipped distribution (approximation) |
| + final action clip (tapered, when it binds) | B, and sampler != recompute (sampler scores pre-clip, recompute post-clip) |
| + normalisation (steps and/or dims) | B: normalised surrogate; constant noise also has the 11-vs-10 divisor mismatch (ratio != 1 at equal params) |
| ft_flow_steps < K | exact log-density of the last ft transitions conditional on x_{K-ft} (no init term), not of the whole path |

## FROZEN configuration for the teacher-student path-KL work (user-approved 2026-10-30)

| Setting | Frozen choice |
|---|---|
| Flow steps | K = 10 |
| Time convention | OpenPI, s: 1 -> 0 (s_k = 1 - k/K) |
| Euler convention | verified OGPO-equivalent (Task 1): x' = x - Delta * v_pi |
| Noise mode | `ogpo_discrete` (never just "sde") |
| Noise schedule | sigma_k = sigma * sqrt(s_k) |
| Score correction | enabled: mu_k = x_k - Delta [v_pi + sigma^2/2 (x_k + (1 - s_k) v_pi)] |
| Intermediate mean clipping | off (`clip_intermediate_actions=false`) |
| Three-sigma sample clipping | off |
| Final latent-action clipping | off |
| Log-prob normalisation | off |
| Chain scoring | full chain (all K transitions) |
| Initial noise | common N(0, I_320) |
| Stochastic dimensions | all 320 (10 rows x 32) |
| Teacher/student covariance | identical and fixed (sigma_k^2 I) |
| Primary objective | full stochastic path KL D(p_S(X|o) || p_T(X|o)) |
| ODE transfer | separate evaluation |

This gives a well-defined, full-dimensional Gaussian Markov path with positive variance at every transition, so the path KL is
finite and analytical.

### Scoring rule (required)
The trajectory is sampled ONCE, from the STUDENT. Teacher and student transition means are evaluated separately at each
student-generated state:  mu_{S,k} = mu_S(x_k^S, s_k, o),  mu_{T,k} = mu_T(x_k^S, s_k, o).
The teacher scores the transitions along the student's trajectory; it never rolls out its own x_k^T. This is what makes the
objective the reverse KL E_{X ~ p_S}[log p_S(X|o) - log p_T(X|o)].

### Per-step KL (verified derivation)
mu_{S,k} - mu_{T,k} = -Delta (1 + sigma^2 (1 - s_k)/2)(v_S - v_T)  (shared x_k; x_k terms cancel), sigma_k^2 = sigma^2 s_k, so
  D_KL,k = Delta^2 (1 + sigma^2 (1 - s_k)/2)^2 / (2 sigma^2 s_k) * ||v_S(x_k, s_k, o) - v_T(x_k, s_k, o)||^2 = w_k ||v_S - v_T||^2.
For K = 10, sigma = 0.05: w_last / w_first = 10 * (1 + 0.05^2 * 0.9 / 2)^2 = 10.0225 (coefficient ratio only).
LOG BOTH the theoretical weight w_k and the measured per-step KL: the empirical contribution also depends on ||v_S - v_T||^2,
which may be larger at early (noisy) steps.

### Qualifications to keep in mind
- `ogpo_discrete` is an exact discrete Gaussian Markov process but NOT the Euler-Maruyama discretisation of the cited SDE; do not
  claim its endpoint distribution preserves the ODE marginals (Task 2 measured a sqrt(K) std difference). EM is a later ablation.
- Deterministic coordinates make KL infinite only where the student and teacher deterministic maps actually differ on
  student-visited states; shared positive variance on all 320 dims removes that singular-support problem here.
- Group diagnostics (executed real rows 0-4 x dims 0-6 = 35; unexecuted real rows 5-9 x dims 0-6 = 35; padded dims 7-31 = 250)
  add up EXACTLY to the full conditional Gaussian KL (diagonal shared covariance), even though predicted velocities are coupled;
  they are not marginal KLs of the groups. Padded dims are not robot actions; the full KL may spend capacity matching them:
  log their share.
- Data processing: D(p_S(X|o) || p_T(X|o)) >= D(p_S(x_K|o) || p_T(x_K|o)), and the same holds after any SHARED deterministic
  transform/projection to executed environment actions. This bounds, but does not equal, the action KL, and says nothing
  about ODE-deployment equivalence (measured separately).
- The OGPO constant-noise normalisation divisor mismatch (ratio != 1 at equal params) is an OGPO defect: NOT reproduced here.

### Task 4 (closed 2026-10-30, test_gaussian_path_kl.py)
1. direct Gaussian KL vs closed form; 2. transition KL vs weighted velocity regression; 3. Monte Carlo path log-ratio vs the
expected analytical path KL; 4. additivity of the 35/35/250 group contributions.

## Carried forward from Task 4 (user, 2026-10-30)
1. Padded dimensions: the synthetic test put 76.8% of the KL on the 250 padded coordinates (an artifact of a synthetic field that
   treats all 32 dims alike; NOT a prediction for pi0.5). In real-model work ALWAYS log KL_executed, KL_unexecuted, KL_padded.
   Keep the full 320-dim KL as the primary objective; do not silently mask padded dims.
2. The synthetic KL magnitude (~448) is not a training target and must not be used to choose a KL coefficient for pi0.5.
   Base sigma = 0.05 is a reproducible engineering choice, provisional for real policy rollouts.

| 5 | `test_pi05_velocity_adapter.py` (+ `src/clvla/ogpo/pi05_velocity.py`) | closed 2026-10-30, FROZEN (10 pass, xulab GPU 1) | real pi0.5 velocity adapter: = native joint pass (0.0), = Flow.sample trajectory (0.0), vs Pi0.sample_actions 2.54e-3; prefix deterministic; no cross-sample leakage (bitwise); batch-1 vs batch-3 within 1 bf16 ulp at chunk scale; VLM gradient exactly 0, expert/heads finite nonzero. Opt-in: CLVLA_RUN_PI05_TESTS=1 |

### Task 5 precision notes
- Same-shape comparisons are bitwise or <= 3e-3. Changing the BATCH SHAPE changes GEMM shapes and bf16 accumulation order
  (deterministic XLA flags give reproducibility, not shape-invariance): differences up to 1 bf16 ulp at the chunk's scale
  (|v| up to ~5 -> 0.03125). Per-entry relative/ulp bounds are wrong for near-zero entries (cancellation inside the last layer).
- Pi0.sample_actions differs by 2.54e-3 from the eager adapter loop (jit-compiled while_loop vs eager ops: fusion).
- FROZEN: do not relax Task 5 thresholds further. Carry forward: sampling and likelihood recomputation must use MATCHING
  batch shapes (different GEMM shapes give different bf16 velocities, which would contaminate a likelihood-ratio test even
  with identical parameters).

| 6 | `test_pi05_sde_sampler.py` (+ `src/clvla/ogpo/pi05_sde.py`) | closed 2026-10-30, FROZEN (18 pass, xulab GPU 1) | real pi0.5 `ogpo_discrete` sampler: means vs float64 reference <= 1.3e-7, transitions = mean + std*eps <= 1.1e-7, sigma=0 == ODE adapter bitwise, final std 0.05*sqrt(0.1), deterministic, prefix built once, gradients only to expert/heads. Run each real-model module in its own process (memory) |
- FROZEN (Task 6): `src/clvla/ogpo/pi05_velocity.py`, `src/clvla/ogpo/pi05_sde.py`, `test_pi05_velocity_adapter.py`,
  `test_pi05_sde_sampler.py`, and the sampler configuration. Likelihood/KL functionality must NOT be added to the sampler.
- Precision (correction to the module docstrings' "the cast is exact"): casting the bf16 velocity output to fp32 preserves
  that bf16 VALUE exactly; it does not make the model's computation fp32 and does not recover precision lost inside the
  bf16 computation. Latent states and transition quantities are fp32 from that point on.

| 7 | `test_pi05_path_logprob.py` (+ `src/clvla/ogpo/pi05_path_logprob.py`, helper `pi05_real_env.py`) | closed 2026-10-30, FROZEN (15 pass, xulab GPU 1) | exact unnormalised path log-prob of frozen Task 6 chains: init + all 10 transitions, 320 dims; vs float64 reference within fp32 summation-error bound; recomputed means == sampler means bitwise; identical-policy ratio exactly 1; grads only to expert/heads, zero w.r.t. the stored chain |
- FROZEN (Task 7): `src/clvla/ogpo/pi05_path_logprob.py`, `test_pi05_path_logprob.py`, `pi05_real_env.py` (hashes: xulab
  `/tmp/frozen_task5_7.sha256`, together with the Task 5/6 files).
- Qualification (error bound): 16 * eps32 * sum_i |a_i| is a SUMMATION-ERROR bound (fp32 reduction over the per-coordinate
  terms). It does not model rounding in the per-coordinate subtraction, division, squaring or logarithm, so it is not a
  rigorous end-to-end fp32 error bound. Measured errors were <= 6.5% of it.
- Qualification (gradient test): Task 7 shows the FIXED-trajectory log-likelihood has a working gradient path into the intended
  parameters (expert/heads only). It does NOT show that differentiating a Monte Carlo estimate of the path KL this way gives
  the correct KL gradient (the expectation also depends on the student's trajectory distribution). Validate the KL gradient
  estimator separately (pathwise vs score-function) before any training.

| 8 | `test_pi05_path_kl.py` (+ `src/clvla/ogpo/pi05_path_kl.py`) | closed 2026-10-30, FROZEN (9 pass, xulab GPU 1, log `logs/task8_20261030T132710Z.log`) | real teacher-student path KL on STUDENT chains, evaluation only (outputs gradient-stopped): identical student -> every per-step KL, log-ratio and initial difference exactly 0, log-probs bitwise equal; perturbed student (one leaf `action_out_proj/kernel`, alpha 0.2 * std, seed 8800, bf16 effective/intended norm 1.00007, 50/51 leaves share the teacher's buffers); direct Gaussian vs weighted-velocity form rel <= 1.3e-6 per step, <= 2.2e-7 per path; MC (b3, N = 120): mean K 52.26, mean R 52.22, paired R - K -0.04 +/- 0.97 (95% CI [-1.94, 1.86], half-width 3.6% of K); initial density cancels exactly |
- Task 8 notes: per-trajectory R - K equals sum_k eps_k . (mu_S - mu_T)_k / sigma_k to <= 1.4e-3 (sd of R - K 10.6;
  predicted variance 2 E[K] = 104.5, empirical 112.8). KL share on padded dims 78-83% (log it; do not mask). The perturbation
  scale was fixed from a one-off pilot; KL ~ 50 is a test magnitude, not a training target. Training-gradient estimator
  (pathwise vs score-function) is a separate task: these value checks say nothing about it.
- FROZEN (Task 8): `src/clvla/ogpo/pi05_path_kl.py`, `test_pi05_path_kl.py` (hashes: xulab `/tmp/frozen_task5_8.sha256`).
- Qualification (fixture): the 20% relative perturbation of `action_out_proj/kernel` is a validation fixture. It gives a useful
  nonzero KL; it is not evidence of a realistic teacher-student initialisation gap or an appropriate training scale.
- Qualification (padded dims): padded dimensions contribute 78-83% of the reported KL, executed dimensions 6-10%. This is not a
  mathematical error: the quantity is the KL between full 320-dim latent path distributions, which is NOT the KL between
  executed environment action distributions. Keep the frozen full-path objective for now and retain the group diagnostics.
- Qualification (gradients): Task 8 establishes the path-KL VALUES only, not that an autodifferentiated objective gives the
  gradient of the expected path KL.

| 9 | `test_pathwise_kl_gradient.py` | closed 2026-10-30, FROZEN (15 pass, CPU, float64, log `logs/task9_audit_20261030T165022Z.log`) | pathwise gradient of the expected path KL on a nonlinear synthetic model (K = 4, D = 3, sigma = 0.05, M = 64): AD vs central differences best rel 1.6e-11 to 8.4e-11 at h = 1e-5, slope 1.99 (h = 1e-1 to 1e-3), all 20 coordinates rel 1.9e-10; full vs detached differ by 10.4% (FD rejects detached, 2.7-38 vs <= 6.6e-9); teacher parameter gradient exactly 0, blocking the teacher's state input loses 58%; linear-Gaussian exact (all sign patterns) rel <= 3.4e-16, MC N = 200000 |z| <= 1.30 |
- Task 9 notes: mathematics only. Says nothing about bf16, the transformer or the real model's memory cost. Identical parameters: eager gradient exactly 0, `jit(grad)`
  8e-15 (different fusion of the two velocities); `grad(jit)` vs `jit(grad)` differ by rounding (different compiled programs), a given program is bitwise repeatable.
- Task 9 audit (linear-Gaussian enumeration): K = 4, D = 2 has eps_0..eps_4 = 10 scalar noise coordinates, 1024 sign patterns (test_5c). The original 256-pattern test
  samples x_0 (eps_0 enumerated) and holds eps_4 = 0: eps_K moves only x_K, which is not in the objective (per-trajectory KL bitwise blind to it). It is NOT conditional on x_0;
  1024 and 256 agree with the analytic recursion to <= 2.7e-16. A genuinely x_0-conditional enumeration differs (J 55.84 vs 210.57 at sigma = 0.05) and is never the full expectation.
  K = 4 weights (test_5d), from the rollout itself: 12.5000, 16.6771, 25.0313, 50.0938 at sigma = 0.05 (Delta = 1/4, not the K = 10 vector).
- Qualification (enumeration): the 256 patterns enumerate all 8 scalar noise coordinates that reach the objective (eps_0..eps_3); the 1024-pattern test is a redundant check.
  Sign enumeration is exact only because this linear-Gaussian objective and its gradient are quadratic in the noise. It does not generalise to the nonlinear synthetic model or to arbitrary policy objectives.
- Qualification (fixture sensitivity): the 10.4% (detached) and 58% (teacher input blocked) effects show this synthetic fixture can expose those shortcuts. Do not extrapolate either figure to pi0.5.

| 10 | `test_pi05_pathwise_gradient.py` (+ `src/clvla/ogpo/pi05_pathwise_kl.py`), `task10_dot_diagnostic.py` | NOT PASSED, under review (10 of 11, xulab GPU 1, log `logs/task10_20261030T182805Z.log`; diagnostic `logs/task10_dot_diagnostic_20261030T184701Z.log`) | real pi0.5 pathwise gradient, batch 1, K = 10, one trajectory: gradient reaches all expert/head leaves, student VLM and frozen teacher exactly 0, teacher input JVP nonzero, same-process eager runs bitwise equal. FAILED: forward vs reverse directional derivative along a random direction, rel 0.23 vs bound 0.05 fixed in advance |
- Task 10 open issues: K differs by execution path (plain 42.636, vjp 42.885, jvp 43.509 in one process, about 2%; plain 42.657 in another process). Forward vs reverse: g_full 0.49%, (g_full - g_blocked) 0.10%, (g_full - g_detached) 17% (dK 331.5, full 388.9, detached 144.4), random 2-21%. bf16 rounding is a hypothesis, not established. The student-state path is therefore not yet verified at the real model.
- Task 10A diagnosis (2026-10-31; scripts `task10a_*.py`, results `outputs/ogpo_port/task10a/`, logs `logs/task10a_*`): Task 10 remains NOT PASSED. Identical hashed inputs; each mode repeats bitwise. Primal divergence starts at vS_0 inside the transformer stack (embedding stages exact). FP32 control (exact promoted values, eager JVP vs VJP): objective spread 0.0019% (plain, value_and_grad, vjp, jit forward; JVP primal not recorded at FP32), scaled derivative error <= 3.5e-5, full-minus-detached 375.615 vs 375.606 (detached 134.0): no AD-plumbing defect found. FP32 master + BF16 forward: forward bitwise identical, eager gradient values unchanged, no real improvement. BF16 production: objective spread about 1%, gradient differs 3.3% between programs (detached 0.2%, blocked 6.6%), v error 0.15% becomes 4-8% of (v_S - v_T) because rms(v_S - v_T) = 0.045 vs rms(v) = 1. `scripts/py-policy` does not apply `CLVLA_XLA_FLAGS`: autotuning was on for Tasks 5-10 and values change between processes without `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`.
  Reviewer's verdict: diagnostic progress, not approval. Full-FP32 eager execution substantially reduces transformation-dependent discrepancies; BF16 rounding is strongly implicated, but its contribution is not isolated from cache precision (that FP32 run recomputed the prefix caches) or compilation, and no compiled FP32 gradient has run. Do not change production precision on this evidence. Next milestone: the canonical compiled BF16 path (Task 10B).
  Not established: closeness of BF16 production gradients to the FP32 gradient (not compared); the single responsible operation inside the scanned stack; why the FP32 jitted gradient fails a 15.4 GB allocation with 1.77 GiB live. The 17% figure did not reproduce (1.5-3.9% same-sample).
- Task 10B (2026-11-01; `task10b*.py`, results `outputs/ogpo_port/task10b/`, logs `logs/task10b_*`): Task 10 remains NOT PASSED. Canonical path `jax.jit(jax.value_and_grad(f, has_aux=True))`, BF16, flags `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`. P2 PASS: bitwise within a process, on fresh compile and across 4 fresh processes (objective 42.5960807800293); without the flags processes differ (42.56, 42.85; gradient 3.4-4.1%). P3 PASS: 8 fixed noise samples finite, all leaves nonzero; pairwise cosine mean 0.315; trajectory variation 118% vs jit/eager 2.6% vs cross-process 0. P4 FAIL under the intended flags: 8 of 8 FD criteria, controls A and B.
  Cause of that failure, isolated on the real a and K: XLA's default Triton GEMM on this GPU gives an FP32 product error of 1.1e-3 (`--xla_gpu_enable_triton_gemm=false` 5.9e-7, CPU 6.4e-7), and output jumps with no matching change in K. Separate labeled control with Triton off: FP32 FD 8 of 8 pass, A and B. It does not rescue BF16: with Triton off the BF16 objective still spans 42.83-43.43 and the gradient differs 5.9% between programs. Task 10A's FP32 reference ran with Triton on; its JVP/VJP agreement checked AD consistency, not closeness to the evaluated function.
  Not established: closeness of the canonical BF16 gradient to the FP32 reference (never compared directly); which production configuration to use; why FP32 Triton-off jit forward differs by 0.021 from eager; FP32 jitted gradient still fails a 15.4 GB allocation. No production flag or precision was changed.
- Task 10C (2026-11-01; `task10c*.py`, results `outputs/ogpo_port/task10c/`, log `logs/task10c_*_20261101T151129Z.log`): Task 10 remains under review; report-only, no threshold. Common inputs verified by hash, canonical BF16 cache loaded by every primary configuration. Means over seeds 10000, 11000-11002, relative to the second name. Canonical C3 vs FP32 reference C4A: full cosine 0.99964, rel 2.7%; state contribution cosine 0.99855, rel 5.4%; objective 0.47%. C1 eager vs C4A: 2.9%, state 5.5%. C2 jit Triton off vs C1: 3.7%, state 27%. C3 vs C2: 4.2%, state 25%. C4B vs C4A: 0.37%, state 0.21%. For scale, the reference gradient differs between seeds by 116-120%, state 77-80%.
  Limits: FD validated the reference at seed 10000 only, with a different cache and one direction set; it is a precision and execution comparison, not an oracle. 4 trajectories, 1 observation, 1 perturbation. Task 10A's remark that program noise concentrates in the state path does not hold for C3 vs C4A (detached 2.7% vs full 2.7%).
- Task 10D (2026-11-01; `task10d.py`, `outputs/ogpo_port/task10d/preregistration.json`, `run_fd.json`, log `logs/task10d_fd_20261101T155250Z.log`): STOPPED at Phase 1 under the preregistered rule; Phase 2 NOT run; Task 10 remains under review. C4A reproduced bitwise (objective, chain, gradient). Best FD relative error: full 0.0051% (slope 2.00) pass; full-minus-detached 0.059% (slope 1.96) pass; random over action heads 0.0090% pass; random over all parameters 1.24% FAIL (criterion 0.1%, scaled 1.1e-8). That direction's projection is -0.00146; at the best step, one fp32 ulp of f (3.8e-6) alone is about 0.13% of the derivative. Best-of-sweep is fragile: neighbouring steps give 0.16-0.20% for full-minus-detached and 0.6-0.9% for random heads.
- Task 10D-v2 (2026-11-01; `task10d_v2.py`, `outputs/ogpo_port/task10d/preregistration_v2.json`, `task10d/v2/run_fd.json`, log `logs/task10dv2_fd_20261101T163049Z.log`): FAILED at amended Phase 1, development seed 14000; held-out seeds not used; Phase 2 NOT run. The original Task 10D stays recorded as FAILED, files unchanged. full 0.0017% pass; full-minus-detached best 0.119% FAIL (limit 0.1%; slope 1.96 in range; 0.12-0.31% over h = 1e-4 to 1e-3); random heads 0.0010% pass (single lucky step, neighbours 0.32%, 2.7%); random all scaled 5.1e-8 pass; supplementary action_out_proj/kernel 0.0034%, report-only. Post-hoc reading, not a pass: at h = 1e-3 about 0.07% is h^2 truncation; no step in this coarse sweep has both truncation and noise below 0.1%.
- Task 10D-v3 calibration (2026-11-01; UNCOUNTED, used seed 14000; `task10d_v3.py`, `task10d/v3/run_fd_calibration.json`, log `logs/task10dv3_calibration_20261101T173958Z.log`): the old full-minus-detached comparison was Check A (full objective along u_state); Check B, the state contribution itself, had never been run. Six-step grid, D0 + c2 h^2 fit: full 0.0089%; Check A 0.053%; Check B state 0.18%, detached 0.13%; random heads 62%; random all scaled 6.9e-5; kernel 0.42%. Absolute FD noise is about 0.1-0.6 derivative units regardless of the direction's size, so the grid cannot resolve small-signal directions. No v3 preregistration written; seed 15000 and held-out seeds unused; Task 10 open pending review of the numerical method.
- Task 10D-v3 calibration 2 (2026-11-01; UNCOUNTED, consumed seeds 10000 and 14000 only; `task10d_v3_cal.py`, `outputs/ogpo_port/task10d/v3/calibration2/seed_<seed>.json`, log `logs/task10dv3_cal2_20261101T183136Z.log`): INCONCLUSIVE, method not justified for preregistration; no v3 preregistration written, seed 15000 and held-out seeds unused. 36-point geometric grid 5e-4 to 3e-3, M2 and M4 fitted, six-block omission. Errors M2/M4 (seed 10000; seed 14000): full 0.010/0.020%; 0.008/0.024%. Check A 0.017/0.033%; 0.042/0.094%. Check B state 0.122/0.242%; 0.099/0.186%, with M2-M4 0.119%; 0.087%, largest block shift M2 0.174%; 0.113%, M4 0.281%; 0.209%, lag-1 +0.82; +0.74 (signed error negative on 10000, positive on 14000). Detached 0.270/0.532%; 0.041/0.039%. Random heads (12 points 0.05-0.5) 0.36%; 1.46% M2, above 0.1%. Random all scaled 6.4e-8; 3.3e-8, within 1e-5. Kernel (supplementary) 0.23%; 0.026% M2. The M2-M4 and block limits of 0.1% of AD were provisional. Omitting the smallest-h block removes most of the state M2 error on both seeds; this was seen after the fact and selects nothing.
- Task 11 (2026-11-01; `task11_synthetic.py`, `task11_real.py`, `task11_followup.py`, results `outputs/ogpo_port/task11/`, logs `logs/task11_*`): readiness audit, no training run. Task 10 stays inconclusive, unchanged. Synthetic (d=3, K=5, 8 seeds, up to 1e6 samples each): pathwise, sf, sf_loo, rtg, akl unbiased against the closed form; local-reward and analytic-KL-without-explicit-term controls detected as biased. Real fixture (pi05_libero teacher, Task 8 perturbed student, not a fine-tuned teacher), BF16 jit: sampled ratio vs analytic KL, 2048 fresh chains per observation, z 0.47, 0.95, 1.27 (limit 3 fixed before that run; the first 256 chains gave z -2.60 on obs 0); identical parameters exactly 0; routing: student trainable nonzero, student VLM, teacher trainable and teacher VLM exactly zero, each in its own derivative (a two-group derivative ran out of memory). Gradient noise on one observation, 32 chains: split-half cosine sf -0.069, sf_loo -0.019, rtg -0.012, akl 0.022, pathwise 0.874. One step on a temporary copy, 16 paired chains: sf_loo h=3e-3 dK +0.037 +- 0.054; pathwise h=3e-3 dK -0.948 +- 0.068. BF16 cast kept only about 1e-4 of the entries changed (effective step 0.0021-0.0022 of nominal 0.003; 0.0004 of 0.001), so an FP32 master copy is required. No environment rollout or reward update was run. Fine-tuned teachers exist only on aims and are untested in this harness.
- Minimal Flow-OPD Stage 3 (2026-11-01; `scripts/opd_stage3.py`, `outputs/opd_stage3/20261101T235006Z_t035/`): one teacher-only pathwise update on a temporary FP32 master, t35, teacher `lora-25demos-s500`, 8 student observations from one Stage 2 episode, 32 trajectories. Program KL gap 0.026% (gate 1%). Fresh paired noise, 128 pairs: KL 661.19 to 650.76, difference -10.43, 95% CI [-13.46, -7.40] over observation means; random-direction control +0.03. BF16 forward step norm 0.0427 vs master 0.0045, cosine 0.107 with the negative gradient. Same 8 observations; no environment success evaluated; Task 10 stays inconclusive.
- FROZEN (Tasks 5-9): hashes in `outputs/ogpo_port/frozen_task5_9.sha256`. It replaces `/tmp/frozen_task5_8.sha256`, lost when xulab rebooted on 2026-10-30; the Task 5-8 hashes were re-established against the independent aims copies (all 9 identical).
