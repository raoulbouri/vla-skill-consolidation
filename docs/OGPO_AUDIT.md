# OGPO code audit (read-only), 2026-10-30

Scope: what the OGPO code does, how its likelihood is computed, how its RL loop works, and how its conventions line up
with OpenPI π₀.₅. No code was ported, installed or run against OGPO. Two questions were settled with a pure-NumPy
re-implementation on CPU (`tmp/ogpo_toy_check.py`, git-ignored, run in our policy env; it imports no OGPO code).

| Item | Value |
|---|---|
| OGPO repo | https://github.com/simchowitzlabpublic/OGPO_public, shallow clone at `/mnt/data/users/bbouri/IRL-project-refs/OGPO_public` |
| OGPO commit | `0b3be413cde766a41257c6b19c0c2b06393a557f` (2026-06-06T18:28:17Z) |
| Paper | arXiv 2605.03065 v4 (2026-06-26), "OGPO: Sample Efficient Full-Finetuning of Generative Control Policies"; read through the arXiv HTML render, whose equation text is partly garbled (noted where it matters) |
| Our OpenPI | `third_party/openpi` @ `215abfb217dbac7d5f1273282331b9b1866c0479` (2026-08-24) |

All OGPO paths below are relative to the OGPO clone; `pg` = `ogpo/agents/modules/pg_helper.py`,
`ag` = `ogpo/agents/ogpo.py`, `run` = `ogpo/runners/online_rl_runner.py`.

## File / function index

| What | Where |
|---|---|
| Score from velocity, drift correction | `pg:7-21` `flow_score_from_velocity`; `pg:24-43` `sde_drift_correction` |
| ODE samplers | `pg:46-78` `sample_flow_actions_ode`; `pg:81-120` `..._ode_with_correction` (legacy denoiser path) |
| SDE sampler (rollouts, PPO "old" log-prob) | `pg:123-254` `sample_flow_actions_sde` |
| Log-prob recompute (PPO "new" log-prob) | `pg:257-414` `compute_flow_log_prob` |
| PPO loss; group advantages; safeMax | `pg:417-448`; `pg:478-509`; `pg:451-476` |
| chi_po / kl_reg advantages and PPO loss | `pg:512-623` |
| Agent wrappers: log-probs, sampling | `ag:148-216` `compute_log_probs_and_entropy`; `ag:224-236` `_sample_actions`; `ag:2248-2333` `sample_actions_with_noise` |
| Actor loss (PPO / AWR / FPO / chi_po+kl_reg branches) | `ag:510-801` |
| Critic loss, TD loss | `ag:1182-1403`; `ag:803-1026`; `ogpo/agents/modules/q_helper.py:14-30, 328-351` |
| Update step, target and pi_slow EMA | `ag:1534-1604`; `ag:1840-1877` |
| Best-of-N rollout action selection | `ag:1880-1967` |
| BC (flow-matching) loss and offline update | `ag:2336-2461`; `ag:1443-1480`; targets `ogpo/agents/modules/igp_targets_helper.py:127-140` |
| BC-to-online switch (optimizers, pi_slow init) | `ag:2640-2718` `reset_optimizers_with_lr`, called at `run:417-418` |
| Online loop | `run:318-700`; sequence sampling `ogpo/utils/datasets.py:118-200` |
| Flow actor network | `ogpo/networks/actors.py:114-212` `ActorVectorField` (flax.linen) |
| PaliGemma frozen encoder | `ogpo/networks/encoders/paligemma.py` (802 lines) |
| Configs / scripts | `ogpo/configs/algos/common.yaml`, `ogpo/configs/algos/ogpo.yaml` (`_base_: common.yaml`), `scripts/ogpo/*.sh` |
| Environments | `envs/*.py`; dispatch in `envs/env_utils.py` `make_env_and_datasets` |

### Effective configuration
`ogpo.yaml` overrides `common.yaml`, and CLI flags override both (`ogpo/utils/config_loader.py:195-217`,
`_set_nested_value` creates missing keys, so `use_tapered_noise`, which is in no yaml file, can be set from a script).
Every `scripts/ogpo/*.sh` (and `ogpo_awr`, `ogpo_fpo`) sets `use_constant_noise=false use_tapered_noise=true
error_correct_sde_to_ode=true` with `constant_noise_std=0.05` (0.01 for `ogpo_awr`/`ogpo_fpo`), and none sets the
normalization or clipping keys. So:

| Key | `ogpo.yaml` default run | `square_image_paligemma.sh` run |
|---|---|---|
| noise schedule | constant, σ = 0.005 (`ogpo.yaml:99-100`) | tapered, σ = 0.05 (`square_image_paligemma.sh:144-148`) |
| `error_correct_sde_to_ode` | true (`ogpo.yaml:145`) | true (`:147`) |
| `flow_steps` / `ft_flow_steps` | 10 / 10 (`ogpo.yaml:130-131`) | 10 / 10 (`:60-61`) |
| `normalize_denoising_horizon`, `normalize_act_space_dimension` | true, true (`ogpo.yaml:141-142`) | true, true (inherited) |
| `clip_intermediate_actions`, `denoised_clip_value` | true, 1.0 (`common.yaml:236-237`) | same (inherited) |
| `randn_clip_value` | 3.0 (`common.yaml:268`) | same (inherited) |
| `act_min`, `act_max` | -1, 1 (`common.yaml:89-90`) | same |
| `horizon_length` (chunk H) | 4 (`common.yaml:167`) | 4 (`:51`); 8 for toolhang/transport scripts |

## B1. Flow sampler and likelihood

Notation: K = `flow_steps`, Δ = 1/K, σ = `constant_noise_std`, A = act_dim = `action_dim × horizon_length` (the
chunk is one flat vector, `ag:2265`).

1. **Initial noise.** `pg:170-171` `actions = jax.random.normal(key, (batch_size, act_dim))`: x₀ ~ N(0, I_A).
2. **Time convention and grid.** t = 0 is noise, t = 1 is data. `pg:183,187`: `dt = 1.0 / flow_steps`,
   `t = i / flow_steps` for i = 0..K−1, so t ∈ {0, 0.1, …, 0.9} for K = 10. Training path
   (`igp_targets_helper.py:134-138`): x_t = (1−t)·x₀ + t·x₁, target v = x₁ − x₀, t ~ U(0, 1), x₀ ~ N(0, I).
3. **Euler step.** ODE: `pg:72` `actions = actions + vels * dt`, i.e. x_{i+1} = x_i + Δ·v_θ(x_i, t_i, o). SDE mean:
   `pg:208` μ_i = x_i + Δ·drift_i.
4. **Drift and score correction.** Score from velocity (CondOT) `pg:21`: ∇log p_t(x) = (t·v − x)/(1−t).
   Correction `pg:39-43`: c = (σ²/2)·(t·v − x)/(1−t) under constant noise; c = (σ²/2)·(t·v − x) under tapered noise
   (σ_t² = σ²(1−t) cancels the 1/(1−t)). Applied when `error_correct_sde_to_ode` and (tapered, or constant and not the
   last step) (`pg:198-206`), so drift = v + c. The ODE path (`pg:46-78`) has no correction. Singular last step under
   constant noise: no correction (`pg:199`) and σ = 0, so the step is deterministic (`pg:216-218, 233-234`).
5. **Per-transition std σ_i** (`pg:213-226`). Tapered: σ_i = σ·√(1−t_i) (`pg:214`). Constant: σ_i = σ, and 0 on the
   last step. Neither: σ_i = max(noise_net(o, t_i), `min_noise_std`). **σ_i is the discrete std directly**:
   `pg:236` `distrax.Normal(mean_next, sigma)`, with no √Δ factor, while the drift correction uses the continuous-SDE
   coefficient σ²/2. NumPy check (exact 1-D velocity, K = 10, target N(0.3, 0.1²), 4×10⁵ samples): ODE endpoint std
   0.0618. Code-style std σ gives 0.0662 (tapered, σ = 0.05) and 0.0696 (constant, σ = 0.05); std σ√Δ gives 0.0619 in
   both. At σ = 0.01 all are within 0.0003 of the ODE. So with the scripts' σ = 0.05, the SDE endpoint is wider than
   the ODE endpoint the correction is meant to preserve.
6. **Clipping.**
   - Mean: `pg:210-211`, μ_i ← clip(μ_i, −1, 1) when `clip_intermediate_actions` (true), at every step including the
     last.
   - Sample: `pg:239-241`, x_{i+1} ← clip(x_{i+1}, μ_i − 3σ_i, μ_i + 3σ_i) (`randn_clip_value` = 3) on every
     stochastic step.
   - Final action: `pg:246-247`, x_K ← clip(x_K, `act_min`, `act_max`) = [−1, 1], applied **after** the last step's
     log-prob was added (`pg:243-244`).
   - The ODE sampler used for advantage-free paths and evaluation passes `clip_intermediate=False` (`ag:264, 272`)
     and clips only the final action (`pg:77`).
7. **Final transition.** Constant noise: deterministic (σ = 0, `pg:233-234`). Tapered: stochastic, σ_{K−1} = σ·√Δ
   (0.0158 for σ = 0.05, K = 10).
8. **Saved chain.** `pg:172, 249, 251`: `chains` = [x₀, x₁, …, x_K] of shape (B, K+1, A). x_K is stored after the
   final clip. `sigmas` is (B, K, A) (`pg:252`). With G group samples (vmap, `ag:226-234`): (G, B, K+1, A).
9. **Accumulation during sampling.**
   logp = Σ_d log N(x₀,d; 0, 1) [only if `ft_start` = 0, `pg:178-181`] + Σ_{i ≥ ft_start, stochastic}
   Σ_d log N(x_{i+1},d; μ_i,d, σ_i,d²) (`pg:243-244`), evaluated on the 3σ-clipped sample with an unclipped Gaussian
   and, for the last step, before the final [−1, 1] clip. It is then normalized in `ag:2318-2331`: divided by
   `logprob_steps` = (`effective_ft` + 1 if the full chain is scored, else `effective_ft`), and by A.
10. **Recompute for the update.** `compute_flow_log_prob` (`pg:257-414`), called through `ag:148-216` with the
    current actor's parameters (`ag:279-292`), re-evaluates v on the stored chain with the same grid, drift, σ and mean
    clip (`pg:318-362`) and scores `chains[:, 1:]` (`pg:371`/`378`). Differences from the sampling-time value:
    - **Parameters.** The "old" log-prob comes from the sampler, which always uses `target_actor` (`ag:2259`,
      ignoring its `use_target` argument for the actor module). The "new" one uses the online actor.
    - **Final clip.** On the last stochastic step (tapered), the recompute scores the post-clip x_K, the sampler the
      pre-clip value. The two differ only when the final [−1, 1] clip binds.
    - **Normalization divisor.** The recompute uses `logprob_steps` = 1 (init) + `contributing_steps`, where the
      latter is `effective_ft − 1` under constant noise (`pg:375`) and `effective_ft` otherwise (`pg:379`). The NumPy
      re-count gives MATCH for tapered (11 = 11 for K = 10, ft = 10; 3 = 3 for ft = 3) and MISMATCH for constant noise
      (sampler 11 vs recompute 10; 3 vs 2). With normalization on (`ogpo.yaml` default), identical parameters then
      give log ρ = L·(1/10 − 1/11)/A, not 0. Estimated from the formulas (not run): for σ = 0.005, L/A ≈ 9×3.9 − 1.4 ≈
      34, so log ρ ≈ 0.3 and ρ ≈ 1.36, outside the clip range 1 ± 0.01. The shipped scripts all use tapered noise, so
      this applies only to the yaml default.
11. **`ft_flow_steps`.** `pg:167-168, 243`; `pg:290-291, 381`: only the last `effective_ft` transitions are scored.
    Earlier ones are omitted, and the initial-noise term is dropped. The full chain is still generated. The value is 10
    = K in the yaml and nearly all scripts (full chain); `scripts/ogpo/adroit.sh` uses 3.
12. **Normalization flags.** `normalize_denoising_horizon`: divide by `logprob_steps` (`pg:397-398`;
    `ag:2327-2328`). `normalize_act_space_dimension`: divide by A (`pg:400-401`; `ag:2330-2331`). Defaults: false in
    `common.yaml:238-239`, true in `ogpo.yaml:141-142`. No script overrides them, so OGPO runs use true/true. The PPO
    ratio is therefore exp(Δlogp / (steps × A)), a per-step, per-dimension geometric mean.
13. **Initial-noise log-prob.** Included when the full chain is scored (`pg:178-179`, `pg:300-303`). The same x₀ and
    N(0, I) appear in both terms, so it cancels in the ratio. It does count in the normalization divisor (item 10).

## B2. Likelihood correctness

| Config | Verdict | Why |
|---|---|---|
| `ogpo.yaml` default (constant σ = 0.005) | **Normalized surrogate of a Gaussian approximation; the old and new terms are also inconsistently normalized** | (a) 3σ sample clipping (`pg:239-241`) gives truncated transitions with point masses at μ ± 3σ, scored with an untruncated Gaussian (`pg:244`, `pg:371`). (b) The log-prob is divided by steps and by A (`ag:2327-2331`, `pg:397-401`). (c) The sampler divides by 11 and the recompute by 10 (B1.10), so ρ ≠ 1 at equal parameters. The last step is a deterministic map and is correctly excluded. |
| `square_image_paligemma.sh` (tapered σ = 0.05) | **Normalized surrogate of a Gaussian approximation** | (a) and (b) as above, with matching divisors (11). (d) The last transition is stochastic and its log-prob is taken before the final [−1, 1] clip in the sampler but after it in the recompute (`pg:243-247` vs `pg:371/378`). The two agree unless the clip binds. |

The mean clip (`pg:210-211`) is a deterministic function of x_i applied identically in both computations, so it does
not break exactness of the density of the process actually sampled. It does change that process: it is no longer the
corrected SDE.

To score the exact joint path density p(x₀, x₁, …, x_K) of what is sampled:

| Setting | Required value | Supported without code change? |
|---|---|---|
| `normalize_denoising_horizon` | false | yes (config/CLI key) |
| `normalize_act_space_dimension` | false | yes |
| `randn_clip_value` | large enough never to bind (e.g. 1e6) | yes (key exists; `inf` through the CLI parser: not determined) |
| final action clip | must not bind on a scored stochastic step: use constant noise (last step deterministic), or `act_min`/`act_max` wide enough | constant noise: yes. Widening `act_min`/`act_max` also widens the executed and evaluated actions (they feed `pg:77, 247`) |
| `ft_flow_steps` | 0 or K (full chain) | yes |
| noise schedule | either (tapered = stochastic last step; constant = deterministic last step, density over the stochastic part) | yes |
| `clip_intermediate_actions` | either (does not affect exactness; changes the process) | yes |

## B3. RL algorithm (as implemented)

- **Offline BC.** `bc_pi_steps` = 500k (`ogpo.yaml:24`; `square_image_paligemma.sh:19`).
  - Loss: flow matching on dataset chunks, masked by `valid` (`ag:2418-2452`).
  - Updates: the target actor is EMA-updated during BC (`ag:1462`).
  - Early stop: when evaluation success ≥ `clip_bc_threshold` = 0.45 with `clip_bc` true (`bc_runner.py:213-222`;
    gate on ODE by default, `ogpo.yaml:57`). This produces the "BC clipped to ~50%" starting policies the paper uses.
- **BC-to-online switch** (`run:417-418` → `ag:2640-2718`).
  - The actor optimizer is reset to AdamW with base LR `ppo_lr` = 4.5e-5 and a cosine schedule.
  - The critic optimizer is also reset with base LR `ppo_lr` (`ag:2652-2654`), not `critic_lr`.
  - If `chi_po`, `kl_reg` or `fwd_kl_reg` is on, `pi_slow` is initialized as a copy of the actor
    (`ag:2706-2708`).
- **Online collection** (`run:529-608`).
  - When the action queue is empty, the agent samples one chunk with `sample_actions_BON` (`best_of_n` = 8): 8 SDE
    samples from the **target actor** (`ag:1896-1898`), keeping the argmax over samples of the min of 2 randomly
    sub-sampled target critics (`subsample_bon` true, `ag:1955-1965`).
  - All H actions of the chunk are executed before re-querying (`run:534-544`).
  - Transitions (obs, action, reward, terminal, mask = 1 − terminated, next_obs) are buffered per episode and written
    at episode end, with `is_success` and `score` (`run:560-600`).
  - Robomimic reward is r − 1 ∈ {−1, 0} (`envs/robomimic_utils.py:223, 419`).
  - Evaluation uses the ODE sampler (`run:677`).
- **Replay buffer.**
  - Size 2M; single-step transitions.
  - Training sequences of length H start at uniform indices, which may cross episode boundaries; a `valid` mask
    handles that (`datasets.py:118-200`).
  - Chunk reward r_H = Σ_{i<H} γ^i r_i (`datasets.py:180-181`); masks are min-accumulated.
  - Online data only (`offline_ratio` = 0; `run:623-627`).
  - Batch 256 × UTD 1. Training starts after `start_training` steps (10k; 20k in the PaliGemma script) and once the
    buffer holds ≥ batch·UTD·H transitions (`run:614-621`).
  - Success buffer = sequences from successful trajectories only (`main_helper.py:313-326`, `success_only=True`). It
    is used for the BC term and, if `use_success_buffer_q`, for an extra TD update (`ag:1709-1737`).
- **Critic.**
  - Ensemble `num_qs` = 10 (`common.yaml:137`), MSE loss (`critic_loss_type: mse`), every head regressed to the same
    target (`q_helper.py:353+`).
  - Target y = r_H + γ^H · mask · Q̄(s_{t+H}, a′) (`q_helper.py:328-350`), γ = 0.99.
  - a′ ~ target-actor SDE sample (`ag:1341` → `ag:1980`). With `q_variance_reduction` (on in the PaliGemma script):
    the mean over 8 such a′ per head (`ag:1297-1321`).
  - Aggregation `q_agg` = mean over heads (`q_helper.py:14-30`).
  - Target networks: Polyak θ̄ ← τθ + (1−τ)θ̄ after every update (`ag:1840-1852`), with τ = 0.05 for the critic. The
    actor uses `actor_tau`: 0.05 in `ogpo.yaml:83`, 0.005 in the scripts.
  - Warm-up: `q_warmup_steps` = 0 and `calql_steps` = 0 in all configs read. UTD = 1 (`utd_q` = `utd_pi` = 1).
- **Group-relative advantages.**
  - For each replay state (`ppo_batch_size` = 256), G = `group_num_samples` = 32 fresh SDE chains are drawn from the
    **target actor** (`ag:722`, `224-236`). These are not the executed actions.
  - Q values come from the target critic ensemble (`ag:375-466`).
  - `vanilla`: A_j = Q̄(s, a_j) − mean_j Q̄(s, a_j), optionally std-normalized (`pg:480-485`).
  - `conservative` (used by the scripts): A_{j,m} = Q_m(s, a_j) − mean_j Q_m(s, a_j), then safeMax over m: min_m if
    all > 0, max_m if all < 0, else 0 (`pg:487-490`, `451-461`).
- **Actor update** (`ag:721-764`).
  - ρ = exp(lp_θ − lp_old), where lp_old is the sampler's normalized log-prob under the target actor and lp_θ is
    recomputed with gradients (`pg:417-420`).
  - Clipped surrogate −mean(min(ρA, clip(ρ, 1 ± 0.01)·A)) (`pg:423-435`).
  - The total also includes `entropy_coeff` (0) × (−entropy rate) and `bc_coeff` × BC flow-matching loss. The BC term
    is computed on the success batch if one exists (`success_flag`), otherwise on the replay batch (`ag:492-508`).
    `bc_coeff` = 0.1 in `ogpo.yaml:96`, 1.0 in the scripts.
  - Then the target actor (`actor_tau`) and `pi_slow` (`tau_slow`) are EMA-updated (`ag:1557-1558`).
  - AWR (`use_awr`) and FPO (`use_fpo`) branches replace the ratio with CFM-based quantities (`ag:546-673`); both are
    off by default.
- **`pi_slow` and `kl_reg`.** `pi_slow` is OGPO's **own internal reference**: a snapshot of its own BC-initialized
  actor at the online switch, then a slow EMA of the online actor with `tau_slow` = 5e-4 (`ag:1860-1877`). It is **not
  an external teacher**. It is used only when `chi_po`, `kl_reg` or `fwd_kl_reg` is on; all are false in `ogpo.yaml`
  and in every script read.
  - With `kl_reg`, log_ratio = stop_grad(lp_θ − lp_slow) on the target-actor chains (`ag:678-683`).
  - The advantage becomes (q_targ − β_kl·log_ratio) − group mean, with q_targ = (1−w)·mean_m Q + w·min_m Q,
    w = sigmoid(α·log_ratio), and β_kl = `kl_reg_beta_base`·std_m(Q)/`chi_po_q_std_target` (`pg:518-571`).
  - The PPO upper clip becomes min(1+ε, 1+R_max/β) (`pg:594-601`). The critic target is also blended pessimistically
    using the batch-mean drift (`ag:1345-1386`).
  - Because the chains come from the target actor and the log-probs are normalized, the averaged log_ratio is a
    sample-based, normalized log-ratio. It is not exactly KL(π_θ ‖ π_slow).
  - `fwd_kl_reg` adds a BC flow-matching loss toward samples from `pi_slow` (`ag:347-374, 750-759`).

## Discrepancies (paper vs code)

1. **Noise schedule.** Paper App. E.2: the reported runs used a constant schedule; the tapered schedule is "provided"
   in the code. Code: every shipped OGPO script uses tapered σ = 0.05; `ogpo.yaml` defaults to constant σ = 0.005; the
   paper's table lists `constant_noise_std` 0.01.
2. **Log-prob normalization.** Paper Eq. 3.2/E.5: ρ is a plain product over the K transitions. Code: divided by step
   count and A (B1.12), on in `ogpo.yaml` and therefore in every script.
3. **Initial-noise term.** Paper ratio excludes it. Code includes it for full chains (cancels in ρ, counted in the
   divisor).
4. **Divisor mismatch under constant noise** (B1.10). This is not in the paper and makes ρ ≠ 1 at equal parameters
   for the yaml default.
5. **Discrete std.** Paper Eq. C.2 writes N(·, σ_k² I), and the code matches it (no √Δ). The marginal-preserving SDE
   the paper cites (Eq. E.7, dX = [v + (σ²/2)s]dτ + σ dW) would discretize to std σ√Δ. The NumPy check shows the code's
   choice widens the endpoint at σ = 0.05.
6. **Transition mean.** Paper Eq. 3.4 as extracted reads N(v_θ(a_k, k/K, s), σ_k² I); the code mean is
   x + Δ·(v + c). Possibly an extraction artifact; not settled.
7. **Old-policy EMA rate.** Paper Eq. A.4: α = 0.995 (τ = 0.005); paper table: `actor_tau` 0.05 default, 0.001 for
   LIBERO. Code: `ogpo.yaml` 0.05; scripts 0.005.
8. **BC coefficient.** Paper: typically 1.0 (App. A.1); off for LIBERO. Code: `ogpo.yaml` 0.1; scripts 1.0.
9. **Regularizers.** Paper: no explicit KL/χ² penalty (Takeaway 3). Code: optional `chi_po`, `kl_reg`, `fwd_kl_reg`
   with `pi_slow`, all off by default.
10. **Clipping.** Paper: the 3σ sample clipping, the mean clipping and the final-action clip are not described in the
    text read.
11. **LIBERO.** Paper App. I.4 reports LIBERO experiments (10 LIBERO-90 training tasks, 2 held-out tasks, MUSE
    language encoder + IMPALA/FiLM, γ = 0.999, group 8, batch 64). The public code has no LIBERO environment, and
    `README.md:108` / `scripts/README.md:14,97` refer to `*_pg_libero.sh` scripts that are not in the repository.
12. **PaliGemma checkpoint.** Paper: a frozen PaliGemma backbone from π₀.₅, checkpoint not specified. Code: the
    docstring defaults to `pi05_base` (`paligemma.py:15-17, 674`); the script points to `pi05_libero/params`
    (`square_image_paligemma.sh:174-176`).

## Not determined

- Whether the in-place dict assignments in `target_update`/`pi_slow_update` (`ag:1852, 1870`) behave as intended
  under `jax.jit`. Not run.
- Whether the CLI parser accepts `inf` for `randn_clip_value`, `act_min` or `act_max`.
- Whether robomimic actions are pre-normalized in the HDF5. A grep of `envs/robomimic_utils.py` found no action
  normalization; actions are only clipped to [−1, 1].
- Paper equations garbled in the HTML render: Eq. 3.4 transition mean; the discount exponent in Eq. A.1.
- Runtime or memory of OGPO with a π₀.₅-size actor: no such configuration exists in the code.

## B4. Compatibility facts

1. **Framework.**

| Package | OGPO (`pyproject.toml` / `uv.lock`) | Our policy env (measured) / OpenPI pin (`third_party/openpi/pyproject.toml`) |
|---|---|---|
| Python | ≥3.11, <3.13 | 3.11.17; sim env 3.8 (LIBERO) |
| jax / jaxlib | ≥0.9.0 / lock 0.10.1 | 0.5.3 (OpenPI pins `jax[cuda12]==0.5.3`) |
| flax | ≥0.12.3 / lock 0.12.7; networks use `flax.linen` (`actors.py:6`) | 0.10.2 (pinned); OpenPI π₀.₅ uses `flax.nnx` |
| optax | lock 0.2.8 | 0.2.4 |
| orbax-checkpoint | lock 0.11.40 | 0.11.13 (pinned) |
| numpy | ≥2.3.5 / lock 2.4.6 | 1.26.4 |
| distrax | lock 0.1.8 (used throughout `pg`) | not installed |
| transformers | override 4.57.1 | 4.53.2 |
| robosuite / mujoco | lock 1.5.1 / 3.9.0 (robomimic extra) | sim env 1.4.1 / 3.2.3 |

   Conflicts: jax 0.5 vs ≥0.9; flax 0.10 vs ≥0.12; numpy 1.x vs 2.x; distrax missing; robosuite 1.4 vs 1.5; Python
   3.8 sim env vs ≥3.11.
2. **LIBERO adapter: no.** `envs/` has no LIBERO module (`d3il`, `d4rl`, `franka_kitchen`, `gymnasium_robotics`,
   `ogbench`, `pusht`, `robomimic`, `roboverse` utils only). `make_env_and_datasets` has no LIBERO branch, and
   neither `pyproject.toml` nor `uv.lock` mentions LIBERO. A repo-wide grep finds LIBERO only in the README files,
   the PaliGemma scripts' checkpoint path, and `paligemma.py` comments.
3. **PaliGemma example.** SigLIP So400m/14 and the Gemma 2B trunk are used as a **frozen encoder** (`paligemma.py:1-25`).
   - Weights come from a π₀.₅ orbax checkpoint; π₀.₅'s action-expert `*_1` parameters are dropped
     (`paligemma.py:540-551`).
   - SigLIP: width 1152, depth 27, MLP 4304, 16 heads, patch 14 (`paligemma.py:719-730`).
   - Each camera image (224 px) gives 256 image tokens, plus a fixed robomimic task prompt (`ROBOMIMIC_PROMPTS`,
     ≤64 tokens). These pass through the Gemma trunk (bidirectional) and are mean-pooled into one 2048-D vector per
     camera (`paligemma.py:751-761`); proprio is appended.
   - Observations are pre-encoded at collection time (`run:548-550`; `freeze_vision_encoder=true`).
   - Trained: a separate MLP flow actor (`ActorVectorField`, hidden [512]×4, output A = 7×H) with a two-tier obs
     encoder, plus a 10-head MLP critic ([512]×5).
   - **No VLA action head is trained or used.**
4. **Actor interface.**
   - The sampler and likelihood call `actor_fn(obs, x, t, params=None, is_encoded=True) -> v` with obs (B, d_obs)
     encoded features, x (B, A) flat, t (B, 1) in [0, 1) with 0 = noise, and v (B, A) (`pg:187-193`, `ag:165-169`).
   - A `noise_fn(obs, t)` is also required but unused for constant or tapered noise.
   - A π₀.₅ velocity v_π(x_s, s, observation), with x_s (B, 10, 32) and s = 1 the noise end, maps to it as follows:
     - reshape x between (B, A) and (B, 10, 32), with A = 320, or a 7-dim subset;
     - set s = 1 − t and v_ogpo = −v_π (see B5);
     - wrap the π₀.₅ `Observation`, or its prefix KV cache, in a closure, since OGPO passes a feature array;
     - noise and log-probs would cover every flattened dimension, including the 25 padded ones, unless the action
       space is restricted;
     - the chunk length the critic and runner assume (`horizon_length`) would be 10 in π₀.₅ terms.

## B5. Convention map: OGPO vs OpenPI π₀.₅

| Item | OGPO | OpenPI π₀.₅ (`third_party/openpi`) |
|---|---|---|
| Time direction, grid | t = 0 noise → 1 data; t_i = i/K, Δ = +1/K (`pg:183,187`) | t = 1 noise → 0 data; start 1.0, dt = −1/K, loop while t ≥ −dt/2 (`src/openpi/models/pi0.py:226-228, 274-278`) |
| Interpolation | x_t = (1−t)x₀ + t·A, x₀ ~ N(0, I) (`igp_targets_helper.py:137`) | x_t = t·ε + (1−t)·A (`pi0.py:199`) |
| Velocity | v = A − x₀ (`igp_targets_helper.py:138`) | u = ε − A (`pi0.py:200`); loss mean((v−u)²) (`pi0.py:214`) |
| Training time distribution | t ~ U(0, 1) (`igp_targets_helper.py:136`) | t = 0.001 + 0.999·Beta(1.5, 1) (`pi0.py:197`) |
| Euler step | x ← x + v/K (`pg:72`) | x ← x + dt·v, dt = −1/K (`pi0.py:271`) |
| Flow steps | 10 (`ogpo.yaml:130`) | 10 (`pi0.py:222`) |
| Action shape | flat (B, action_dim·H), H = 4 or 8 (`ag:2265`) | (B, 10, 32) (`pi0_config.py:25`, `pi05_libero` `action_horizon=10`, `training/config.py:745`); 7 real dims for LIBERO (`policies/libero_policy.py:100`) |
| Action normalization | none found; clip to [−1, 1] (`pg:77, 247`) | quantile: (x − q01)/(q99 − q01)·2 − 1 (`transforms.py:141-145`), inverse `transforms.py:175-180`; quantile for π₀.₅ (`training/config.py:187`) |
| Executed chunk | all H actions (`run:534-544`) | our eval: first 5 of 10 (`configs/experiment.yaml:32`; `examples/libero/main.py:29, 148`) |
| Our eager re-implementation | | `scripts/inspect_pi05.py` class `Flow` (`:149`), `sample()` dt = −1/K, x ← x + dt·v (`:205-215`) |

**Exact mapping.** With s = 1 − t, the same tensor x, and the same base noise:
- **Velocity:** v_OGPO(x, t) = dx/dt = −dx/ds = −v_π(x, s = 1 − t).
- **Grid:** OGPO's grid t_i = i/K is s_i = 1 − i/K = {1, 0.9, …, 0.1}, exactly OpenPI's evaluation points.
- **Euler step:** x + (1/K)·v_OGPO = x − (1/K)·v_π, identical to OpenPI's step.
- **Interpolant:** OGPO's (1−t)x₀ + t·A is OpenPI's s·ε + (1−s)·A.
- **Score:** OGPO's (t·v_OGPO − x)/(1−t) becomes −(x + (1−s)·v_π)/s in OpenPI variables.
- **Drift correction:** OGPO's tapered correction (σ²/2)(t·v_OGPO − x) becomes −(σ²/2)(x + (1−s)·v_π) when added to
  dx/dt; since ds = −dt, it enters dx/ds with a positive sign: +(σ²/2)(x + (1−s)v_π).

## Facts relevant to integration (neutral)

- OGPO's likelihood machinery is about 300 lines of JAX (`pg:123-414`). It depends only on `actor_fn` and `noise_fn`
  callables and on distrax Normal; the actor network itself is not involved.
- The shipped runs use the tapered schedule. There the old and new log-prob divisors match. The yaml default
  (constant noise plus normalization) does not.
- The PPO "old" policy is the EMA target actor, not a frozen copy from collection time. Advantage samples are drawn
  fresh at update time from the target actor at replay states, not taken from executed actions.
- Rollouts use Best-of-N SDE samples ranked by the critic; evaluation uses the ODE.
- In the online phase, the critic optimizer's base LR is `ppo_lr`, not `critic_lr` (`ag:2652-2654`).
- The public code trains a small MLP actor on frozen VLM features and never fine-tunes a VLA's action expert.
- No LIBERO environment, LIBERO launch script or MUSE encoder exists in the public code, although the paper reports
  LIBERO experiments.
- Version gaps (jax, flax, numpy, distrax) mean OGPO's code cannot be imported into our policy env as is.
- `pi_slow` and the KL/χ² options are internal regularizers against OGPO's own slow EMA policy and are off by default.
