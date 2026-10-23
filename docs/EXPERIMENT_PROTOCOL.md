# Experiment protocol

Study: **How should vision-language-action policies continually learn new skills?** Starting point: the official
π₀.₅-LIBERO checkpoint. New skills: a custom, frozen 10-task sequence from LIBERO-90.

Status: **protocol v1.0.0, frozen 2026-10-22** (`configs/experiment.yaml`, `protocol_sha256 a3d55cd82c5a…`).
**Task manifest v1.0.0** (`configs/task_manifest.yaml`, `content_sha256 b1a0789d63ea…`). Only infrastructure and the
checkpoint baseline exist so far. No continual-learning algorithm has been implemented.

## 1. Freezing rules

- Everything under `protocol:` in `experiment.yaml` is hashed. `clvla.config` refuses to load the file if the hash is
  stale, so no frozen setting can change silently.
- The task manifest carries a content checksum. `clvla.tasks` refuses edited copies, and
  `scripts/build_task_manifest.py --check` re-derives it from LIBERO metadata.
- To change either: edit it, bump the version, re-freeze (`scripts/freeze_config.py` / regenerate the manifest), and
  add an entry to the change log (§9). Never reuse results across versions.
- Only `evaluation:` (episode count, split, videos, server address) varies per run. Its values are written into each
  run's `run_metadata.json`.

## 2. Frozen settings and their source

| Setting | Value | Source |
|---|---|---|
| Checkpoint | `gs://openpi-assets/checkpoints/pi05_libero`, OpenPI config `pi05_libero`, fingerprint `cd22f6be…` | task requirement; `docs/ASSETS.md` |
| Normalization | bundled `assets/physical-intelligence/libero/norm_stats.json`, quantile normalization | OpenPI `create_trained_policy` |
| Action / state | 7-D OSC delta (6) + gripper (1, −1 open / +1 close); 8-D state = eef pos (3) + axis-angle (3) + gripper qpos (2) | OpenPI `examples/libero/main.py`, `libero_policy.py` |
| Action chunk | horizon 10; execute the first 5 actions, then re-query | `pi05_libero` config; main.py `replan_steps=5` |
| Images | render 256×256; rotate 180°; `resize_with_pad` to 224; agentview + wrist; right-wrist slot zero-masked | main.py `LIBERO_ENV_RESOLUTION`, `LiberoInputs` |
| Episode | 10 dummy settle steps `[0,0,0,0,0,0,−1]`, then at most **400** policy steps; success = env `done` | main.py `libero_90` branch ("longest training demo has 373 steps") |
| Seeds | global 7; eval seeds [7]; `env.seed(7)` per task env | main.py default `seed=7` |
| Demo split | per task, `demo_0…demo_44` train (45), `demo_45…demo_49` validation (5), in numeric order | this project |
| Init-state split | official 50 init states per task: **0–9 selection** (any model/checkpoint/hyperparameter choice), **10–49 test** (reporting only) | this project; OpenPI evaluates on 0–49 |
| Demo images | 128×128 (official HDF5) | LIBERO data; see §6 |

## 3. Task selection (LIBERO-90 → 10 tasks)

`scripts/build_task_manifest.py` applies these rules to the pinned official LIBERO metadata and the real demo files.
It is deterministic and re-derivable with `--check`.

- **R1 verified:** the task resolves in official `libero_90`; its demo file has exactly 50 demos whose stored
  instruction equals the benchmark instruction; the file is in the HF-API dataset lock; 50 init states. *All 90
  tasks pass.*
- **R2 no instruction overlap:** the instruction is neither identical to nor a substring of any instruction in the
  four fine-tuning suites (libero_spatial/object/goal/10). *Excludes 20 and 44 ("turn on the stove" = libero_goal),
  77 (= libero_10 book task), and the libero_10 sub-goals 24, 67, 68, 70, 72.*
- **R3 no libero_10 scenes:** the scene is not used by libero_10 (libero_10 tasks are LIBERO-100 scenes shared with
  LIBERO-90; libero_spatial/object/goal use their own scenes). *Excludes KITCHEN_SCENE3/4/6/8, LIVING_ROOM_SCENE1/2/5/6
  and STUDY_SCENE1, leaving 53 candidates in 11 scenes.*
- **R4 horizon fits:** the longest demo has at most 375 steps, below the 400-step cap. *All candidates pass.*
- **R5 diversity:** at most one task per scene.
- **R6 skill mix (rule-based from the instruction):** articulated 2, container 2, spatial_relation 4, multi_step 2.
- **R7 least similar to pretraining:** among all selections satisfying R1–R6 (exhaustive search), minimize the summed
  nearest-neighbour word-Jaccard similarity to the pretraining instructions; ties go to the smaller index tuple.
- **Order:** sort by index, then `random.Random(0).shuffle`. The order is stored explicitly, and code never relies on
  suite ordering.

### Frozen sequence (manifest v1.0.0)

| Pos | Idx | Scene | Category | Instruction | Demo len (min–max) | Nearest pretraining instruction (Jaccard) |
|---|---|---|---|---|---|---|
| 0 | 78 | STUDY_SCENE2 | container | pick up the book and place it in the front compartment of the caddy | 112–211 | libero_10: "…in the **back** compartment of the caddy" (0.71) |
| 1 | 84 | STUDY_SCENE3 | spatial_relation | pick up the red mug and place it to the right of the caddy | 102–161 | libero_10 book/caddy (0.33) |
| 2 | 16 | KITCHEN_SCENE2 | spatial_relation | stack the black bowl at the front on the black bowl in the middle | 103–191 | libero_spatial bowl→plate (0.20) |
| 3 | 56 | LIVING_ROOM_SCENE3 | container | pick up the butter and put it in the tray | 88–131 | libero_object "pick up the butter and place it in the basket" (0.33) |
| 4 | 35 | KITCHEN_SCENE7 | articulated | open the microwave | 102–177 | libero_goal "open the middle drawer of the cabinet" (0.20) |
| 5 | 42 | KITCHEN_SCENE9 | spatial_relation | put the frying pan under the cabinet shelf | 142–213 | libero_goal "put the bowl on top of the cabinet" (0.25) |
| 6 | 28 | KITCHEN_SCENE5 | articulated | close the top drawer of the cabinet | 58–110 | libero_10 "…bottom drawer of the cabinet and close it" (0.38) |
| 7 | 3 | KITCHEN_SCENE10 | multi_step | put the butter at the back in the top drawer of the cabinet and close it | 169–212 | libero_10 "put the black bowl in the bottom drawer of the cabinet and close it" (0.36) |
| 8 | 89 | STUDY_SCENE4 | spatial_relation | pick up the book on the right and place it under the cabinet shelf | 101–179 | libero_10 book/caddy (0.30) |
| 9 | 63 | LIVING_ROOM_SCENE4 | multi_step | stack the left bowl on the right bowl and place them in the tray | 165–327 | libero_spatial bowl→plate (0.18) |

Full names, BDDL, init-state and demo paths, demo sha256 values and the exclusion list are in the manifest.

### Overlap with the checkpoint's fine-tuning data: what is and is not claimed

- **Verified:** no selected instruction equals, or is a sub-goal of, any LIBERO-Spatial/Object/Goal/10 instruction.
  No selected task uses a libero_10 scene. All 10 are genuine LIBERO-90 tasks, resolved against the official
  benchmark.
- **Not claimed: strict disjointness.** All tasks share the Panda robot, the OSC action space, the camera setup and
  many object meshes with the fine-tuning suites. Several are close paraphrases of pretraining skills: task 78 differs
  from a libero_10 task only in "front" vs "back" compartment; task 56 is libero_object's butter pick-and-place with a
  tray instead of a basket; task 28 closes a drawer like libero_10's drawer task. `pi05_base` pretraining data is not
  public, so overlap with it cannot be checked. Treat "new skill" as *new task instance* (new scene and instruction),
  and report per-task results with the overlap column above.

## 4. Data use

- Demonstrations: the official LIBERO-90 HDF5 for the 10 manifest tasks only (`docs/ASSETS.md`). Train on demos 0–44
  and validate on 45–49 (positional, numeric order).
- Preprocessing path (implemented, not yet used for training): `clvla.data.episode_to_openpi_frames` emits
  `physical-intelligence/libero`-style samples:
  - `image` / `wrist_image`: rotated 180°;
  - `state`: `ee_states` ++ `gripper_states`;
  - `actions`: chunks of 10 raw actions, clamped at the episode end;
  - `prompt`: the instruction.

  OpenPI's own transform chain then applies repack → `LiberoInputs` → quantile `Normalize` with the bundled stats →
  resize to 224 + tokenize + pad. No other action or state transformation exists. In particular there is no delta
  conversion, because `extra_delta_transform=False` for `pi05_libero`.

## 5. Evaluation procedure

- `bash scripts/baseline_eval.sh [--episodes-per-task N] [--split selection|test] [--save-videos]` starts a **fresh**
  official OpenPI server (its sampling RNG starts at `jax.random.key(0)`, so request order makes it reproducible) on
  the guarded GPU, runs `scripts/run_eval.py` in the sim env, and stops the server.
- For each task in manifest order: one fresh `OffScreenRenderEnv` at 256 px, `env.seed(7)`; episode *i* uses init
  state `split[i]`. The loop is OpenPI main.py's loop (`clvla.evaluation.run_episode`, unit-tested against a scripted
  env).
- Outputs go to `outputs/baseline/<UTC>_<name>/`:
  - `run_metadata.json`: resolved config, protocol and manifest hashes, seeds, checkpoint fingerprint, git SHAs of the
    project, OpenPI and LIBERO with dirty flags, package versions, GPU, server command, init-state indices, and the
    list of deviations;
  - config and manifest snapshots;
  - `episodes.partial.jsonl` (written live), `episodes.csv` / `episodes.jsonl`;
  - `per_task.csv` and `summary.json`: pooled and task-mean success, Wilson 95% CIs, infra-error counts;
  - optional videos.
- Rerun a saved evaluation: `bash scripts/baseline_eval.sh --from-run outputs/baseline/<run>`. It uses the run's
  snapshots and evaluation settings and refuses to run if the checkpoint fingerprint changed.
- Separation: checkpoint baselines are written under `baseline/`, smoke/control runs under `smoke/`. Future runs go to
  `finetune/` and `continual/`. `make_run_dir` never overwrites.
- Infrastructure control: `bash scripts/control_check.sh` runs the same loop, and OpenPI's unmodified main.py, on
  libero_spatial (in-distribution; OpenPI reports 98.8%) to show the pipeline itself is sound.

### Deviations from OpenPI's documented LIBERO evaluation (also stored in every run's metadata)

1. A custom 10-task LIBERO-90 subset, not an official suite. **Not comparable to published LIBERO numbers.**
2. N episodes per task (baseline: 10) on init states 10…10+N−1 of the test split. OpenPI uses 50 episodes on 0–49.
3. Exceptions are recorded as `infra_error` rather than silently counted as failures. Both rates are reported.
4. NumPy is reseeded per episode with a derived seed. This has no effect on the simulator or the policy.

Everything else matches main.py at the pinned commit: rendering, rotation, resizing, state, replanning, settle
steps, max steps, `env.seed`, a fresh env per task, and the success criterion.

## 6. Known data issues: decisions needed before any fine-tuning

These do not affect the checkpoint baseline. They do affect how demos should be fed to behavior cloning, so they are
flagged here rather than silently "fixed".

1. **Observation/action timing in the official HDF5.** LIBERO's `scripts/create_dataset.py` stores the observation
   *returned by* `env.step(actions[t])` as `obs[t]`. So `obs[t]` is the state **after** `actions[t]`, which equals
   `sim(states[t+1])`. This is verified in `tests/integration/test_simulator.py`: the aligned residual is ≤ 1.7e-3,
   versus a 0.014 median unaligned. The evaluation loop observes and then acts, so the HDF5's (`obs[t]`,
   `actions[t]`) pairs are shifted by one step relative to deployment. The loader currently keeps LIBERO's pairing
   unchanged.
2. **Image resolution.** The demos are 128×128. `pi05_libero` was trained on 256×256 re-renders and is evaluated at
   256. Both are resized to 224, but upsampled 128 px images differ in sharpness from what the checkpoint saw.

Recommended resolution (needs approval because it changes the training data): regenerate the 10 tasks' demos by
replaying each demo's MuJoCo `states` in the official env at 256 px. This is the same idea as OpenVLA's re-rendering,
and the restore → observe path is already verified by the convention test. It produces pre-action observations at the
deployment resolution, fixing both issues while keeping the official trajectories and actions. Simpler alternatives:
(a) shift the pairing by one (`obs[t−1]` with `actions[t]`, re-rendering only the first frame), or (b) use the HDF5
as-is, as LIBERO's own lifelong-learning baselines do. Option (b) should be documented as a known confound.

## 7. Metrics (baseline now; continual learning later)

- Per task: success rate over N episodes with a Wilson 95% CI, infra errors, and mean steps.
- Aggregate: the task-mean success rate (each task weighted equally) and the pooled rate.
- Continual-learning metrics are not implemented yet; they will reuse this per-task evaluation after each stage. The
  plan is a success matrix R[i, j] = success on task j after learning task i, plus forward/backward transfer and
  forgetting.

## 8. Baseline result

See `docs/SETUP_REPORT.md` and `outputs/baseline/` (run id recorded there).

## 9. Change log

| Date | Item | Change |
|---|---|---|
| 2026-10-22 | protocol 1.0.0, manifest 1.0.0 | initial freeze |
