# Setup report: 2026-10-23

Everything below was actually run on `xulab` (`/mnt/data/users/bbouri/IRL-project`). Nothing is claimed that was not
executed. No continual-learning algorithm has been started. Git history is local (no remote):
`fbbae19` → `db16617` → `cf529ff` → `9917279` → (this report's commit).

## 1. Definition of done: status

| Requirement | Status | Evidence |
|---|---|---|
| Reproduce the environment | ✅ | `bash scripts/setup_envs.sh` (rerun end to end, rc=0; pins verified) |
| Validate checkpoint | ✅ | `validate_assets.py --deep` (sha256 of 16 files) + `verify_checkpoint_source.py` (GCS size + MD5) |
| Validate real demonstrations | ✅ | all 90 LIBERO-90 files sha256-match the HF API lock (`--deep-all`, 66.66 GB); 10 manifest tasks pass full schema checks |
| Resolve all ten task IDs | ✅ | `tests/integration/test_simulator.py`: resolve in official LIBERO-90 and instantiate all 10 |
| Run unit tests | ✅ | 96 passed (py3.11), 94 passed + 2 env-specific skips (py3.8) |
| Simulator → policy inference smoke test | ✅ | `test_end_to_end.py`: 60-step capped episode through the official server, 0 errors, video saved |
| Baseline results file | ✅ | `outputs/baseline/<run>/{episodes.csv,episodes.jsonl,per_task.csv,summary.json}` (§6) |

## 2. Exact commands run (in order)

```bash
# audit
ssh xulab 'hostname; cat /etc/os-release; nproc; free -g; nvidia-smi; df -h /mnt/data /; which git uv docker; ...'
# Phase 0/1: code + envs
git clone --recurse-submodules https://github.com/Physical-Intelligence/openpi.git third_party/openpi   # 215abfb
(cd third_party/openpi && GIT_LFS_SKIP_SMUDGE=1 uv sync)                                                 # policy env, 2 min
uv venv --python 3.8 examples/libero/.venv && uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt \
   --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client -e third_party/libero h5py==3.11.0 pytest==8.3.5                # sim env
openpi.shared.download.maybe_download("gs://openpi-assets/checkpoints/pi05_libero")                      # 12.4 GB, 399 s
bash scripts/setup_envs.sh                              # idempotent re-run of all of the above + freezes: rc=0
scripts/py-policy scripts/freeze_config.py              # protocol_sha256 a3d55cd8...
scripts/py-policy scripts/build_locks.py                # checkpoint.lock.json, dataset.lock.json (HF API)
# Phase 2
scripts/py-sim scripts/build_task_manifest.py           # manifest v1.0.0, content_sha256 b1a0789d...
scripts/py-sim scripts/build_task_manifest.py --check   # OK: re-derives exactly
scripts/py-policy scripts/validate_assets.py --deep     # ALL ASSETS VALID (37 s)
scripts/py-policy scripts/validate_assets.py --deep-all # ALL ASSETS VALID (311 s)
scripts/py-policy scripts/verify_checkpoint_source.py   # OK: every local file matches GCS size and MD5
# Phases 4-5
bash scripts/run_tests.sh all                           # all 8 stages rc=0 (table §3)
# Phase 6
bash scripts/baseline_eval.sh --episodes-per-task 10    # pilot (default XLA flags)
bash scripts/control_check.sh 2                         # in-distribution control
bash scripts/baseline_eval.sh --from-run <pilot> --kind smoke --positions 3,6 --episodes-per-task 3   # x2: not reproducible
CLVLA_USE_GPU=1 scripts/py-policy tmp/det_probe.py      # x4: cross-process nondeterminism found; XLA flags fix it
bash scripts/baseline_eval.sh --from-run <pilot> --kind smoke --positions 3,6 --episodes-per-task 3   # x2 with flags: identical
bash scripts/baseline_and_repro.sh 10                   # canonical baseline + full re-execution + comparison
```

## 3. Tests

`bash scripts/run_tests.sh all` (logs `logs/tests/20261023T041814Z/`):

| Stage | Env | Result |
|---|---|---|
| unit-policy-env | py3.11, CPU | **96 passed** |
| unit-sim-env | py3.8, CPU | **94 passed**, 2 skipped (OpenPI-transform tests: policy env only) |
| env-policy / env-sim | each env | **1 passed** each; other env's checks skipped by design |
| data-real | py3.11, CPU | **2 passed**: a real demo of task 78 through OpenPI's training transforms (98.5% of normalized actions and 98.4% of states within ±1.05); raw pretraining-suite HDF5 statistics consistent with the bundled norm stats |
| simulator | py3.8, EGL GPU 1 | **4 passed**: all 10 tasks resolve and instantiate; reset, observe and safe actions; demo ↔ simulator conventions (§7) |
| checkpoint-gpu | py3.11, GPU 1 | **5 passed**, 1 skipped (sim-only import test): compatibility gate, bundled norm stats inside the policy, inference `(10, 7)` finite, about 63 ms per call, load 6.9 s |
| server-e2e-gpu | py3.8 + server | **2 passed**: official server round trip; a 60-step episode of task 78 (12 policy calls, 0 errors, 1.6 s, not successful within the 60-step smoke cap; success is not required) |

Not tested: training and fine-tuning code (out of scope at this stage).

## 4. Checkpoint identity

`gs://openpi-assets/checkpoints/pi05_libero`, OpenPI config `pi05_libero`, at
`/mnt/data/users/bbouri/cache/openpi/openpi-assets/checkpoints/pi05_libero`. It has 16 files (12.44 GB) and lock
fingerprint `cd22f6becde68e647885fb9beb54a6a8fc296b117d87b6de03dea9d53dd07fee`. All files match GCS MD5. Bundled norm
stats `assets/physical-intelligence/libero/norm_stats.json` are used as-is.

## 5. Dataset provenance and selected tasks

Official LIBERO HDF5 from `https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets` at
`f13aa24a3da8c43c7225569f28c562979fa0e35a`. This is the HF source added to LIBERO's own downloader upstream; the
pinned LIBERO code points to the equivalent UT Box zip. License: CC BY 4.0. It is reused at
`/mnt/data/users/bbouri/cl/data/libero/libero_90`, and all 90 files are sha256-verified against the HF API.

Frozen sequence (manifest v1.0.0), LIBERO-90 indices in order: **78, 84, 16, 56, 35, 42, 28, 3, 89, 63** (names,
instructions and rationale in `docs/EXPERIMENT_PROTOCOL.md` §3).

## 6. Baseline: official checkpoint, unmodified

Settings: 10 episodes per task on test init states 10–19, seed 7, max 400 steps, OpenPI main.py loop.

**Canonical run** (deterministic XLA flags): `outputs/baseline/20261023T044001Z_pi05_libero_official_det`

| Pos | Idx | Task | Success (Wilson 95% CI) |
|---|---|---|---|
| 0 | 78 | book → front compartment of caddy | 0/10 (0.00–0.28) |
| 1 | 84 | red mug → right of caddy | 0/10 |
| 2 | 16 | stack front bowl on middle bowl | 0/10 |
| 3 | 56 | butter → tray | **9/10** (0.60–0.98) |
| 4 | 35 | open the microwave | 1/10 |
| 5 | 42 | frying pan under cabinet shelf | 0/10 |
| 6 | 28 | close top drawer | **4/10** (0.17–0.69) |
| 7 | 3 | butter → top drawer, close it | 0/10 |
| 8 | 89 | book on right → under shelf | 0/10 |
| 9 | 63 | stack bowls → tray | 0/10 |
| | | **Total** | **14/100**, task mean 0.14, 0 infra errors, about 17 min |

Reproducibility: the full re-execution (`bash scripts/baseline_eval.sh --from-run <canonical run>` →
`outputs/smoke/20261023T045728Z_repro_of_…`) also gave **14/100**. However, **7 of 100 episodes differed**: 1 on
task 35 and 6 on task 28. The per-task counts changed accordingly (35: 1 → 0, 28: 4 → 5). The other 8 tasks (80
episodes) matched exactly in both success and step count. The deterministic XLA flags removed the cross-process
policy nondeterminism; the earlier 6-episode check and the single-inference probe were both bit-identical. A residual
source remains, which shows up on the articulated tasks (microwave, drawer). It is not yet identified: suspects are
contact-rich MuJoCo or EGL behaviour, or XLA ops not covered by the flag. **Treat single-run per-task numbers as
having roughly ±1–2 episodes of rerun noise on top of the CI.

**Pilot run** (default XLA flags; superseded but kept): `outputs/baseline/20261023T042043Z_pi05_libero_official`.
Result: 14/100 (task mean 0.14); 56 → 10/10, 28 → 3/10, 63 → 1/10, others 0/10; 0 infra errors; 12.6 minutes.

**Infrastructure control** (`scripts/control_check.sh 2`, libero_spatial, in-distribution): **20/20** with this
pipeline (`outputs/smoke/20261023T042445Z_control_libero_spatial`) and **20/20** with OpenPI's unmodified
`examples/libero/main.py` on the same server. The low LIBERO-90 numbers therefore reflect the policy, not the
pipeline.

These numbers are for a custom LIBERO-90 subset with 10 episodes per task. Wilson 95% CIs are wide (e.g. 3/10 →
0.11–0.60), and the numbers are **not comparable to published LIBERO benchmarks**.

## 7. Findings that need a decision before fine-tuning (nothing was changed silently)

1. **Demo observation/action offset.** The official LIBERO HDF5 stores the observation *after* `actions[t]` as
   `obs[t]` (LIBERO `scripts/create_dataset.py`). This is verified: the aligned residual is ≤ 1.7e-3, versus a 0.014
   median unaligned.
2. **Demo resolution.** The demos are 128 px, while the checkpoint was trained and is evaluated at 256 px.
   **Recommendation:** re-render the 10 tasks' demos from their MuJoCo `states` at 256 px in the official env. This
   fixes both issues while keeping the official trajectories (see protocol §6).
3. **Task 56 has no headroom** (10/10 zero-shot; it is a near-copy of a libero_object task). Keep it as a "known
   skill" control, or replace it in manifest v1.1 using an added baseline-headroom rule evaluated on the selection
   init states (protocol §8).
4. **Evaluation budget.** 10 episodes per task cannot separate, for example, 30% from 50%. For reported numbers use
   all 40 test init states per task, which is about 70 minutes per evaluation on one GPU at the current speed.

## 8. Known issues (non-blocking)

- Docker is unusable (not in the `docker` group), so OpenPI's documented non-Docker path with two uv envs is used.
- OpenPI's pinned LIBERO requirements ship `hydra-core` without `omegaconf`. The sim env runs pytest with
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`; LIBERO evaluation does not use hydra.
- Default XLA autotuning makes rollouts irreproducible across server processes, so deterministic flags are now the
  default. They cost about 35% speed on inference-bound episodes (about 10.5 s vs 7.7 s per 400-step episode). Even
  with the flags, 7/100 episodes differed in a full rerun (§6); the cause is not yet identified.
- GPUs are shared: everything is guarded to GPU 1 and refuses to start if another user's compute process is present.
- `pi05_base` pretraining data is not public, so overlap with it cannot be audited.

## 9. Next steps (in order; each needs your go-ahead)

1. Decide items 7.1–7.3. If you choose re-rendering: build `scripts/rerender_demos.py` with a validation test
   (replayed states match, success at the final state), then convert the 10 tasks to a LeRobot dataset in OpenPI's
   format.
2. A single-task fine-tuning smoke run (one task, a few hundred steps, GPU 1) to measure step time and memory before
   planning the sequential runs.
3. Find the residual rerun nondeterminism on tasks 35 and 28: replay the recorded action sequence open-loop to separate simulator from policy.
4. Define the continual-learning metrics module (success matrix, forgetting, forward/backward transfer) on top of
   `clvla.evaluation`, with unit tests on synthetic matrices.
5. Then the sequential fine-tuning baseline (naive sequential FT), before any novel method.
