# Setup

How to reproduce the environment for the π₀.₅-LIBERO continual-learning project on `xulab`, and how to run the tests.
All commands run from the repository root (`/mnt/data/users/bbouri/IRL-project`).

## Machine (audited 2026-10-22)

| Item | Value |
|---|---|
| Host | `autolab` (ssh alias `xulab`), Ubuntu 24.04.3 LTS, kernel 7.0.0-38 |
| CPU / RAM | 32 cores / 249 GB |
| GPU | 2 × RTX 5090 (32 GB, Blackwell sm_120), driver 580.178.04 (CUDA 13.0). **Shared**: this project uses GPU 1 only (`CLVLA_GPU`) |
| Disk | `/mnt/data` 26 TB (24 TB free) for all assets and outputs; `/` is 87% full, so no caches go there |
| Tools | git, `uv` 0.12.23 (`~/.local/bin`), system Python 3.12.3. Docker 29.1 is installed but the user is not in the `docker` group |

Because Docker is not usable, the project follows OpenPI's documented **non-Docker** LIBERO workflow, with two separate
environments (OpenPI and LIBERO need incompatible Python, NumPy and PyTorch versions):

| Env | Path | Python | Purpose | Key pins |
|---|---|---|---|---|
| policy | `third_party/openpi/.venv` | 3.11.17 | OpenPI model, policy server, checkpoint/data tests, unit tests | jax[cuda12] 0.5.3, flax 0.10.2, orbax-checkpoint 0.11.13, torch 2.7.1+cu126, numpy 1.26.4 (all from OpenPI `uv.lock`) |
| sim | `third_party/openpi/examples/libero/.venv` | 3.8.20 | LIBERO simulator, evaluation client, simulator tests | mujoco 3.2.3, robosuite 1.4.1, torch 1.11.0+cu113, numpy 1.22.4, plus `env/sim-extra-requirements.txt` (h5py 3.11.0, pytest 8.3.5) |

The two envs communicate only through OpenPI's websocket policy server (`openpi_client`). The `clvla` package
(`src/clvla`) is put on `PYTHONPATH` by the wrappers `scripts/py-policy` and `scripts/py-sim`, not installed. It is
Python-3.8-compatible and depends only on numpy and pyyaml. Exact package lists: `env/policy-env.freeze.txt`,
`env/sim-env.freeze.txt`, `env/openpi-uv.lock`.

## Reproduce from scratch

```bash
git clone <this repo> IRL-project && cd IRL-project      # or use the existing checkout on xulab
cp .env.example .env                                     # paths only; edit if your storage differs
bash scripts/setup_envs.sh                               # pinned OpenPI + LIBERO, both venvs, LIBERO path config, freezes
scripts/py-policy -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"
#   ^ only if the checkpoint is not in $OPENPI_DATA_HOME yet (12.4 GB, ~7 min, anonymous GCS, resumable by rerunning)
# Demonstrations: LIBERO-90 HDF5 from the official HF mirror (see docs/ASSETS.md), e.g.
#   scripts/py-policy -c "from huggingface_hub import snapshot_download as s; s('yifengzhu-hf/LIBERO-datasets', repo_type='dataset', revision='f13aa24a3da8c43c7225569f28c562979fa0e35a', local_dir='$LIBERO_DATA_ROOT', allow_patterns=['libero_90/*'])"
scripts/py-policy scripts/validate_assets.py --deep       # config, manifest, checkpoint (sha256), demos
scripts/py-policy scripts/verify_checkpoint_source.py     # checkpoint vs GCS size + MD5 (provenance)
bash scripts/run_tests.sh all                             # all test stages (needs GPU 1 free)
bash scripts/baseline_eval.sh --episodes-per-task 10      # checkpoint baseline
```

`setup_envs.sh` is idempotent. It checks out OpenPI `215abfb217dbac7d5f1273282331b9b1866c0479` with its official
LIBERO submodule `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c`, verifies both SHAs and the LIBERO remote URL, and runs
`uv sync --frozen`. It then builds the sim env exactly as `examples/libero/README.md` ("Without Docker") describes,
adds the two extra packages, writes `.libero/config.yaml`, and records the freezes. Caches go to `/mnt/data`
(`UV_CACHE_DIR`, `UV_PYTHON_INSTALL_DIR`, `HF_HOME`, `OPENPI_DATA_HOME`).

## Configuration and paths

- `.env` / `.env.example`: storage paths, `CLVLA_GPU`, and the default server port. Never put credentials here.
- `env.sh`: sourced by every script; exports `OPENPI_DIR`, `LIBERO_DIR`, interpreter paths, `LIBERO_CONFIG_PATH`
  (project-local, so `~/.libero` from other projects is never used), and `MUJOCO_GL=egl`.
- `configs/experiment.yaml`: the frozen protocol (see `docs/EXPERIMENT_PROTOCOL.md`).
- `configs/task_manifest.yaml`: the frozen task sequence.

## Tests

The tests are staged; each stage runs in the env it needs. Tests that need the other env **skip with a reason**, so a
skip is never a hidden pass.

| Stage | Command | Needs | What it checks |
|---|---|---|---|
| unit (policy env) | `scripts/py-policy -m pytest tests/unit` | nothing (CPU) | config/manifest/asset/data/metrics/metadata logic on synthetic fixtures, mocked env + policy; OpenPI's real `pi05_libero` transform chain on synthetic data |
| unit (sim env) | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/py-sim -m pytest tests/unit` | nothing | same tests on Python 3.8 (portability) |
| env | `tests/integration/test_environment.py` in each env | — | imports and pinned versions |
| data | `scripts/py-policy -m pytest -s tests/integration/test_data_real.py` | demos + checkpoint assets (CPU) | a real demo through OpenPI's training transforms with bundled norm stats; raw-HDF5 statistics vs checkpoint norm stats |
| simulator | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/py-sim -m pytest -s tests/integration/test_simulator.py` | sim env, EGL | all 10 tasks resolve in official LIBERO-90 and instantiate; reset, observe, safe actions; demo ↔ simulator conventions |
| checkpoint | `CLVLA_USE_GPU=1 scripts/py-policy -m pytest -s tests/integration/test_checkpoint.py` | GPU | loads the exact `pi05_libero` checkpoint, compatibility gate, bundled norm stats in use, inference on valid observations |
| server + e2e | `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 scripts/py-sim -m pytest -s tests/integration/test_server.py tests/integration/test_end_to_end.py` | GPU | official server ↔ client round trip; simulator → policy → simulator episode capped at 60 steps, with video |

`bash scripts/run_tests.sh {cpu|sim|all}` runs the stages in order and prints a summary. Logs go to `logs/tests/<stamp>/`.
CPU-only machines: use `cpu`; the GPU stages are skipped.

## Shared-GPU etiquette (xulab)

- Only GPU 1 is used, set by `CLVLA_GPU`. The policy server sets `CUDA_VISIBLE_DEVICES=1` and
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`, which uses about 8.7 GB. MuJoCo EGL rendering uses `MUJOCO_EGL_DEVICE_ID=1`.
- `scripts/gpu_guard.sh` refuses to start if another user has compute processes on that GPU. It is read-only and
  never touches other processes. Every GPU entry point calls it.
- Unit, data and simulator tests do not use CUDA. The policy-env wrapper forces CPU unless `CLVLA_USE_GPU=1`.
- Long runs: `setsid nohup bash scripts/baseline_eval.sh ... > logs/<name>.log 2>&1 < /dev/null &`. The server is
  stopped automatically when the evaluation ends.

## Known environment quirks

- **hydra pytest plugin (sim env):** OpenPI's pinned LIBERO requirements install `hydra-core` 1.2.0 without
  `omegaconf`, so hydra's auto-loaded pytest plugin crashes pytest. LIBERO's evaluation path never imports hydra, so
  the env is left as pinned and pytest runs with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` in the sim env.
- **LIBERO first-import prompt:** without `LIBERO_CONFIG_PATH`, LIBERO asks for dataset paths interactively and writes
  `~/.libero/config.yaml`. `env.sh` points it at the project-local `.libero/` instead (generated by `setup_envs.sh`).
- **Blackwell GPUs:** jax 0.5.3 runs on the RTX 5090 because the locked `nvidia-cuda-nvcc-cu12` 12.9 provides an
  sm_120-capable `ptxas`. No system CUDA toolkit is needed.
- **Policy server binding:** OpenPI's `serve_policy.py` binds `0.0.0.0`; the client connects to `127.0.0.1`.
- **GPU nondeterminism:** with default XLA flags, each server process picks GPU kernels by autotuning. Actions are
  bit-identical within a process but differ by about 1e-4 between processes, and closed-loop rollouts amplify that
  into different episode outcomes. In a measured case, the same 3 episodes of task 28 gave 2/3 and then 0/3.
  `serve_policy.sh` therefore sets `XLA_FLAGS=$CLVLA_XLA_FLAGS`, which defaults to
  `--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0`. With it, single inferences are bit-identical across
  processes, but a full 100-episode rerun still differed in 7 episodes (see SETUP_REPORT §6). It costs about 35% speed. The flags are recorded in each run's metadata.
