# Working in this repository (people and coding agents)

π₀.₅-LIBERO continual-learning project: on-policy distillation (OPD) from per-task teachers into one student, on a frozen
sequence of LIBERO-90 tasks. Start with `README.md` (status, setup, how to run), then `docs/cl_opd/README.md` for the
continual-learning results and open questions, and `docs/OPD_EXPERIMENT_REGISTRY.md` for every earlier run.

## Setup in one screen

1. `cp .env.example .env` and set the paths. Every machine-specific path lives in `.env`; the code uses paths relative
   to the repository root (`outputs/` must be at the root; a symlink to a large disk is fine).
2. `bash scripts/setup_envs.sh` builds two uv environments from the pinned OpenPI submodule: `scripts/py-policy`
   (Python 3.11, JAX, OpenPI) and `scripts/py-sim` (Python 3.8, LIBERO simulator). Always run Python through these
   wrappers; they source `env.sh` (which loads `.env`) and set `PYTHONPATH`. `py-policy` is CPU-only unless
   `CLVLA_USE_GPU=1`.
3. Assets: base checkpoint (README, Setup), demonstrations (README, Continual learning, step 1), teachers
   (`scripts/py-policy scripts/fetch_cl_assets.py`). Hugging Face repo `raoulb091602/vla-skill-consolidation` is public.
4. Check: `scripts/py-policy -m pytest tests/unit` (CPU, a few seconds) and
   `sha256sum -c tests/ogpo_port/frozen_task5_9.sha256`.

## Rules (do not break these)

- **Frozen modules.** The files listed in `tests/ogpo_port/frozen_task5_9.sha256` (OGPO sampler, path log-probability,
  path KL and their tests) are validated and must stay byte-identical. Every launcher in `scripts/cl/` refuses to run
  if they changed. Change behaviour in new code or behind a new flag instead.
- **Held-out data.** With profile n25, demo_25..49 are held out (n10: demo_10..49). Never train on them, select
  checkpoints with them, or tune thresholds on them. Reserved seeds 12000-12007, 13000-13007 and 15000 stay unused.
- **Outputs are append-only.** A new experiment gets a new config file (with its own `name:`) and a new timestamped
  run directory under `outputs/`. Never edit, overwrite or delete an earlier run's files; the driver refuses to resume
  a run whose config hash changed.
- **Defaults reproduce the past.** New trainer options are off by default, and with them off the code must reproduce
  the earlier results exactly (for example `--traj-clip-k 0`). Keep it that way and add a test when you add an option.
- **Preregister.** Write down the success thresholds of an experiment before looking at its results
  (`docs/OPD_*_PREREGISTRATION.json` are examples). Report failures as failures.
- **Shared GPUs.** On a machine other people use, never touch their processes. `scripts/gpu_guard.sh` (called by
  every launcher) refuses a GPU another user is on; do not bypass it without the owner's agreement.
- **Secrets.** No API keys or tokens in files, configs, command lines or logs. W&B keys are read from stdin
  (`scripts/launch_training_wandb.sh` shows the pattern).
- **Third-party code.** Do not edit `third_party/openpi` (pinned submodule, checked by `setup_envs.sh`).
- **No machine-specific paths** in committed files: add a variable to `.env.example` instead.

## Where things are

| What | Where |
|---|---|
| Continual-learning driver and config | `scripts/cl_opd_run.py`, `configs/cl_opd.yaml` (smoke: `configs/cl_opd_smoke.yaml`) |
| Launchers (one task, smoke, resume check, t063 no-replay ablation) | `scripts/cl/` |
| Trainer (OPD, pathwise path-KL gradient, replay, clipping) | `scripts/opd_lora_train.py` (+ `opd_lora_student.py`, `opd_lora_collect.py`) |
| Buffer, sampler, success matrix, clipping rule | `src/clvla/cl_opd/` |
| Flow-policy likelihood and KL (frozen) | `src/clvla/ogpo/` |
| Evaluation | `scripts/combined_eval.py`, `scripts/combined_serve.py` |
| Gradient analysis and pre-flight diagnostics | `scripts/cl_opd_grad_*.py`, `scripts/cl_opd_clip_validate.py`, `scripts/cl_opd_preflight_*.py` |
| Unit / integration tests | `tests/unit/`, `tests/integration/` (GPU), `tests/ogpo_port/` |

The analysis and pre-flight scripts read the two original continual-learning runs
(`outputs/cl_opd/full_20261108T214604Z`, `outputs/cl_opd/t063_noreplay_20261109T150326Z`) by their relative paths;
those runs are not on Hugging Face yet, so these scripts only run where the runs are.

## Checks before a pull request

`scripts/py-policy -m pytest tests/unit -q`, the frozen-hash check, and for trainer changes a dry run
(`DRYRUN=1 scripts/py-policy scripts/cl_opd_run.py --config <cfg> --run outputs/cl_opd/dry --task 1 --dry-run`) plus the
smoke run (`bash scripts/cl/smoke.sh`) on a GPU.
