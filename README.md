# How should VLA policies continually learn new skills? (π₀.₅-LIBERO × LIBERO-90)

Reproducible infrastructure for continual-learning experiments that start from the official **π₀.₅-LIBERO**
checkpoint (OpenPI `pi05_libero`) and learn a **custom, frozen 10-task sequence drawn from LIBERO-90** (not the
official `libero_10` suite). This stage covers infrastructure and the checkpoint baseline only. No continual-learning
algorithm exists yet.

| Read | For |
|---|---|
| `docs/SETUP.md` | machine, environments, exact setup commands, tests (CPU vs simulator vs GPU) |
| `docs/ASSETS.md` | checkpoint and demonstration sources, revisions, paths, checksums, validation |
| `docs/EXPERIMENT_PROTOCOL.md` | frozen settings, task selection, splits, evaluation, deviations, open data decisions |
| `docs/SETUP_REPORT.md` | what was run, test results, baseline numbers, blockers, next steps |

```bash
bash scripts/setup_envs.sh                              # pinned OpenPI + official LIBERO, two uv envs
scripts/py-policy scripts/validate_assets.py --deep      # config, manifest, checkpoint, demos
bash scripts/run_tests.sh all                            # unit + integration stages (GPU stages use GPU 1 only)
bash scripts/baseline_eval.sh --episodes-per-task 10     # official checkpoint on the 10 frozen tasks
```

Layout: `src/clvla/` (config, assets, tasks, data, policy adapter, evaluation, experiment metadata), `scripts/`,
`tests/{unit,integration}/`, `configs/` (frozen `experiment.yaml`, `task_manifest.yaml`, asset locks, LIBERO
snapshot), `env/` (environment freezes), `third_party/openpi` (pinned submodule, which contains the official LIBERO
submodule). Large assets and run outputs live outside git (see `.gitignore` and `.env.example`).
