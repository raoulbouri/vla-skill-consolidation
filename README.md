# vla-skill-consolidation

**How should vision-language-action policies continually learn new skills?** Course project, 16-831 Introduction to Robot Learning, Fall 2026, CMU (Sreyas Venkataraman, Vinit Awale, Saumya Shah, Raoul Bouri).

We study sequential imitation learning from the open-source **π₀.₅-LIBERO** checkpoint (OpenPI `pi05_libero`) on a frozen sequence of ten held-out **LIBERO-90** tasks, along two axes: expert capacity (LoRA updates vs. separate action-transformer (MoT) experts) and skill integration (weight merging, expert-guided distillation (OPD), routing).

**Checkpoints and results:** [Google Drive folder](https://drive.google.com/drive/folders/1ZHJ52XBXTALNCE7W_SHNAMHb-n3bjYZO?usp=sharing) (see [Packages](#packages-on-google-drive)).

## Status (2026-11-02)

| Done | Result |
|---|---|
| Infrastructure, frozen 10-task manifest (LIBERO-90 indices 78, 84, 16, 56, 35, 42, 28, 3, 89, 63), baseline | `pi05_libero` 14/100 on the official init states (`docs/SETUP_REPORT.md`) |
| Aligned 256 px demonstrations, LeRobot datasets, split profiles n10 (train demo_0..9) and n25 (train demo_0..24) | `configs/datasets/processed_v1.yaml` |
| 30 per-task teachers: `teacher-n25-s500` (LoRA, 25 demos), `expert300m-{10,25}demos-s250` (action expert only, 10 or 25 demos) | LoRA teachers 209/250 vs base 37/250 on held-out demo_25..49 starts |
| OGPO port for π₀.₅ (Tasks 1-11): stochastic sampler, exact path log-probability, path KL, pathwise gradient | Tasks 5-9 frozen; Task 10 gradient validation inconclusive (`docs/TASK10_11_SUMMARY.md`) |
| On-policy distillation (OPD) pilots on t35 "open the microwave" with a LoRA student and Teacher-10D | KL 10x lower, executed-action error 3-4x lower, but 0/40 held-out success vs teacher 24/40 |
| Diagnostics of that failure | Main cause: the student rarely reaches the handle (12/40 vs base 38/40); from teacher states it finishes 19/20 at the 75% point |
| Two-phase curriculum (teacher-controlled warm-start, gate at update 100, then on-policy OPD) | Coded, smoke-tested, preregistered; **not run yet** |

Every run, path and number: `docs/OPD_EXPERIMENT_REGISTRY.md`. OGPO derivations and decisions: `tests/ogpo_port/README.md`, `docs/OGPO_AUDIT.md`.

## Setup

```bash
git clone --recursive https://github.com/raoulbouri/vla-skill-consolidation.git && cd vla-skill-consolidation
cp .env.example .env                                     # set the paths in .env
bash scripts/setup_envs.sh                               # pinned OpenPI + official LIBERO, two uv envs
scripts/py-policy -c "from openpi.shared import download; download.maybe_download('gs://openpi-assets/checkpoints/pi05_libero')"   # base checkpoint, 12.4 GB, public
scripts/py-policy -m pytest tests/unit                   # CPU unit tests (policy env)
```

- Two environments: `scripts/py-policy` (OpenPI, Python 3.11) and `scripts/py-sim` (LIBERO simulator, Python 3.8). Details, LIBERO-90 download and the test stages: `docs/SETUP.md`, `docs/ASSETS.md`.
- Processed data (aligned HDF5 + LeRobot, about 17 GB) is not uploaded; rebuild it with `bash scripts/prepare_data_overnight.sh` (set `CLVLA_SPLIT_PROFILE=n10` or `n25`).
- On a shared GPU machine, every launcher calls `scripts/gpu_guard.sh` and refuses a GPU another user is on. Set `CLVLA_GPU` in `.env`; on 24 GB GPUs also set `SERVER_MEM_FRACTION=.47`.
- Integrity: `sha256sum -c tests/ogpo_port/frozen_task5_9.sha256` must pass; the frozen OGPO modules are not to be edited.

## Packages on Google Drive

Folder: <https://drive.google.com/drive/folders/1ZHJ52XBXTALNCE7W_SHNAMHb-n3bjYZO?usp=sharing>. Each package is a plain `.tar` with a `.SHA256SUMS` file. Paths inside packages 1 and 3 are relative to the repository root, so extracting there puts every file where the scripts expect it. The base π₀.₅-LIBERO checkpoint is not included (public, see Setup).

| Package | Size | Contents |
|---|---|---|
| `vla-skill-consolidation_pkg1_opd-core_20261102.tar` | 4.7 GiB | t35 teachers used by OPD (Teacher-10D, Teacher-25D, LoRA teacher) and all OPD student adapters |
| `vla-skill-consolidation_pkg2_results_20261102.tar` | 18 MiB | Metrics, logs, reports and evaluation records of every run, no weights |
| `vla-skill-consolidation_pkg3_teacher-deltas_20261102.tar` | 34 GiB | All 30 teachers as deltas against the base |

```bash
cd vla-skill-consolidation
tar xf /path/to/vla-skill-consolidation_pkg1_opd-core_20261102.tar          # -> outputs/...
sha256sum -c /path/to/vla-skill-consolidation_pkg1_opd-core_20261102.SHA256SUMS
```

**Package 1: OPD core** (extract in the repository root)

| Path | What |
|---|---|
| `outputs/opd_t35_student10d/20261102T024213Z/teacher10d_t035/` | Teacher-10D t35 (`expert300m-10demos-s250`), the teacher of every OPD run |
| `outputs/stage1_stage2_combined/20261102T002606Z_xulab_check/teacher25d_t035/` | Teacher-25D t35 (`expert300m-25demos-s250`) |
| `outputs/opd_stage2/teacher_t035/` | LoRA teacher t35 (`teacher-n25-s500`) |
| `outputs/opd_t35_student10d/20261102T024213Z/student10d/checkpoints/adapter_{000,025,050,075,100}.npz` | Student-10D, 100-update run (rollouts from demo_0..9 starts) |
| `outputs/opd_t35_student10d/20261102T044223Z_500updates_random_starts_arm004/student10d/checkpoints/adapter_{000,...,500}.npz` | Student-10D, 500-update run (random starts, 0.04 rad arm noise): the baseline for the next experiment |

A teacher directory holds `delta.npz` (only the leaves the teacher changed or added) and `leaf_sha.json` (BF16 sha256 of every leaf of the teacher and of the base). Rebuilding refuses to continue unless every leaf matches:

```python
from pathlib import Path
import numpy as np
from clvla.config import load_experiment_config
from clvla.ogpo.pi05_sde_policy import model_of
from clvla.policy import load_official_policy
from opd_lora_student import build_student_policy, load_delta_teacher    # scripts/ on PYTHONPATH

c = load_experiment_config("configs/experiment.yaml")
base, _ = load_official_policy(c.checkpoint.openpi_config, c.checkpoint.local_dir, expected_asset_id=c.checkpoint.norm_stats_asset_id, state_dim=c.policy.state_dim, action_dim=c.policy.action_dim, action_horizon=c.policy.action_horizon)
teacher, _, _ = load_delta_teacher(model_of(base), Path("outputs/opd_t35_student10d/20261102T024213Z/teacher10d_t035"))
lora_teacher, _, _ = load_delta_teacher(model_of(base), Path("outputs/opd_stage2/teacher_t035"), paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora")
student = build_student_policy(c, np.load("outputs/opd_t35_student10d/20261102T044223Z_500updates_random_starts_arm004/student10d/checkpoints/adapter_500.npz"))
```

To serve one for evaluation: `scripts/combined_serve.py --kind teacher_delta --teacher-delta <dir>` or `--kind student --adapter <npz>` (see `scripts/combined_worker.sh`, `scripts/combined_eval.py`).

**Package 2: results** (extract anywhere, e.g. `mkdir results && tar xf ...pkg2... -C results`). Two top-level folders, `xulab/` and `aims/`, mirror the `outputs/` and `logs/` trees of the two machines the work ran on. Weight files (`*.npz`, checkpoint shards) and W&B run folders are excluded; training curves are in W&B project `rahulbouri16/IRL-final-project`. Start from `docs/OPD_EXPERIMENT_REGISTRY.md`, which names each run directory. Highlights:

| Path | What |
|---|---|
| `aims/outputs/teacher_eval/comparison_teacher-n25-s500.md` | LoRA teachers vs base, 10 tasks |
| `aims/outputs/stage1_stage2_combined/20261102T002606Z/` | Expert teachers vs base (Stage 1, partial) |
| `aims/outputs/specialists/*/` | Every teacher run's spec, history, summary |
| `xulab/outputs/opd_t35_student10d/*/` | OPD runs: `student10d/metrics.json`, `training_metrics.csv`, `comparison.json`, `side_by_side.csv` |
| `xulab/outputs/opd_t35_diag/20261102T_diag/` | Diagnostics: `preregistration.json`, `analysis.txt`, per-episode geometry, takeovers, scores |
| `xulab/outputs/ogpo_port/` | OGPO Tasks 5-11 result JSONs |
| `*/logs/archive_logs_upto_20261101.tar.gz` | Older logs |

**Package 3: all 30 teachers as deltas** (extract in the repository root) -> `outputs/teacher_deltas/<variant>/t<LIBERO-90 index>_<task>/` with `delta.npz`, `leaf_sha.json`, `train_spec.yaml`, `summary.json`. Variants: `expert300m-10demos-s250`, `expert300m-25demos-s250` (load with `load_delta_teacher(...)`), `lora-25demos-s500` (the LoRA `teacher-n25-s500` runs: pass `paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"`). Made by `scripts/export_teacher_deltas.sh` from the full checkpoints; the three t35 deltas are byte-identical to the ones in package 1.

## Next experiment: two-phase curriculum (prepared)

Teacher-10D drives the robot for updates 1-100 (successful episodes only, observations chosen by phase) while the student samples its own flow chains under the unchanged pathwise KL; the run pauses, a gate on development starts 960000-960039 decides whether to continue, and updates 101-500 run on the student's own rollouts. Preregistration: `docs/OPD_CURRICULUM_PREREGISTRATION.json`.

```bash
T=outputs/opd_t35_student10d/20261102T024213Z/teacher10d_t035              # from package 1
RUN=outputs/opd_t35_curriculum/$(date -u +%Y%m%dT%H%M%SZ)_teacher_warmstart
bash scripts/opd_curr_phase1.sh $RUN $T                                    # about 2 h: 100 updates, then the gate (gate/gate.json)
bash scripts/opd_curr_phase2.sh $RUN $T outputs/opd_t35_student10d/20261102T044223Z_500updates_random_starts_arm004/student10d/checkpoints/adapter_500.npz   # after review, about 3.5 h
```

The pathwise gradient compile needs about 120 GiB of host RAM and about 19 GB of GPU memory (a 32 GB GPU works; 61 GB of host RAM does not).

## Rules we keep

- Held-out evaluation states (demo_10..49 for n10, demo_25..49 for n25) are never used for training or checkpoint selection; reserved seeds 12000-12007, 13000-13007 and 15000 stay unused.
- New experiments get a preregistration (thresholds fixed before results) and a timestamped output directory; earlier outputs are never overwritten.
- Failures are reported as failures; reruns get new directories.

## Layout

`src/clvla/` (config, data, policy adapter, evaluation, training, `ogpo/` flow-policy likelihood and KL), `scripts/` (data, training, evaluation, OPD, diagnostics, curriculum), `tests/{unit,integration,ogpo_port}/`, `configs/` (frozen `experiment.yaml`, `task_manifest.yaml`, dataset and training specs), `docs/`, `third_party/openpi` (pinned submodule with the official LIBERO). Large assets and outputs live outside git (`.gitignore`, `.env.example`).
