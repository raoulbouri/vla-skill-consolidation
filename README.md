# vla-skill-consolidation

**How should vision-language-action policies continually learn new skills?** Course project, 16-831 Introduction to Robot Learning, Fall 2026, CMU (Sreyas Venkataraman, Vinit Awale, Saumya Shah, Raoul Bouri).

We study sequential imitation learning from the open-source **π₀.₅-LIBERO** checkpoint (OpenPI `pi05_libero`) on a frozen sequence of ten held-out **LIBERO-90** tasks, along two axes: expert capacity (LoRA updates vs. separate action-transformer (MoT) experts) and skill integration (weight merging, expert-guided distillation (OPD), routing).

**Checkpoints and results:** Hugging Face [`raoulb091602/vla-skill-consolidation`](https://huggingface.co/raoulb091602/vla-skill-consolidation) (see [Weights and results](#weights-and-results-on-hugging-face)).

## Status (2026-11-03)

| Done | Result |
|---|---|
| Infrastructure, frozen 10-task manifest (LIBERO-90 indices 78, 84, 16, 56, 35, 42, 28, 3, 89, 63), baseline | `pi05_libero` 14/100 on the official init states (`docs/SETUP_REPORT.md`) |
| Aligned 256 px demonstrations, LeRobot datasets, split profiles n10 (train demo_0..9) and n25 (train demo_0..24) | `configs/datasets/processed_v1.yaml` |
| 30 per-task teachers: `teacher-n25-s500` (LoRA, 25 demos), `expert300m-{10,25}demos-s250` (action expert only, 10 or 25 demos) | LoRA teachers 209/250 vs base 37/250 on held-out demo_25..49 starts |
| OGPO port for π₀.₅ (Tasks 1-11): stochastic sampler, exact path log-probability, path KL, pathwise gradient | Tasks 5-9 frozen; Task 10 gradient validation inconclusive (`docs/TASK10_11_SUMMARY.md`) |
| On-policy distillation (OPD) pilots on t35 "open the microwave" with a LoRA student and Teacher-10D | KL 10x lower, executed-action error 3-4x lower, but 0/40 held-out success vs teacher 24/40 |
| Diagnostics of that failure | Main cause: the student rarely reaches the handle (12/40 vs base 38/40); from teacher states it finishes 19/20 at the 75% point |
| Two-phase curriculum without replay (teacher-controlled warm-start, then on-policy OPD) | Warm-start worked (update-100 student moves the door in 22/40 development episodes); student-driven Phase 2 lost it (no contact after about update 150), stopped at update 247 |
| **Default distillation: warm-start, then student rollouts with a per-phase replay buffer** (`scripts/opd_distill.sh`) | **Student matches the teacher on held-out demo_10..49: 26/40 vs 24/40 ODE, 22/40 vs 20/40 SDE (teacher minus student -0.05, CI [-0.20, 0.10]); base 1/40** |

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

## Weights and results on Hugging Face

Model repo: <https://huggingface.co/raoulb091602/vla-skill-consolidation> (private for now; ask Raoul for access). Weights are stored as plain files at the paths the scripts expect, so download them into the repository root; the base π₀.₅-LIBERO checkpoint is not included (public, see Setup). `outputs/`, `checksums/` and `results/` are git-ignored.

| Part | Size | Repo path |
|---|---|---|
| 1. OPD core: t35 teachers used by OPD (Teacher-10D, Teacher-25D, LoRA teacher) and all OPD student adapters | 4.7 GiB | `outputs/opd_*`, `outputs/stage1_stage2_combined/...` (table below) |
| 2. Results: metrics, logs, reports and evaluation records of every run, no weights | 18 MiB | `results/vla-skill-consolidation_pkg2_results_20261102.tar` |
| 3. All 30 teachers as deltas against the base | 34 GiB | `outputs/teacher_deltas/` |
| Checksums for 1, 3 and the distilled student (paths relative to the repository root) | | `checksums/` (`..._opd-distill-t35_20261103.SHA256SUMS` for the distilled run) |

```bash
cd vla-skill-consolidation                                   # repository root
huggingface-cli login                                         # once, with a token that can read the repo
R=raoulb091602/vla-skill-consolidation
huggingface-cli download $R --local-dir . --include "outputs/opd_*" "outputs/stage1_stage2_combined/*" "checksums/*"      # part 1
huggingface-cli download $R --local-dir . --include "outputs/teacher_deltas/expert300m-10demos-s250/*"                    # part 3, one variant (or "outputs/teacher_deltas/*")
huggingface-cli download $R --local-dir . --include "results/*"                                                          # part 2
sha256sum -c checksums/vla-skill-consolidation_pkg1_opd-core_20261102.SHA256SUMS
sha256sum -c --ignore-missing checksums/vla-skill-consolidation_pkg3_teacher-deltas_20261102.SHA256SUMS
```

**Part 1: OPD core**

| Path | What |
|---|---|
| `outputs/opd_t35_student10d/20261102T024213Z/teacher10d_t035/` | Teacher-10D t35 (`expert300m-10demos-s250`), the teacher of every OPD run |
| `outputs/stage1_stage2_combined/20261102T002606Z_xulab_check/teacher25d_t035/` | Teacher-25D t35 (`expert300m-25demos-s250`) |
| `outputs/opd_stage2/teacher_t035/` | LoRA teacher t35 (`teacher-n25-s500`) |
| `outputs/opd_t35_student10d/20261102T024213Z/student10d/checkpoints/adapter_{000,025,050,075,100}.npz` | Student-10D, 100-update run (rollouts from demo_0..9 starts) |
| `outputs/opd_t35_student10d/20261102T044223Z_500updates_random_starts_arm004/student10d/checkpoints/adapter_{000,...,500}.npz` | Student-10D, 500-update run (random starts, 0.04 rad arm noise), 0/40 held-out |
| `outputs/opd_t35_curriculum/20261103T103057Z_phase2_replay_from_u100/student/checkpoints/adapter_{000,025,050,075,100,200,300,400}.npz` | **Distilled student** (default recipe): `adapter_400.npz` is the one that matched the teacher. The run's metrics, logs, `comparison.json`, `side_by_side.csv` and held-out episode records are uploaded at the same path |

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

**Part 2: results** (`cd results && tar xf vla-skill-consolidation_pkg2_results_20261102.tar`). Two top-level folders, `xulab/` and `aims/`, mirror the `outputs/` and `logs/` trees of the two machines the work ran on. Weight files (`*.npz`, checkpoint shards) and W&B run folders are excluded; training curves are in W&B project `rahulbouri16/IRL-final-project`. Start from `docs/OPD_EXPERIMENT_REGISTRY.md`, which names each run directory. Highlights:

| Path | What |
|---|---|
| `aims/outputs/teacher_eval/comparison_teacher-n25-s500.md` | LoRA teachers vs base, 10 tasks |
| `aims/outputs/stage1_stage2_combined/20261102T002606Z/` | Expert teachers vs base (Stage 1, partial) |
| `aims/outputs/specialists/*/` | Every teacher run's spec, history, summary |
| `xulab/outputs/opd_t35_student10d/*/` | OPD runs: `student10d/metrics.json`, `training_metrics.csv`, `comparison.json`, `side_by_side.csv` |
| `xulab/outputs/opd_t35_diag/20261102T_diag/` | Diagnostics: `preregistration.json`, `analysis.txt`, per-episode geometry, takeovers, scores |
| `xulab/outputs/ogpo_port/` | OGPO Tasks 5-11 result JSONs |
| `*/logs/archive_logs_upto_20261101.tar.gz` | Older logs |

**Part 3: all 30 teachers as deltas**: `outputs/teacher_deltas/<variant>/t<LIBERO-90 index>_<task>/` with `delta.npz`, `leaf_sha.json`, `train_spec.yaml`, `summary.json`. Variants: `expert300m-10demos-s250`, `expert300m-25demos-s250` (load with `load_delta_teacher(...)`), `lora-25demos-s500` (the LoRA `teacher-n25-s500` runs: pass `paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"`). Made by `scripts/export_teacher_deltas.sh` from the full checkpoints; the three t35 deltas are byte-identical to the ones in part 1.

## Default experiment: teacher-to-student distillation (see `docs/OPD_EXPERIMENT_REGISTRY.md` section 8)

One LoRA student learns t35 "open the microwave" from Teacher-10D under the pathwise path KL (the student samples its own flow chains; the teacher's velocities are the target):

1. **Teacher warm-start, updates 1-100.** The teacher drives the robot from random starts (0.04 rad arm noise), successful episodes only; 8 observations per episode chosen by phase (approach, pre-contact, contact, early opening). The full training state is saved at update 100.
2. **Student rollouts with replay, updates 101-400.** The student drives: 4 episodes per update, 8 observations from each, plus 8 observations drawn from a per-phase replay buffer (2 approach, 2 pre-contact, 2 contact, 2 opening). The buffer holds observations only (labels are recomputed on the current student every time): up to 1000 per phase, seeded with the warm-start's teacher-driven observations (at most half, never evicted), the rest filled by reservoir sampling from every decision point of the student's episodes. A behaviour stop ends training if robot-door contact falls below 5% of the last 100 student episodes.
3. **Evaluation.** Teacher, final student and base on the 40 held-out states demo_10..49 (ODE and SDE), then `comparison.json` and `side_by_side.csv`.

```bash
T=outputs/opd_t35_student10d/20261102T024213Z/teacher10d_t035              # from part 1
RUN=outputs/opd_t35_distill/$(date -u +%Y%m%dT%H%M%SZ)
GPU=1 bash scripts/opd_distill.sh $RUN $T                                  # about 7 h; STAGE=train or STAGE=eval runs one part
```

Result of the run that defined this recipe (Phase 2 continued the warm-start run's saved update-100 state; the run was stopped at update 459 after training success had levelled off at about 70%, and the update-400 checkpoint was evaluated): student 26/40 ODE, 22/40 SDE; Teacher-10D 24/40, 20/40; base 1/40, 0/40; door moved 32/40 vs teacher 26/40. Without the replay buffer (one episode per update) the student stopped touching the door after about 50 student-driven updates. Preregistration: `docs/OPD_REPLAY_PREREGISTRATION.json`.

The pathwise gradient compile needs about 120 GiB of host RAM and about 21 GB of GPU memory with the simulator (a 32 GB GPU works; 61 GB of host RAM does not); evaluation with three policy servers needs about 29 GB.

## Rules we keep

- Held-out evaluation states (demo_10..49 for n10, demo_25..49 for n25) are never used for training or checkpoint selection; reserved seeds 12000-12007, 13000-13007 and 15000 stay unused.
- New experiments get a preregistration (thresholds fixed before results) and a timestamped output directory; earlier outputs are never overwritten.
- Failures are reported as failures; reruns get new directories.

## Layout

`src/clvla/` (config, data, policy adapter, evaluation, training, `ogpo/` flow-policy likelihood and KL), `scripts/` (data, training, evaluation, OPD, diagnostics, curriculum), `tests/{unit,integration,ogpo_port}/`, `configs/` (frozen `experiment.yaml`, `task_manifest.yaml`, dataset and training specs), `docs/`, `third_party/openpi` (pinned submodule with the official LIBERO). Large assets and outputs live outside git (`.gitignore`, `.env.example`).
