# OPD experiment registry (teachers, OPD pilots, diagnostics)

Written 2026-11-02. Covers every training run and experiment on `aims` (`/usr1/home/bbouri/IRL-project`) and `xulab` (`/mnt/data/users/bbouri/IRL-project`) from 2026-10-29 on. Earlier work: `docs/SETUP_REPORT.md` (baseline), `docs/TASK10_11_SUMMARY.md` and `tests/ogpo_port/README.md` (OGPO port, Tasks 5-11). aims and xulab cannot reach each other; files move through the Mac only when small.

## 1. Checkpoints to keep for downstream evaluation

| What | Machine | Path | Notes |
|---|---|---|---|
| pi05_libero base | both | `$OPENPI cache/openpi` (see `docs/ASSETS.md`) | Unadapted pi0.5 |
| LoRA teachers `teacher-n25-s500` (10 tasks) | aims | `outputs/specialists/20261029T*_teacher-n25-s500-t*/checkpoints/.../500` | Indexed as `outputs/checkpoints/lora-25demos-s500` |
| Expert teachers `expert300m-10demos-s250` (10 tasks) | aims | `outputs/specialists/20261030T*_expert300m-10demos-s250-t*/.../250` | Indexed as `outputs/checkpoints/expert300m-10demos-s250` |
| Expert teachers `expert300m-25demos-s250` (10 tasks) | aims | `outputs/specialists/20261030T*_expert300m-25demos-s250-t*/.../250` | Indexed as `outputs/checkpoints/expert300m-25demos-s250` |
| Teacher-10D t35 delta (OPD teacher) | xulab | `outputs/opd_t35_student10d/20261102T024213Z/teacher10d_t035` | `delta.npz` sha256 prefix `8cfc9c41f741cea3`; rebuilt teacher verified leaf by leaf |
| Teacher-25D t35 delta | xulab | `outputs/stage1_stage2_combined/20261102T002606Z_xulab_check/teacher25d_t035` | prefix `c95eed3efe573eea` |
| LoRA teacher t35 delta (Stage 2) | xulab | `outputs/opd_stage2/teacher_t035` | prefix `d0d689ac669c94d8` |
| Student-10D, 100 updates | xulab | `outputs/opd_t35_student10d/20261102T024213Z/student10d/checkpoints/adapter_{000..100}.npz` | |
| Student-10D, 500 updates (baseline for the next run) | xulab | `outputs/opd_t35_student10d/20261102T044223Z_500updates_random_starts_arm004/student10d/checkpoints/adapter_{000..500}.npz` | 9 checkpoints |

Evaluation loads only `params/` and `assets/` of an OpenPI checkpoint (`policy_config.create_trained_policy`); `train_state/` holds optimizer moments, used only to resume training.

## 2. Teacher training (aims)

| Run | Spec | Result |
|---|---|---|
| `teacher-n25` 2000 steps (t078, t084) | LoRA, 25 demos | Overfit after step 500; checkpoints deleted 2026-10-29 at the user's request (`CHECKPOINT_DELETED.txt`) |
| `teacher-n25-s500` (10 tasks) | LoRA, 25 demos, 500 steps | Held-out demo_25..49: 209/250 vs base 37/250; t35 24/25 (`outputs/teacher_eval/comparison_teacher-n25-s500.md`) |
| `expert-pilot-n{10,25}-t042` | Expert mode pilot | Stopped; no final checkpoint |
| `expert300m-{10,25}demos-s250` (20 runs) | Expert-only (Gemma 2B + SigLIP frozen), 250 steps | 4 `_FAILED` attempts retried successfully; held-out results in section 3 |
| `memprobe-{A,B,C}` | 6-step memory probes | Probe only |

## 3. OPD sequence on t35 "open the microwave"

| Date | Experiment | Machine, path | Result |
|---|---|---|---|
| 11-01 | Stage 2: student SDE rollout smoke | xulab `outputs/opd_stage2/20261101T230405Z_t035_demo0` | PASS; replay bitwise in the same programme |
| 11-01 | Stage 3: one FP32-master pathwise update | xulab `outputs/opd_stage3/20261101T235006Z_t035` | Paired KL 661.19 to 650.76, CI [-13.46, -7.40]; limited GO |
| 11-02 | Combined Stage 1 (teacher eval, 10 tasks) | aims `outputs/stage1_stage2_combined/20261102T002606Z` | Stopped by user at 1058/1500. t35 on demo_25..49: 10D 16 ODE / 13 SDE, 25D 11 / 11 of 24, base 1/25. NO-GO for scaling |
| 11-02 | Combined Stage 2 on aims | same, `stage2/` | Never reached a gradient: aims RAM (61 GB) |
| 11-02 | One-batch LoRA check (Teacher-25D) | xulab `.../20261102T002606Z_xulab_check` | PASS: KL 545.30 to 517.03, CI [-37.93, -18.61]; host peak 110.7 GiB |
| 11-02 | Student-10D, 100 updates, rollouts from demo_0..9 starts | xulab `outputs/opd_t35_student10d/20261102T024213Z` | Held-out demo_10..49: student 0/40 ODE, 0/40 SDE; Teacher-10D 24/40, 20/40; base 1/40, 0/40. Diagnostic KL 440 to 99 |
| 11-02 | 500 updates, random object starts only | xulab `..._20261102T041612Z_500updates_random_starts` | ABORTED at 64 rollouts (robot arm start fixed); not used |
| 11-02 | 500 updates, random starts + 0.04 rad arm noise | xulab `..._20261102T044223Z_500updates_random_starts_arm004` | Completed, no stop rule. Student500 0/40 ODE and SDE; Teacher-10D 24/40, 20/40; base 0/40; teacher minus student +0.60, CI [0.45, 0.75]. KL per update 220 (1-100) to 8.6 (401-500); 1/500 training rollouts succeeded; 5 gradient spikes > 5000 (max 23,816) |
| 11-02 | Diagnostics Exp 1-3 (no training), starts 950000-950039 | xulab `outputs/opd_t35_diag/20261102T_diag` (`preregistration.json`, `analysis.txt`) | See section 4 |

## 4. Diagnostic findings (Exp 1-3)

- Handle reach (grip site within 0.12 m): teacher 34/40, base 38/40, Student100 14/40, Student500 12/40; door moved: 24, 10, 1, 3; success 20, 4, 0, 0.
- Student paths leave the teacher's (> 5 cm) at a median executed step 19; teacher reaches the handle at step 31.
- Takeover from teacher states at 25/50/75% of successful episodes: Student500 6/20, 10/20, 19/20; teacher resume 15/20, 19/20, 20/20.
- Executed-action xyz error on teacher states, base function vs Student500: approach 0.251 to 0.061, near handle 0.448 to 0.129, opening 0.351 to 0.110.
- KL on teacher states, checkpoint 0 to 500: full 781 to 80, executed 383 to 36, padded 13 to 1.8; on the student's own states 25 at checkpoint 500.
- Verdict (criteria fixed beforehand): mainly state coverage, plus weaker near-handle and early-opening skill; the KL drop is in executed coordinates, not padding.

## 5. Code location

Synced on 2026-11-02 with xulab as the source of truth; both machines hold identical `src scripts tests configs docs env.sh`. On aims set `SERVER_MEM_FRACTION=.47` (24 GB GPUs). Cleanup record: `docs/CLEANUP_MANIFEST_20261102.md`.

| Script | Experiment |
|---|---|
| `combined_prep.py`, `combined_serve.py`, `combined_eval.py`, `combined_worker.sh`, `combined_analyze_stage1.py` | Combined Stage 1 teacher evaluation; serve and eval reused by later runs |
| `opd_stage2*.{py,sh}`, `opd_stage3.sh` | Stage 2 smoke, Stage 3 one update |
| `opd_export_teacher_delta.py` | Export a teacher's changed leaves for transfer to xulab |
| `opd_lora_student.py`, `opd_lora_train.py`, `opd_lora_collect.py` | LoRA student and OPD trainer |
| `opd_t35_pilot.sh`, `opd_t35_500.sh`, `opd_t35_check.py`, `opd_t35_compare.py` | 100- and 500-update t35 runs and their evaluation |
| `opd_diag_*.{py,sh}` | Diagnostics Exp 1-3 (now take `--start-seed0`, `--noise-seed0`; scorer takes `--ckpts`, `--teacher-set`, `--student-set`, `--compare-ckpt`, `--out-name`) |
| `opd_curr_phase1.sh`, `opd_curr_phase2.sh`, `opd_curr_gate.py`, `opd_curr_compare.py` | Two-phase curriculum (section 7) |

## 6. Handoff (2026-11-02)

Code: <https://github.com/raoulbouri/vla-skill-consolidation>. Checkpoints and results: Google Drive folder linked from `README.md`, three packages built on xulab (`outputs/_packages/`, packages 1-2) and aims (`outputs/_packages/`, package 3); each has a `.SHA256SUMS` and a `.filelist`. Package 3 comes from `scripts/export_teacher_deltas.sh` (aims `outputs/teacher_deltas/`). `load_delta_teacher` now takes `paligemma_variant` and `action_expert_variant`, so LoRA-teacher deltas load too (checked on CPU: 71 leaves verified for the t35 LoRA teacher, 51 for Teacher-10D).

## 7. Two-phase curriculum (prepared, not started)

Preregistration: xulab `outputs/opd_t35_curriculum/preregistration.json`. New trainer options in `opd_lora_train.py`, all off by default so earlier runs reproduce: `--teacher-control-updates`, `--teacher-success-only`, `--max-teacher-tries`, `--stop-after`, `--resume`. The collector now attaches a geometry snapshot to each request and reports handle reach, contact and door onset; `combined_eval.py --geometry` adds the same to held-out episodes.
Smoke tests (xulab `outputs/opd_t35_curriculum/smoke_trainer`): teacher-driven update with one re-roll, pause, resume and a student-driven update all passed (fresh-noise KL 813 to 778, CI [-47, -24]; host peak 118 GiB; GPU peak 13 GiB); gate rollouts, student takeover, scorer and gate script passed; `combined_eval.py --geometry` not yet run on GPU.
Launch: `scripts/opd_curr_phase1.sh outputs/opd_t35_curriculum/<timestamp> outputs/opd_t35_student10d/20261102T024213Z/teacher10d_t035`, then after review `scripts/opd_curr_phase2.sh <same dir> <teacher delta> outputs/opd_t35_student10d/20261102T044223Z_500updates_random_starts_arm004/student10d/checkpoints/adapter_500.npz`.
