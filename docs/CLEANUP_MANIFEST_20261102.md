# Cleanup, 2026-11-02

Approved by the user: groups A (duplicate copies) and B (obsolete runs). Groups C (teacher `train_state/`, about 63 GB on aims) and D (Task 10B/10C gradient dumps, 43 GB on xulab) were left untouched. Every deleted file is listed with its size and sha256 prefix in `docs/CLEANUP_MANIFEST_20261102_files.tsv.xulab` (101 files, 1.94 GiB) and `docs/CLEANUP_MANIFEST_20261102_files.tsv.aims` (67 files, 14.07 GiB). No checkpoint used for evaluation was touched; all 30 links under aims `outputs/checkpoints/` still resolve.

## Deleted

| Group | Machine | Path | Why safe |
|---|---|---|---|
| A | aims | `tmp/teacher_t035`, `tmp/teacher10d_t035`, `tmp/teacher25d_t035` | Every file sha256-identical to the xulab copies (`outputs/opd_stage2/teacher_t035`, `outputs/opd_t35_student10d/20261102T024213Z/teacher10d_t035`, `outputs/stage1_stage2_combined/20261102T002606Z_xulab_check/teacher25d_t035`); the one file without a xulab copy, `teacher_t035/train_spec.yaml`, was copied there first. Re-exportable from the aims checkpoints with `scripts/opd_export_teacher_delta.py` |
| A | xulab | `outputs/opd_stage3/20261101T235006Z_t035/temporary_update/master_after_one_step.npz` | Temporary parameter copy from the one-step Stage 3 check; results are in `report.json` |
| B | aims | `outputs/finetune/20261029T17{1504,1650}Z_memprobe-{B,C}*/checkpoints/` | Step-6 memory-probe checkpoints; run metadata kept |
| B | xulab | `outputs/opd_t35_student10d/20261102T041612Z_500updates_random_starts/student10d/{checkpoints,batches}` | Aborted object-only run (64 rollouts, never used); logs, metrics and `ABORTED.txt` kept |
| B | xulab | `outputs/opd_t35_diag/smoke/` | Smoke test of the diagnostic scripts; the full run is `outputs/opd_t35_diag/20261102T_diag` |

## Moved or archived (nothing lost)

- Logs dated up to 2026-11-01 on each machine: `logs/archive_logs_upto_20261101.tar.gz`, verified with `gzip -t`, `tar d` and a file count (xulab 94, aims 65), file list in `logs/archive_logs_upto_20261101.contents.txt`. Logs cited by name in the ledger are inside the archive.
- aims failed and pilot training directories moved to `outputs/specialists/_failed_and_pilot_runs/` (7 directories, metadata only).
- xulab probe scripts `probe_model.py`, `probe_door.py`, `probe_door2.py` moved into `outputs/opd_t35_diag/20261102T_diag/probes/`.

## Code sync

xulab is the source of truth. The aims versions of `combined_eval.py`, `combined_serve.py`, `combined_worker.sh`, `opd_lora_collect.py`, `opd_lora_student.py`, `opd_lora_train.py` were archived in aims `outputs/_code_archive/scripts_aims_before_sync_20261102.tar.gz` and replaced. Their only aims-specific content: the n25 split as default, server memory fraction .47 (now `SERVER_MEM_FRACTION=.47` on aims, default .3), and a `jax.clear_caches()` re-jit after the first update's checks. Both machines now hold the same 171 files under `src scripts tests configs docs env.sh` (sha256 compared by path).
