# Cleanup, 2026-11-03

Approved by the user: B1 failed and smoke run weights, B2 packaging copies already uploaded to Hugging Face, and removal of finished one-off scripts (git commit `2b06f71`, recoverable from history). Teacher optimizer state (`train_state/`, about 63 GB on aims) and the Task 10B/10C gradient dumps (43 GB on xulab) were not selected and stay. Every deleted file is listed with its size and sha256 prefix (or `hardlink` when another copy remains) in `docs/CLEANUP_MANIFEST_20261103_files.tsv.xulab` (163 files, 5.35 GiB freed) and `docs/CLEANUP_MANIFEST_20261103_files.tsv.aims` (370 files, 34.00 GiB freed).

| Group | Machine | Path | Why safe |
|---|---|---|---|
| B1 | xulab | `outputs/opd_t35_curriculum/20261103T021041Z_teacher_warmstart/**/*.npz` | Failed launch before any update (`FAILED.txt`); its logs stay |
| B1 | xulab | `outputs/opd_t35_curriculum/smoke_trainer/` | Smoke test of the curriculum code; results recorded in the registry |
| B2 | xulab | `outputs/_packages/` (package 1 and 2 tars, git bundle, file lists, test script) | Uploaded to `huggingface.co/raoulb091602/vla-skill-consolidation` and hash-verified (all 145 weight files and the results tar) |
| B2 | xulab, aims | `outputs/_hf_stage/` | Hard links to kept originals plus the upload tool's cache; the script refused to run if any staged file was a sole copy |
| B2 | xulab | `outputs/_git_stage_pre_curriculum/` | Pre-curriculum script versions, committed in `3db6f35` and `9386bdc` |
| B2 | aims | `outputs/_packages/` (package 3 tar, results part, git bundle, file lists) | Package 3 contents remain in `outputs/teacher_deltas/` (30 teachers) and on Hugging Face |

Kept: xulab `.env` backup moved to `outputs/_env_backup/dotenv_backup_20261103`. The curriculum run `outputs/opd_t35_curriculum/20261103T021211Z_teacher_warmstart` is untouched, including its update-100 training state.
