# Assets

Every external asset, where it comes from, where it lives, and how it was verified. Large files never enter git:
`.gitignore` excludes `*.hdf5`, `*.ocdbt`, `outputs/`, `logs/`, and the checkpoint lives outside the repository.

Rerun all checks with `scripts/py-policy scripts/validate_assets.py --deep`. It exits 1 on any problem and writes a
JSON report to `outputs/asset_validation/`. Downloads are resumable: rerunning OpenPI's `maybe_download` or HF
`snapshot_download` fetches only missing files. The lock files make an interrupted download visible as a size
mismatch, a missing file or a leftover partial file.

## 1. Code (pinned)

| Repo | URL | Revision | Notes |
|---|---|---|---|
| OpenPI | https://github.com/Physical-Intelligence/openpi | `215abfb217dbac7d5f1273282331b9b1866c0479` (main, 2026-08-24) | git submodule `third_party/openpi`; deps from its `uv.lock` (copy: `env/openpi-uv.lock`) |
| LIBERO (official) | https://github.com/Lifelong-Robot-Learning/LIBERO | `f78abd68ee283de9f9be3c8f7e2a9ad60246e95c` | OpenPI's own submodule `third_party/openpi/third_party/libero`; MIT license |
| OpenPI `third_party/aloha` | https://github.com/Physical-Intelligence/aloha | `d1dc83afd89ded4379851257fe5d85632d31d5ec` | initialized by OpenPI, unused here |

Upstream LIBERO `master` (`8f1084e`, 2025-03-15) is 3 commits ahead of the pinned LIBERO. Those commits change only
`README.md`, `benchmark_scripts/download_libero_datasets.py`, `libero/libero/utils/download_utils.py` (adding the
Hugging Face download source) and `requirements.txt`. **No BDDL, init-state or asset file differs**, so task
definitions are identical.

## 2. Checkpoint: official π₀.₅-LIBERO

| Field | Value |
|---|---|
| Source | `gs://openpi-assets/checkpoints/pi05_libero` (public, anonymous GCS via gcsfs) |
| OpenPI config | `pi05_libero` (`Pi0Config(pi05=True, action_horizon=10)`, trained 30k steps from `pi05_base` on `physical-intelligence/libero`) |
| Download | `openpi.shared.download.maybe_download` (OpenPI's supported mechanism), 2026-10-22, 399 s |
| Local path | `$OPENPI_DATA_HOME/openpi-assets/checkpoints/pi05_libero` = `/mnt/data/users/bbouri/cache/openpi/openpi-assets/checkpoints/pi05_libero` |
| Size | 16 files, 12.44 GB (JAX/orbax OCDBT `params/` + `assets/`) |
| Lock | `configs/checkpoint.lock.json`: size + sha256 of every file |
| Fingerprint | `cd22f6becde68e647885fb9beb54a6a8fc296b117d87b6de03dea9d53dd07fee` (sha256 over the lock; recorded in every run's metadata) |
| Normalization stats | bundled `assets/physical-intelligence/libero/norm_stats.json` (sha256 `b3a44bb2810436fb…`); state 8-D, actions 7-D, mean/std/q01/q99; pi05 uses quantile normalization. Used as-is, never recomputed |
| Tokenizer | `gs://big_vision/paligemma_tokenizer.model`, fetched by OpenPI on first use into `$OPENPI_DATA_HOME/big_vision` (4.1 MB) |

Validation (2026-10-23):

- `validate_assets.py --deep`: all 16 files match the lock (sha256), the required files are present, the norm stats
  have the right dimensions and are finite with q01 ≤ q99, and there are no partial files.
- `verify_checkpoint_source.py`: all 16 local files match the **GCS object size and MD5**. This is provenance against
  the source, not just self-consistency.
- `tests/integration/test_checkpoint.py`: loads through OpenPI's `create_trained_policy` behind a compatibility gate.
  The policy's `Normalize` transform holds exactly the bundled q01/q99. Inference returns finite `(10, 7)` action
  chunks in about 63 ms on an RTX 5090.

Not used and not substitutable: `pi05_droid`, `pi0_*`, `pi05_base`, any self-trained weights (`clvla.config`
refuses any other checkpoint URI or config).

## 3. Demonstrations: official LIBERO-90 HDF5

| Field | Value |
|---|---|
| Canonical source | LIBERO's own download script. Pinned LIBERO code points to the UT Box `libero_100` zip (LIBERO-90 + LIBERO-10). Upstream LIBERO `master` adds `--use-huggingface`, which downloads https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets (published by LIBERO co-author Yifeng Zhu) |
| Revision | `f13aa24a3da8c43c7225569f28c562979fa0e35a` (= HF `main` on 2026-10-22) |
| License / access | LIBERO README: datasets CC BY 4.0 (HF card tag: apache-2.0); public, not gated |
| Local path | `$LIBERO_DATA_ROOT/libero_90/` = `/mnt/data/users/bbouri/cl/data/libero/libero_90/`: 90 files, 66.66 GB. Reused read-only from the earlier CL-benchmark download (2026-10-04, `snapshot_download`, same repo/revision) instead of downloading a second copy |
| Lock | `configs/dataset.lock.json`: size + LFS sha256 of all 90 files, taken from the **HF API** at the pinned revision (independent of the local copy) |
| Format | robomimic-style HDF5: `data/demo_<i>/{actions (T,7) f64, obs/agentview_rgb (T,128,128,3) u8, obs/eye_in_hand_rgb, obs/ee_states (T,6), obs/gripper_states (T,2), obs/ee_pos, obs/ee_ori, obs/joint_states (T,7), states (T,77) MuJoCo, robot_states (T,9), rewards, dones}`; `data.attrs.problem_info.language_instruction`; images in OpenGL orientation (`macros_image_convention: opengl`) |
| Coverage | every LIBERO-90 task has exactly 50 demos (lengths 58–373 steps) whose stored instruction equals the benchmark instruction, plus 50 official init states |

Not the same thing: `physical-intelligence/libero` on HF (OpenPI's fine-tuning data) contains only LIBERO-Spatial,
Object, Goal and 10, in OpenVLA's 256 px no-op-filtered re-renders. It has **no LIBERO-90 tasks**, so it cannot
supply demonstrations for this study. The checkpoint is not a source of demonstrations either.

Validation (2026-10-23):

- All 90 files: local size and the HF-cache-recorded sha256 match the HF API lock exactly.
- The 10 manifest tasks: full sha256 matches the HF LFS sha256 (7.07 GB). Every demo also passes full schema
  validation: keys, shapes, dtypes, equal lengths, finite values, |action| ≤ 1, exactly 50 demos, 128×128 images,
  and the stored instruction equals the manifest instruction.
- Data and simulator agree: restoring demo MuJoCo `states` in the official env reproduces the stored 8-D state
  (residual ≤ 1.7e-3) and agentview image (mean error 0.68/255), with a one-step offset (see
  `docs/EXPERIMENT_PROTOCOL.md` §6).

### Selected-task demo files (the only ones the study uses)

| Pos | LIBERO-90 idx | File | GB | sha256 |
|---|---|---|---|---|
| 0 | 78 | `STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy_demo.hdf5` | 0.75 | `5cffae59b652908ca8e430af8d27ad5d20c31e6ea25a0b563fd7a85781a4c162` |
| 1 | 84 | `STUDY_SCENE3_pick_up_the_red_mug_and_place_it_to_the_right_of_the_caddy_demo.hdf5` | 0.59 | `ec62bb4cf08fa8c6eea7190d59a1c59dae125f9055c70942f114a36dbe81615b` |
| 2 | 16 | `KITCHEN_SCENE2_stack_the_black_bowl_at_the_front_on_the_black_bowl_in_the_middle_demo.hdf5` | 0.64 | `f4d4ae39d812c212e075bfa1f82ef37a56942bb8c981544468afb0c8ca8c47a4` |
| 3 | 56 | `LIVING_ROOM_SCENE3_pick_up_the_butter_and_put_it_in_the_tray_demo.hdf5` | 0.51 | `96c1728b1e138a09f8799b981acdb08de41e8ebb6244294c27cf142547eaef1f` |
| 4 | 35 | `KITCHEN_SCENE7_open_the_microwave_demo.hdf5` | 0.74 | `ba7e2fa1622f94adda3d93a44e5bc021391f5f0a31c3c1418cdc980a97207d21` |
| 5 | 42 | `KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf_demo.hdf5` | 0.86 | `1ee3bb8ab3ed9deaa27cb410b0b9a29874e8687e940bb4a0f4557586cf64bddf` |
| 6 | 28 | `KITCHEN_SCENE5_close_the_top_drawer_of_the_cabinet_demo.hdf5` | 0.38 | `504b27586d2de6b663d1ad1cfd9519bee8761b7fdb568e1444e28158ea4667fe` |
| 7 | 3 | `KITCHEN_SCENE10_put_the_butter_at_the_back_in_the_top_drawer_of_the_cabinet_and_close_it_demo.hdf5` | 0.94 | `b938c9ad909d6e9448eb7dd75be50f15b8790252b45bd8460b80d03da0b5a8f8` |
| 8 | 89 | `STUDY_SCENE4_pick_up_the_book_on_the_right_and_place_it_under_the_cabinet_shelf_demo.hdf5` | 0.60 | `7980c84e18335a4875878b44d0e9db9908908799329018563f199c61d569409c` |
| 9 | 63 | `LIVING_ROOM_SCENE4_stack_the_left_bowl_on_the_right_bowl_and_place_them_in_the_tray_demo.hdf5` | 1.07 | `6e5dd6cb435e94cd90886e2ae3a5f7145d3034710d7c1605bad1c97d56acb2b3` |

## 4. Simulator assets and task definitions

Shipped inside the pinned LIBERO checkout (`third_party/openpi/third_party/libero/libero/libero/`): `bddl_files/`
(task definitions), `init_files/` (50 init states per task, `*.pruned_init`), and `assets/` (meshes, textures,
scenes). `configs/libero90_tasks.json` is a snapshot of every LIBERO-90/10/Goal/Spatial/Object task name,
instruction and file, generated from this checkout by `scripts/build_task_manifest.py`. The unit tests check the
manifest against it offline.

## 5. Caches and outputs

| What | Path |
|---|---|
| OpenPI downloads | `/mnt/data/users/bbouri/cache/openpi` (`OPENPI_DATA_HOME`) |
| uv packages / Pythons | `/mnt/data/users/bbouri/cache/uv`, `/mnt/data/users/bbouri/cache/uv-python` |
| Hugging Face | `/mnt/data/users/bbouri/cache/hf` (`HF_HOME`) |
| Run artifacts | `outputs/{baseline,smoke,finetune,continual,asset_validation}/` (git-ignored) |
| Logs | `logs/` (git-ignored) |
