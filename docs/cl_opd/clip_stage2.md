> Archived copy of `outputs/cl_opd/analysis/clip_stage2/REPORT.md` (2026-11-21). Paths are relative to the repository root. Where it says "uncommitted", that was before commit 0f83c96; the code is now in git.

# Stage 2: per-trajectory gradient clipping, implementation + offline validation (no training)

Files (uncommitted, xulab): src/clvla/cl_opd/traj_clip.py (rule), scripts/opd_lora_train.py (`# traj-clip` blocks,
`--traj-clip-k`, default 0 = off), scripts/cl_opd_run.py + src/clvla/cl_opd/config.py (`per_task.traj_clip_k`),
tests/unit/test_traj_clip.py, scripts/cl_opd_clip_validate.py. Originals: pre_stage2/. Raw: results.jsonl,
sensitivity.{json,md}, validate.log; helpers summarize_results.py, distributions.py.

Rule: s_i = min(1, k * median_{class(i)} |g_i| / |g_i|), classes fresh / replay / old of the current batch; update
= sum_i a_i s_i g_i with the unchanged loss weights a_i; then the unchanged optax chain (global clip 1.0, AdamW).
Two passes: pass 1 = the existing loop + per-trajectory norms/fingerprints; pass 2 recomputes only clipped trajectories
and adds a_i (s_i - 1) g_i (exactly the clipped update; equal to a full second pass up to FP32 rounding, measured below).

## Results (18 stored batches: 6 exact-weight anchors + 12 checkpoint-weight batches with the largest simulated effect)
1. CPU tests: tests/unit/test_traj_clip.py 11/11 (incl. trainer pass-2/finish on CPU JAX); full unit suite 216 passed,
   1 skipped (the trainer test, when another test imported JAX first).
2. Clipping disabled: grad_norm / grad_expert / grad_heads equal the logged values at all 6 anchors (rel. diff 0.0);
   per-trajectory norms equal Stage 1's to <= 7e-8. Trainer without the marked blocks is byte-identical (sha 2dde6ac).
3. Clipping enabled vs Stage 1 simulation, k = 3 and k = 2, 36 comparisons: identical clipped sets, angle between the
   enabled update and the simulated one 0.000 deg, |delta angle-to-unclipped| <= 0.037 deg (sketch error). All pass (< 1 deg).
   Recomputed gradients bitwise identical to pass 1 in every case (deterministic XLA).
   Correction form vs full second pass (exact anchors, k=3): rel. error <= 4.5e-6, angle <= 2.6e-4 deg.
4. Logging: exact top-trajectory influence agrees with the sketch-based influence; logged bound >= every influence.
5. Expert and heads: both receive the correction (e.g. noreplay U100: 34281 / 55967); both groups' errors <= 5.4e-6.
6. Cost: gradient 0.31 s/trajectory; norm 0.002 s/trajectory; pass 2 0.3 s per clipped trajectory; influence 0.3-0.6 s;
   host finish 3.2 s. => ~4-6 s per update (Phase 2 median update 95 s: ~5%; Phase 1 32-52 s: ~10-18%).
   Full second pass instead: +13 s (40 traj) to +30 s (96 traj) per update. Device memory: +1.7 GB (FP32 correction sum)
   while clipping; validator peak 22.5 of 32 GiB.
7. Historical runs: sha256 of all 1596 files under both runs identical before/after; frozen tasks 5-9 hashes OK.

## Before / after (k = 3; batches chosen for large effects, so frequencies are upper-biased)
| class | trajectories | clipped | max/median norm before (median, max) | after (max) |
|---|---|---|---|---|
| fresh | 464 | 6.2% | 181, 2428 | 3.00 |
| replay | 88 | 8.0% | 3.2, 445 | 3.00 |
| old | 480 | 0% | 1.4, 3.0 | 2.54 |
Max single-trajectory influence: exact anchors median 0.56 -> max 0.12 after; other batches median 0.985 -> max 0.15.
Spikes: replay U100 0.998 -> 0.018; noreplay U100 0.999 -> 0.043; noreplay U200 0.999 -> 0.053 (update turns 74-85 deg).
k = 2: same picture, clips 9% of fresh/replay and 1 old trajectory, max influence 0.16.

## Threshold sensitivity (all Stage 1 batches; sensitivity.md)
Replay run: k = 3 almost never clips (median 0%, angle <= 0.3 deg); k = 1.5 starts clipping ordinary batches (6% fresh/replay).
No-replay run at the c100/c200 weights: dominant trajectories in most batches; k = 3 cuts median max influence
0.43-0.67 -> 0.05; k = 5 leaves 0.07-0.22. Phase-1 fresh class (16 trajectories): bootstrap CV of the median 7-9%,
clip decisions flip for <= 5.5% (k = 3) / 6.7% (k = 2) of trajectories under resampling; the median tolerates up to 7
outliers of 16. k = 3 is a candidate chosen for ~zero collateral, not an established optimum.

## Discrepancies / limitations
- None with the exact references. The sketch-based Stage 1 angles differ by <= 0.04 deg (sketch error, expected).
- Not run end to end inside the training loop (that is training); trainer call sites are covered by the strip-hash
  test, the CPU trainer test, and the per-update recompute check that will be logged in any run.
- Phase-1 small batches (16 fresh, no-replay run) keep a top influence of up to 0.10-0.15 after clipping (1/16 = 0.06).
- The overhead figure excludes rollouts' variance; the 3.2 s host step could be reduced if needed.
