> Archived copy of `outputs/cl_opd/analysis/grad_stage1/REPORT.md` (2026-11-21). Paths are relative to the repository root. Where it says "uncommitted", that was before commit 0f83c96; the code is now in git.

# Stage 1: offline per-trajectory gradient analysis (t063 continual OPD)

Read-only. Inputs: stored batches + checkpoints of `full_20261108T214604Z/k1_t063` (replay, w=0.5) and
`t063_noreplay_20261109T150326Z/k1_t063`. Outputs here only; no file under either run changed (checked by the job).
Tables: summary.md; per-batch numbers: report.json; per-trajectory arrays: <run>/u*_<weights>.npz.

## 1. Validation
At the three exact-weight anchors per run (rows 1, 101, 201) every logged metric is reproduced:
gradient norms rel. diff <= 4e-8, KL (mean/fresh/replay/old/executed/unexecuted/padded), velocity RMS and
action errors identical. Stored batches + checkpoints are a faithful record of training.

## 2. What a spike is (exact anchors, the only conclusive evidence)
| anchor | logged grad | top trajectory norm | x median | share of batch direction | LOO angle | top trajectory KL (batch median) |
|---|---|---|---|---|---|---|
| replay U100 | 15.5k | 1.24e6 | 631 | 0.998 | 89 deg | 1334 (18) |
| no-replay U100 | 65.8k | 2.63e6 | 1607 | 0.999 | 83 deg | 503 (138) |
| no-replay U200 | 23.8k | 9.5e5 | 747 | 0.999 | 87 deg | 238 (22) |
- One trajectory makes the whole update; the other 39-79 are irrelevant. Because clip_by_global_norm(1.0) always
  rescales, the step has full size along that single trajectory's gradient (83-89 deg from the rest of the batch).
- Culprit is always a fresh, student-driven t063 rollout in opening / pre-contact (gripper-critical). Never t063 replay,
  never old t035.
- Loss is not extreme (KL 3.6-73x median) and velocity RMS is normal while the gradient is 600-1600x median:
  a sharp point of the pathwise-KL gradient, not a large error. KL-based filtering would miss it; norm-based catches it.
- Ordinary batches: max single-trajectory share 2-4% (replay, 80 traj) / 11-16% (no-replay, 40 traj).

## 3. Sensitivity to approximate weights
Logged spikes are almost never reproduced away from their own weights (replay 0/18, no-replay 0/31 at both checkpoints;
recomputed/logged norm ratios 0.01-6x). Spearman of per-trajectory norms between c100 and c200: 0.24 / 0.09; top-1
agreement 3-5%. => Outliers are properties of (trajectory, current weights), not fixed bad observations.
Consequences: (a) only the 3 exact spike anchors say anything about spikes; approximate-weight batches are used only
for ordinary-batch statistics; (b) an offline observation blacklist would not work; mitigation must act online.

## 4. Clipping simulations (per-trajectory clip, then the existing weighted mean)
At the exact spike anchors, every relative scheme (batch or class median x2/x3, fixed p90/p95) gives max share
0.02-0.05, LOO <= 9 deg, update within 6-9 deg of "batch without the outlier" (vs 83-89 deg unclipped).
Fixed p99 fails in no-replay (threshold contaminated by spikes: 137k at c100).
Collateral on ordinary batches (median): batch/class median x3 clips <= 5% of trajectories and turns the update
0 deg (replay) / 6 deg (no-replay); x2 turns 0-10 deg; x1 / p50 / p75 turn 2-16 deg and shrink the norm 10-20%.
Phase-1 trap: at replay U0 fresh teacher trajectories have median norm 5.1k vs old 1.0k. A batch-wide median
(dominated by the 40 old obs) clips 94-100% of fresh, turns the update 14-22 deg and cuts its norm to 0.34-0.51,
i.e. silently re-weights toward t035. Class-wise median x3 clips nothing there.
Fixed absolute thresholds drift ~4x across checkpoints (no-replay p50: 4930 -> 1596 -> 1296) => not portable.

## 5. Old/new interaction (exact sums)
cos(g_new, g_old) median 0.70-0.74 in Phase 2 (p10 -0.49), -0.52 at task start. w=0.5 gives ~50/50 share of the
update; w=0.2 gives ~84/16. Old t035 gradients never caused a spike; w is a plasticity/retention knob (Stage 3),
not a stability fix.

## 6. Recommendation
Online per-trajectory gradient clipping with threshold = 3 x median per-trajectory norm of the trajectory's own source
class (fresh t063 / t063 replay / old t035) in the current batch; clipped trajectories are scaled, not dropped;
loss weights and the optax chain (global clip 1.0, AdamW) unchanged. k=3 is the smallest multiplier with ~zero
collateral on ordinary batches that removes every observed spike; k=2 is the sensitivity arm.

## 7. Limitations
- Spike evidence is n=3 exact anchors (checkpoints exist only every 100 updates). Consistent across both runs, but small.
- One-step analysis: shows clipping removes single-trajectory takeover, not that it prevents collapse over 300 updates
  (drift, premature handoff and t063 difficulty remain possible causes).
- Root cause of the ~1e6-norm gradients (pathwise-KL sharpness at gripper events) is not identified; clipping treats
  the symptom. Logging per-trajectory norms during training will show whether spikes persist underneath.
- Cost: per-trajectory gradients took ~0.9 s/trajectory here (sequential, incl. sketching); training-time cost of
  per-example gradients must be measured.
- Class median with few members (Phase 1 fresh = 8 obs x 2 chains) is noisier.
