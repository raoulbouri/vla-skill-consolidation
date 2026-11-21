> Archived copy of `outputs/cl_opd/analysis/clip_stage2/sensitivity.md` (2026-11-21). Paths are relative to the repository root. Where it says "uncommitted", that was before commit 0f83c96; the code is now in git.

## Threshold sensitivity over all Stage 1 batches (medians; spike = logged grad > 1e4)

| run | controller | batches | k | clip frac fresh / replay / old | angle to unclipped (deg) | max influence before -> after |
|---|---|---|---|---|---|---|
| replay | teacher | 21 | 1.5 | 0.000 / - / 0.000 | 0.2 | 0.045 -> 0.045 |
| replay | teacher | 21 | 2.0 | 0.000 / - / 0.000 | 0.0 | 0.045 -> 0.045 |
| replay | teacher | 21 | 3.0 | 0.000 / - / 0.000 | 0.0 | 0.045 -> 0.045 |
| replay | teacher | 21 | 4.0 | 0.000 / - / 0.000 | 0.0 | 0.045 -> 0.045 |
| replay | teacher | 21 | 5.0 | 0.000 / - / 0.000 | 0.0 | 0.045 -> 0.045 |
| replay | teacher spike | 18 | 1.5 | 0.125 / - / 0.013 | 1.5 | 0.049 -> 0.048 |
| replay | teacher spike | 18 | 2.0 | 0.031 / - / 0.000 | 0.3 | 0.049 -> 0.049 |
| replay | teacher spike | 18 | 3.0 | 0.000 / - / 0.000 | 0.0 | 0.049 -> 0.049 |
| replay | teacher spike | 18 | 4.0 | 0.000 / - / 0.000 | 0.0 | 0.049 -> 0.049 |
| replay | teacher spike | 18 | 5.0 | 0.000 / - / 0.000 | 0.0 | 0.049 -> 0.049 |
| replay | student | 26 | 1.5 | 0.031 / 0.125 / 0.000 | 0.6 | 0.020 -> 0.020 |
| replay | student | 26 | 2.0 | 0.000 / 0.000 / 0.000 | 0.1 | 0.020 -> 0.020 |
| replay | student | 26 | 3.0 | 0.000 / 0.000 / 0.000 | 0.0 | 0.020 -> 0.019 |
| replay | student | 26 | 4.0 | 0.000 / 0.000 / 0.000 | 0.0 | 0.020 -> 0.019 |
| replay | student | 26 | 5.0 | 0.000 / 0.000 / 0.000 | 0.0 | 0.020 -> 0.020 |
| replay | student spike | 24 | 1.5 | 0.062 / 0.062 / 0.000 | 1.8 | 0.021 -> 0.020 |
| replay | student spike | 24 | 2.0 | 0.031 / 0.000 / 0.000 | 1.3 | 0.021 -> 0.020 |
| replay | student spike | 24 | 3.0 | 0.000 / 0.000 / 0.000 | 0.3 | 0.021 -> 0.020 |
| replay | student spike | 24 | 4.0 | 0.000 / 0.000 / 0.000 | 0.0 | 0.021 -> 0.020 |
| replay | student spike | 24 | 5.0 | 0.000 / 0.000 / 0.000 | 0.0 | 0.021 -> 0.020 |
| noreplay | teacher | 42 | 1.5 | 0.125 / - / - | 4.7 | 0.133 -> 0.109 |
| noreplay | teacher | 42 | 2.0 | 0.062 / - / - | 1.9 | 0.133 -> 0.113 |
| noreplay | teacher | 42 | 3.0 | 0.000 / - / - | 0.0 | 0.133 -> 0.127 |
| noreplay | teacher | 42 | 4.0 | 0.000 / - / - | 0.0 | 0.133 -> 0.133 |
| noreplay | teacher | 42 | 5.0 | 0.000 / - / - | 0.0 | 0.133 -> 0.133 |
| noreplay | teacher spike | 39 | 1.5 | 0.125 / - / - | 32.1 | 0.355 -> 0.105 |
| noreplay | teacher spike | 39 | 2.0 | 0.062 / - / - | 28.4 | 0.355 -> 0.111 |
| noreplay | teacher spike | 39 | 3.0 | 0.062 / - / - | 20.2 | 0.355 -> 0.126 |
| noreplay | teacher spike | 39 | 4.0 | 0.062 / - / - | 14.6 | 0.355 -> 0.165 |
| noreplay | teacher spike | 39 | 5.0 | 0.062 / - / - | 9.3 | 0.355 -> 0.215 |
| noreplay | student | 36 | 1.5 | 0.109 / 0.125 / - | 36.9 | 0.427 -> 0.046 |
| noreplay | student | 36 | 2.0 | 0.062 / 0.125 / - | 34.2 | 0.427 -> 0.047 |
| noreplay | student | 36 | 3.0 | 0.062 / 0.000 / - | 29.5 | 0.427 -> 0.050 |
| noreplay | student | 36 | 4.0 | 0.047 / 0.000 / - | 25.6 | 0.427 -> 0.054 |
| noreplay | student | 36 | 5.0 | 0.031 / 0.000 / - | 22.1 | 0.427 -> 0.068 |
| noreplay | student spike | 36 | 1.5 | 0.094 / 0.125 / - | 62.6 | 0.665 -> 0.046 |
| noreplay | student spike | 36 | 2.0 | 0.062 / 0.000 / - | 59.7 | 0.665 -> 0.047 |
| noreplay | student spike | 36 | 3.0 | 0.031 / 0.000 / - | 53.5 | 0.665 -> 0.047 |
| noreplay | student spike | 36 | 4.0 | 0.031 / 0.000 / - | 48.9 | 0.665 -> 0.051 |
| noreplay | student spike | 36 | 5.0 | 0.031 / 0.000 / - | 45.3 | 0.665 -> 0.071 |

## Phase 1 fresh class (16 trajectories): stability of the class median

| run | batches | bootstrap CV of the median (median) | decision flip rate k=3 / k=2 (median, max) | max/median norm (max) |
|---|---|---|---|---|
| replay | 39 | 0.071 | 0.000 (0.025) / 0.000 (0.046) | 389.80 |
| noreplay | 81 | 0.090 | 0.000 (0.055) / 0.000 (0.067) | 13996.30 |
