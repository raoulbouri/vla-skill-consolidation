# Continual-learning OPD: results, analysis, open questions (2026-11-21)

Recipe and commands: `README.md`, section "Continual learning across tasks". Run directories on the original machine:
`outputs/cl_opd/full_20261108T214604Z` (with old-task replay) and `outputs/cl_opd/t063_noreplay_20261109T150326Z`
(task 1 without replay, started from the same task-0 student).

## Results (held-out demo_25..49, 25 starts per cell, ODE sampling)

| After task | Run | t035 | t063 | t042 (next, untrained) |
|---|---|---|---|---|
| 0 (t035) | full | 24/25 | 1/25 | |
| 1 (t063) | full, old-task replay w=0.5 | 25/25 | **0/25** | 0/25 |
| 1 (t063) | no old-task replay | **0/25** | 13/25 | 0/25 |

Teachers on the same starts: t035 24/25, t063 25/25. Base π₀.₅: t035 1/25, t063 0/25. Replay keeps the old task but
the new one does not learn; without replay the new task is learned and the old one is forgotten.

Training curve of the replay run: student-driven success on t063 reached 39% over updates 101-150, then fell to 0 and
stayed there, right after gradient spikes (gradient norm up to 6e4, global clip 1.0).

## What the analysis found

- `grad_stage1.md`: every spike update is one fresh, student-driven t063 trajectory with a gradient 600-1600x the batch
  median, carrying 99.8-99.9% of the update's direction. The part of the gradient that flows back through the
  student's own sampling chain causes it; it is large where the velocity field is steep, which on t063 is near gripper
  open/close decisions (t035 has none, t063 has 4 per demo).
- `clip_stage2.md`, `clip_stage2_sensitivity.md`: per-trajectory clipping (`per_task.traj_clip_k: 3`) implemented and
  validated offline on the stored batches: the spike trajectory's influence falls from 0.998 to 0.02-0.05, the global
  clip and the loss weights stay, and with clipping off the trainer is bit-identical to the original. It has not been
  used in training yet.
- Pre-flight, CPU part (`scripts/cl_opd_preflight_cpu.py`): t016 (2 gripper toggles, 17% of steps near one) is the next
  riskiest task; t042 has 1. Memory for task 4 (5 teachers resident) fits a 32 GB card. The GPU part (spike anatomy,
  look-ahead, interference, forgetting threshold; `scripts/cl_opd_preflight_gpu.py --section abc`) needs a whole
  32 GB card and has not run yet.

## Open questions and the experiments that answer them

| | Experiment | Answers |
|---|---|---|
| A | Teacher takeover on t063: the student drives, the teacher takes over mid-episode; does it recover? (simulation only, about 30-60 min) | Is the teacher reliable at states the student visits? |
| B | From the t063 update-100 state with a fresh optimizer, about 50 student-driven updates, arm 1 clipping k=3, arm 2 clipping + an action-space trust region (about 1.5 GPU-hours per arm; training) | Is clipping enough, or is a trust region needed? Does Adam's second moment stay inflated after a spike? |
| C | KL and development-set success at the no-replay t063 checkpoints 100-400 | Does the velocity KL track task success on t063? Is 400 updates enough? |

Not yet tested: the old-task buffer holds only teacher-driven states; plasticity loss after earlier tasks.
