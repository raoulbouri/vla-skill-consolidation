"""Continual-learning OPD: tasks learned one after another with the t35 recipe (teacher warm-start, then student-driven
pathwise OPD); old tasks kept by a fixed-total replay buffer whose observations are labelled by each task's frozen teacher.
NumPy-only helpers, testable on CPU:

  config   configs/cl_opd.yaml (tasks, update counts, batch sizes, buffer size, seeds): every choice lives there
  phases   event-based task phases from per-step geometry, time quarters where geometry is unavailable
  buffer   priority-ordered, nested, fixed-total replay buffer on disk (any prefix of a task's order is balanced)
  sampler  per-update old-task batch: split evenly by task, then by phase, drawn without replacement, resumable
  matrix   success-rate matrix, NBT, normalised NBT, forward transfer, Wilson intervals
"""
