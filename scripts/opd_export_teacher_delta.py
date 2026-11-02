"""Export what a LoRA teacher adds to or changes from its base checkpoint, plus sha256 of every leaf as the policy loads it (BF16). Read-only on both checkpoints.
  scripts/py-policy scripts/opd_export_teacher_delta.py <teacher_params> <base_params> <out_dir>
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from flax import traverse_util
from openpi.models import model as _model


def leaves(path):
    return traverse_util.flatten_dict(_model.restore_params(path, restore_type=np.ndarray), sep="/")


def bf16_sha(a):
    b = np.asarray(jnp.asarray(a).astype(jnp.bfloat16))
    return hashlib.sha256(str(b.dtype).encode() + str(b.shape).encode() + b.tobytes()).hexdigest()


teacher, base, out = leaves(sys.argv[1]), leaves(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
t_sha, b_sha = {k: bf16_sha(v) for k, v in teacher.items()}, {k: bf16_sha(v) for k, v in base.items()}
delta = {k: v for k, v in teacher.items() if t_sha[k] != b_sha.get(k)}
np.savez(out / "delta.npz", **{k.replace("/", "__"): v for k, v in delta.items()})
(out / "leaf_sha.json").write_text(json.dumps({"teacher_bf16": t_sha, "base_bf16": b_sha, "delta_keys": sorted(delta), "n_teacher": len(teacher), "n_base": len(base)}, indent=1))
print(len(teacher), len(base), len(delta), sum(v.size for v in delta.values()))
