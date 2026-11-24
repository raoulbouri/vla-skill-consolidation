"""configs/cl_opd.yaml: load, check, hash. The hash is stored in the run directory; a run never continues under another config."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml

VARIANTS = {"expert300m": ("gemma_2b", "gemma_300m"), "lora": ("gemma_2b_lora", "gemma_300m_lora")}
REQUIRED = {"tasks", "student", "profile", "per_task", "old_tasks", "evaluation"}


def load(path) -> dict:
    c = yaml.safe_load(Path(path).read_text())
    check(c)
    return c


def check(c: dict):
    missing = REQUIRED - set(c)
    assert not missing, f"config lacks {sorted(missing)}"
    ids = [t["id"] for t in c["tasks"]]
    assert len(ids) == len(set(ids)), "duplicate task ids"
    assert len({t["position"] for t in c["tasks"]}) == len(ids), "duplicate positions"
    for t in c["tasks"]:
        assert t["variant"] in VARIANTS, f"{t['id']}: unknown teacher variant {t['variant']}"
        assert t["geometry"] in ("microwave", "none"), f"{t['id']}: unknown geometry {t['geometry']}"
    assert c["student"] in ("full", "lora")
    p, o = c["per_task"], c["old_tasks"]
    assert 0 < p["warm_updates"] < p["total_updates"], "need 0 < warm_updates < total_updates"
    assert all(0 < u <= p["total_updates"] for u in p["checkpoint_updates"])
    assert p.get("traj_clip_k", 0) == 0 or p["traj_clip_k"] >= 1.0, "traj_clip_k: 0 (off) or a multiplier >= 1 of the class median"
    assert p.get("kl_gate", True) in (True, False), "kl_gate: true (default) or false"
    assert p.get("trust_region_k", 0) == 0 or p["trust_region_k"] >= 1.0, "trust_region_k: 0 (off) or a multiplier >= 1 of the running median"
    assert 0.0 <= o["weight"] < 1.0 and o["per_update"] >= 0 and o["buffer_total"] > 0
    assert c["evaluation"]["modes"] in ("ode", "sde", "ode,sde")


def sha(c: dict) -> str:
    return hashlib.sha256(json.dumps(c, sort_keys=True).encode()).hexdigest()


def held_out(profile: str) -> list:
    return {"n10": list(range(10, 50)), "n25": list(range(25, 50))}[profile]
