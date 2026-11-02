"""Task manifest v1.1 proposal logic (pure; category/similarity functions injected from build_task_manifest.py).

Changes from v1.0 (docs/NEXT_EXPERIMENTS.md):
  * the known-skill control (task 56) leaves the primary sequence and is kept as a separate control
  * R8: base-policy success <= r8_max successes in 10 episodes on selection init states 0-9 (real screening results)
  * minimal change: the other nine tasks and their order are retained; the replacement takes the control's position
  * v1.0 rules still apply to the replacement: R1-R4 (not in v1.0 `selection.excluded`), R5 (scene unused by the
    retained tasks), R6 (same category as the removed task unless allow_category_change), R7 (least similar to
    pretraining instructions; ties -> smaller index)
The result is a CANDIDATE (frozen: false) that runners refuse to load; freezing is a separate, explicit step.
"""

from __future__ import annotations

import copy
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from clvla.tasks import manifest_content_sha256

PRETRAIN_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")


def scene_of(name: str) -> str:
    return re.match(r"^((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_", name).group(1)


def screening_table(per_task: Sequence[Dict[str, Any]], selection_states: Sequence[int],
                    init_states_used: Sequence[int]) -> Dict[int, Tuple[int, int, int]]:
    """{libero_90_index: (successes, episodes, infra_errors)} from a screening summary; refuses non-selection states."""
    if not set(init_states_used) <= set(selection_states):
        raise ValueError(f"screening used init states {sorted(init_states_used)} outside the selection split")
    out = {}
    for t in per_task:
        idx = int(t["libero_90_index"])
        if idx in out:
            raise ValueError(f"task {idx} appears twice in screening results")
        out[idx] = (int(t["successes"]), int(t["episodes"]), int(t["infra_errors"]))
    return out


def r8_status(idx: int, table: Dict[int, Tuple[int, int, int]], r8_max: int, min_episodes: int = 10) -> str:
    if idx not in table:
        return "unscreened"
    s, n, err = table[idx]
    if err or n < min_episodes:
        return "invalid"
    return "pass" if s <= r8_max else "fail"


def propose_replacement(v10: Dict[str, Any], snapshot: Dict[str, Any], table: Dict[int, Tuple[int, int, int]], *,
                        control: int, r8_max: int, category_fn: Callable[[str], str],
                        jaccard_fn: Callable[[str, str], float], allow_category_change: bool = False,
                        candidate_indices: Optional[Sequence[int]] = None) -> Dict[str, Any]:
    tasks = v10["tasks"]
    removed = next((t for t in tasks if t["libero_90_index"] == control), None)
    if removed is None:
        raise ValueError(f"control task {control} is not in the v1.0 manifest")
    retained = [t for t in tasks if t["libero_90_index"] != control]
    used_scenes = {t["scene"] for t in retained}
    excluded = {int(k) for k in v10["selection"]["excluded"]}
    in_v10 = {t["libero_90_index"] for t in tasks}
    pre = [(s, r["language"]) for s in PRETRAIN_SUITES for r in snapshot[s]]
    rows = {r["idx"]: r for r in snapshot["libero_90"]}
    pool = candidate_indices if candidate_indices is not None else sorted(rows)
    considered, eligible = [], []
    for idx in pool:
        r = rows[idx]
        reasons = []
        if idx in excluded:
            reasons.append("v1.0 R1-R4 exclusion")
        if idx in in_v10:
            reasons.append("already in v1.0")
        if scene_of(r["name"]) in used_scenes:
            reasons.append("R5 scene used by a retained task")
        cat = category_fn(r["language"])
        if not allow_category_change and cat != removed["category"]:
            reasons.append(f"R6 category {cat} != {removed['category']}")
        status = r8_status(idx, table, r8_max)
        if status != "pass":
            reasons.append(f"R8 {status}")
        sim, near_suite, near = max((jaccard_fn(r["language"], p), s, p) for s, p in pre)
        entry = {"libero_90_index": idx, "name": r["name"], "instruction": r["language"], "scene": scene_of(r["name"]),
                 "category": cat, "screening": list(table[idx]) if idx in table else None, "r8": status,
                 "nearest_pretrain_jaccard": round(sim, 4), "nearest_pretrain_suite": near_suite,
                 "nearest_pretrain_instruction": near, "rejected_because": reasons}
        considered.append(entry)
        if not reasons:
            eligible.append(entry)
    audit = [{"libero_90_index": t["libero_90_index"], "r8": r8_status(t["libero_90_index"], table, r8_max),
              "screening": list(table.get(t["libero_90_index"], ())) or None} for t in retained]
    best = min(eligible, key=lambda e: (e["nearest_pretrain_jaccard"], e["libero_90_index"])) if eligible else None
    return {"removed": removed, "retained": retained, "considered": considered, "replacement": best,
            "retained_r8_audit": audit,
            "needs_screening": [e["libero_90_index"] for e in considered if e["r8"] == "unscreened"
                                and e["rejected_because"] == ["R8 unscreened"]]}


def candidate_manifest(v10: Dict[str, Any], proposal: Dict[str, Any], replacement_row: Dict[str, Any], *,
                       screening_ref: str, r8_max: int, allow_category_change: bool) -> Dict[str, Any]:
    """v1.1 CANDIDATE manifest: v1.0 order with the control's slot taken by the replacement; frozen: false."""
    m = copy.deepcopy(v10)
    pos = proposal["removed"]["position"]
    new_tasks = []
    for t in m["tasks"]:
        if t["position"] == pos:
            new_tasks.append(dict(replacement_row, position=pos))
        else:
            new_tasks.append(t)
    m.update({
        "manifest_version": "1.1.0-candidate",
        "frozen": False,
        "status": "candidate: review screening evidence, then freeze with scripts/propose_manifest_v1_1.py --freeze",
        "supersedes": {"manifest_version": v10["manifest_version"], "content_sha256": v10["content_sha256"]},
        "tasks": new_tasks,
        "control_tasks": [{"libero_90_index": proposal["removed"]["libero_90_index"], "name": proposal["removed"]["name"],
                           "reason": "known-skill control: high zero-shot success of the base policy"}],
    })
    m["selection"] = dict(m["selection"], **{
        "v1_1_changes": {
            "R8": f"base pi05_libero successes <= {r8_max}/10 on selection init states 0-9 (screening: {screening_ref})",
            "replacement_rule": "minimal change: retained tasks and order unchanged; replacement at the control's position",
            "allow_category_change": allow_category_change,
            "considered": proposal["considered"], "retained_r8_audit": proposal["retained_r8_audit"]}})
    m.pop("content_sha256", None)
    m["content_sha256"] = manifest_content_sha256(m)
    return m


def missing_or_failed_audit(proposal: Dict[str, Any]) -> List[int]:
    return [a["libero_90_index"] for a in proposal["retained_r8_audit"] if a["r8"] != "pass"]
