"""Derive the frozen 10-task LIBERO-90 sequence from the pinned official LIBERO metadata.

Selection rule (deterministic; documented in docs/EXPERIMENT_PROTOCOL.md):
  R1  the task resolves in official LIBERO libero_90, its demo file has exactly 50 demos whose instruction equals the
      benchmark instruction, its demo file is listed in configs/dataset.lock.json, and it has 50 init states
  R2  its instruction is not identical to, nor a substring of, any instruction of the suites pi05_libero was
      fine-tuned on (libero_spatial, libero_object, libero_goal, libero_10)
  R3  its scene is not one of the scenes used by libero_10 (libero_10 tasks are LIBERO-100 scenes shared with
      LIBERO-90; libero_spatial/object/goal use their own scenes)
  R4  longest demo <= max_steps - 25 (fits the 400-step evaluation cap with margin)
  R5  at most one task per scene
  R6  category quota: articulated 2, container 2, spatial_relation 4, multi_step 2 (rule-based categories below)
  R7  among all selections satisfying R1-R6, minimise the summed nearest-neighbour word-Jaccard similarity to the
      pretraining instructions; ties -> lexicographically smallest sorted LIBERO-90 index tuple
  Order: a fixed permutation of the selected tasks, sorted by index then shuffled with random.Random(ORDER_SEED).

Usage (sim env):
  scripts/py-sim scripts/build_task_manifest.py            # write configs/libero90_tasks.json + task_manifest.yaml
  scripts/py-sim scripts/build_task_manifest.py --check    # re-derive and fail if the committed manifest differs
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
from pathlib import Path

import yaml

from clvla.config import load_experiment_config
from clvla.data import validate_demo_file
from clvla.tasks import manifest_content_sha256, parse_manifest

ROOT = Path(__file__).resolve().parents[1]
PRETRAIN_SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
QUOTA = {"articulated": 2, "container": 2, "spatial_relation": 4, "multi_step": 2}
ORDER_SEED = 0
MANIFEST_VERSION = "1.0.0"
STOP = {"the", "it", "of", "and", "a", "on", "in", "to", "up"}


def scene_of(name: str) -> str:
    return re.match(r"^((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_", name).group(1)


def tokens(s: str) -> set:
    return set(re.findall(r"[a-z]+", s.lower())) - STOP


def jaccard(a: str, b: str) -> float:
    ta, tb = tokens(a), tokens(b)
    return len(ta & tb) / len(ta | tb)


def category(instr: str) -> str:
    """Rule-based skill category from the instruction text."""
    s = re.sub(r"^pick up (.+?) and (put|place) it ", r"put \1 ", instr)  # 'pick up X and put it in Y' is one skill
    if re.search(r" and (put|place|close|open|turn)\b", s):
        return "multi_step"
    if re.match(r"^(open|close|turn on|turn off) ", s):
        return "articulated"
    if re.search(
        r"on top of|under |to the (right|left|front) of|\bstack\b|shelf|at the (front|back)|\bmiddle\b|"
        r"on the (left|right)\b",
        s,
    ):
        return "spatial_relation"
    if re.search(r" in(to)? the .*(tray|basket|drawer|compartment)", s):
        return "container"
    return "other"


def inventory() -> dict:
    import torch
    from libero.libero import benchmark, get_libero_path

    bd = benchmark.get_benchmark_dict()
    snap = {"libero_repo": "https://github.com/Lifelong-Robot-Learning/LIBERO"}
    for s in ("libero_90", *PRETRAIN_SUITES):
        suite = bd[s]()
        rows = []
        for i in range(suite.n_tasks):
            t = suite.get_task(i)
            row = {
                "idx": i,
                "name": t.name,
                "language": t.language,
                "problem_folder": t.problem_folder,
                "bddl_file": t.bddl_file,
                "init_states_file": t.init_states_file,
            }
            if s == "libero_90":
                p = Path(get_libero_path("init_states")) / t.problem_folder / t.init_states_file
                row["num_init_states"] = int(torch.load(str(p)).shape[0])
            rows.append(row)
        snap[s] = rows
    return snap


def derive(snap: dict, cfg, dataset_lock: dict) -> dict:
    pre = [(s, r["language"]) for s in PRETRAIN_SUITES for r in snap[s]]
    l10_scenes = {scene_of(r["name"]) for r in snap["libero_10"]}
    data_root = Path(cfg.data.root)
    cands, excluded = [], {}
    for r in snap["libero_90"]:
        name, instr = r["name"], r["language"]
        rel = f"libero_90/{name}_demo.hdf5"
        reasons = []
        idx = validate_demo_file(
            data_root / rel,
            expected_instruction=instr,
            expected_demos=cfg.data.demos_per_task,
            expected_image_hw=(cfg.data.image_size, cfg.data.image_size),
            check_values=False,
        )
        if rel not in dataset_lock["files"]:
            reasons.append("R1: demo file not in dataset lock")
        if r["num_init_states"] != 50:
            reasons.append(f"R1: {r['num_init_states']} init states")
        exact = sorted({s for s, p in pre if p == instr})
        sub = sorted({s for s, p in pre if p != instr and instr in p})
        if exact:
            reasons.append(f"R2: instruction identical to {exact}")
        if sub:
            reasons.append(f"R2: instruction is a sub-goal of {sub}")
        if scene_of(name) in l10_scenes:
            reasons.append("R3: scene used by libero_10")
        if max(idx.lengths) > cfg.simulator.max_steps - 25:
            reasons.append(f"R4: longest demo {max(idx.lengths)} steps")
        sim, near_suite, near = max((jaccard(instr, p), s, p) for s, p in pre)
        info = {
            "idx": r["idx"],
            "name": name,
            "instruction": instr,
            "scene": scene_of(name),
            "category": category(instr),
            "nearest_pretrain_jaccard": round(sim, 4),
            "nearest_pretrain_suite": near_suite,
            "nearest_pretrain_instruction": near,
            "demo_len_min": min(idx.lengths),
            "demo_len_max": max(idx.lengths),
            "demo_len_mean": round(sum(idx.lengths) / len(idx.lengths), 1),
            "row": r,
            "exact_match_suites": exact,
            "subgoal_of_suites": sub,
            "same_scene_as_libero_10": scene_of(name) in l10_scenes,
        }
        if reasons:
            excluded[r["idx"]] = reasons
        else:
            cands.append(info)

    # R5-R7: best task per (scene, category), then exact search over category->scene assignments.
    best = {}
    for c in cands:
        key = (c["scene"], c["category"])
        if c["category"] in QUOTA and (
            key not in best
            or (c["nearest_pretrain_jaccard"], c["idx"]) < (best[key]["nearest_pretrain_jaccard"], best[key]["idx"])
        ):
            best[key] = c
    scenes = sorted({s for s, _ in best})
    slots = [cat for cat, n in sorted(QUOTA.items()) for _ in range(n)]
    perms = sorted(set(itertools.permutations(slots)))  # 18900 distinct category orders
    choice, choice_key = None, None
    for scene_subset in itertools.combinations(scenes, len(slots)):
        for perm in perms:
            picks = [best.get((sc, cat)) for sc, cat in zip(scene_subset, perm)]
            if any(p is None for p in picks):
                continue
            key = (round(sum(p["nearest_pretrain_jaccard"] for p in picks), 6), tuple(sorted(p["idx"] for p in picks)))
            if choice_key is None or key < choice_key:
                choice, choice_key = picks, key
    if choice is None:
        sys.exit("no selection satisfies R1-R6; present a manual proposal for approval instead")

    ordered = sorted(choice, key=lambda p: p["idx"])
    random.Random(ORDER_SEED).shuffle(ordered)
    tasks = []
    for pos, p in enumerate(ordered):
        r = p["row"]
        rel = f"libero_90/{p['name']}_demo.hdf5"
        tasks.append(
            {
                "position": pos,
                "libero_90_index": p["idx"],
                "name": p["name"],
                "instruction": p["instruction"],
                "scene": p["scene"],
                "category": p["category"],
                "bddl_file": f"{r['problem_folder']}/{r['bddl_file']}",
                "init_states_file": f"{r['problem_folder']}/{r['init_states_file']}",
                "num_init_states": r["num_init_states"],
                "demo_file": rel,
                "demo_sha256": dataset_lock["files"][rel]["sha256"],
                "demo_size_bytes": dataset_lock["files"][rel]["size"],
                "n_demos": cfg.data.demos_per_task,
                "demo_len_min": p["demo_len_min"],
                "demo_len_max": p["demo_len_max"],
                "demo_len_mean": p["demo_len_mean"],
                "overlap": {
                    "exact_instruction_match": p["exact_match_suites"],
                    "subgoal_of": p["subgoal_of_suites"],
                    "same_scene_as_libero_10": p["same_scene_as_libero_10"],
                    "nearest_pretrain_suite": p["nearest_pretrain_suite"],
                    "nearest_pretrain_instruction": p["nearest_pretrain_instruction"],
                    "nearest_pretrain_jaccard": p["nearest_pretrain_jaccard"],
                },
            }
        )
    return {"tasks": tasks, "excluded": excluded, "n_candidates": len(cands), "objective": choice_key[0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    dataset_lock = json.loads((ROOT / cfg.data.lock_file).read_text())
    snap = inventory()
    result = derive(snap, cfg, dataset_lock)
    libero_dir = Path(os.environ["LIBERO_DIR"])
    import subprocess

    libero_sha = subprocess.run(
        ["git", "-C", str(libero_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "frozen": True,
        "benchmark": "libero_90",
        "description": "Custom 10-task sequential-learning sequence drawn from LIBERO-90 (NOT the official libero_10 "
        "suite). Fixed order; never reselect between runs.",
        "source": {
            "libero_repo": "https://github.com/Lifelong-Robot-Learning/LIBERO",
            "libero_commit": libero_sha,
            "dataset_repo": dataset_lock["repo"],
            "dataset_revision": dataset_lock["revision"],
            "data_root_env": "LIBERO_DATA_ROOT",
        },
        "selection": {
            "procedure": "scripts/build_task_manifest.py (rules R1-R7 in its docstring and docs/EXPERIMENT_PROTOCOL.md)",
            "quota": QUOTA,
            "order_seed": ORDER_SEED,
            "pretrain_suites_checked": list(PRETRAIN_SUITES),
            "num_eligible_candidates": result["n_candidates"],
            "objective_sum_nearest_jaccard": result["objective"],
            "excluded": {str(k): v for k, v in sorted(result["excluded"].items())},
            "overlap_caveat": "No task is strictly disjoint from pretraining in skills/objects: all share the Panda "
            "robot, OSC action space and many object meshes with LIBERO-Goal/Object/Spatial/10, and "
            "several instructions paraphrase pretraining ones (see per-task overlap). Only exact "
            "instruction, sub-goal and libero_10-scene overlap are excluded.",
        },
        "tasks": result["tasks"],
    }
    manifest["content_sha256"] = manifest_content_sha256(manifest)
    parse_manifest(manifest, "<generated>")
    out = ROOT / "configs/task_manifest.yaml"
    header = (
        "# FROZEN task manifest: generated by scripts/build_task_manifest.py; do not edit by hand.\n"
        "# content_sha256 covers every other key; clvla.tasks refuses to load an edited copy.\n"
    )
    text = header + yaml.safe_dump(manifest, sort_keys=False, width=120, allow_unicode=False)
    if args.check:
        committed = parse_manifest(yaml.safe_load(out.read_text()), str(out))
        if committed.content_sha256 != manifest["content_sha256"]:
            sys.exit(
                f"MISMATCH: committed manifest {committed.content_sha256[:12]} != re-derived "
                f"{manifest['content_sha256'][:12]}"
            )
        print(f"OK: committed manifest re-derives exactly (content_sha256 {manifest['content_sha256']})")
        return
    if out.exists():
        old = yaml.safe_load(out.read_text())
        if old.get("content_sha256") != manifest["content_sha256"]:
            sys.exit(
                f"{out} exists with different content; a frozen manifest is never overwritten. Bump "
                "MANIFEST_VERSION and move the old file aside deliberately if the change is intended."
            )
    (ROOT / "configs/libero90_tasks.json").write_text(json.dumps(snap, indent=1) + "\n")
    out.write_text(text)
    for t in manifest["tasks"]:
        print(t["position"], t["libero_90_index"], t["category"], t["name"], t["overlap"]["nearest_pretrain_jaccard"])
    print("content_sha256", manifest["content_sha256"])


if __name__ == "__main__":
    main()
