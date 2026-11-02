"""Manifest v1.1 proposal: R8 from real screening tables, minimal replacement, audit, candidate never frozen."""

import json

import pytest
import yaml

from clvla.selection_v11 import candidate_manifest, propose_replacement, r8_status, screening_table
from clvla.tasks import ManifestError, manifest_content_sha256, parse_manifest
from conftest import load_script


@pytest.fixture(scope="module")
def rules():
    mod = load_script("build_task_manifest")  # the exact v1.0 category/jaccard functions
    return mod.category, mod.jaccard


@pytest.fixture
def inputs(repo_root):
    v10 = yaml.safe_load((repo_root / "configs/task_manifest.yaml").read_text())
    snap = json.loads((repo_root / "configs/libero90_tasks.json").read_text())
    scfg = yaml.safe_load((repo_root / "configs/proposed/screening_v1.1.yaml").read_text())
    return v10, snap, scfg


def table(rows):
    return {i: (s, 10, 0) for i, s in rows.items()}


def test_screening_table_rejects_non_selection_states():
    per = [{"libero_90_index": 55, "successes": 1, "episodes": 10, "infra_errors": 0}]
    assert screening_table(per, range(10), range(10)) == {55: (1, 10, 0)}
    with pytest.raises(ValueError, match="outside the selection split"):
        screening_table(per, range(10), range(10, 20))
    with pytest.raises(ValueError, match="twice"):
        screening_table(per + per, range(10), range(10))


def test_r8_status():
    t = {1: (2, 10, 0), 2: (3, 10, 0), 3: (0, 10, 1), 4: (0, 5, 0)}
    assert [r8_status(i, t, 2) for i in (1, 2, 3, 4, 5)] == ["pass", "fail", "invalid", "invalid", "unscreened"]


def test_replacement_follows_r5_r6_r8_and_similarity(inputs, rules):
    v10, snap, scfg = inputs
    cat, jac = rules
    t = table({55: 0, 57: 1, 58: 5, 59: 2, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0})
    p = propose_replacement(v10, snap, t, control=56, r8_max=2, category_fn=cat, jaccard_fn=jac,
                            candidate_indices=scfg["replacement_candidates"])
    rep = p["replacement"]
    assert rep is not None and rep["category"] == "container" and rep["scene"] == "LIVING_ROOM_SCENE3"
    assert rep["libero_90_index"] in (55, 57, 59) and rep["r8"] == "pass"
    eligible = [e for e in p["considered"] if not e["rejected_because"]]
    assert rep == min(eligible, key=lambda e: (e["nearest_pretrain_jaccard"], e["libero_90_index"]))
    by = {e["libero_90_index"]: e for e in p["considered"]}
    assert "R8 fail" in by[58]["rejected_because"]
    assert any(r.startswith("R6") for r in by[6]["rejected_because"])  # K1 tasks are not containers


def test_no_replacement_when_r8_fails_everywhere_and_category_change_is_opt_in(inputs, rules):
    v10, snap, scfg = inputs
    cat, jac = rules
    t = table({55: 9, 57: 8, 58: 10, 59: 7, 6: 0, 7: 1, 8: 0, 9: 6, 10: 2})
    strict = propose_replacement(v10, snap, t, control=56, r8_max=2, category_fn=cat, jaccard_fn=jac,
                                 candidate_indices=scfg["replacement_candidates"])
    assert strict["replacement"] is None
    relaxed = propose_replacement(v10, snap, t, control=56, r8_max=2, category_fn=cat, jaccard_fn=jac,
                                  allow_category_change=True, candidate_indices=scfg["replacement_candidates"])
    assert relaxed["replacement"]["libero_90_index"] in (6, 7, 8, 10)


def test_unscreened_candidates_are_reported_not_guessed(inputs, rules):
    v10, snap, scfg = inputs
    cat, jac = rules
    p = propose_replacement(v10, snap, {}, control=56, r8_max=2, category_fn=cat, jaccard_fn=jac,
                            candidate_indices=scfg["replacement_candidates"])
    assert p["replacement"] is None and set(p["needs_screening"]) == {55, 57, 58, 59}
    assert all(a["r8"] == "unscreened" for a in p["retained_r8_audit"])


def test_retained_tasks_are_audited_not_swapped(inputs, rules):
    v10, snap, scfg = inputs
    cat, jac = rules
    t = table({55: 0, 28: 4, **{i: 0 for i in scfg["retained_tasks"] if i != 28}})
    p = propose_replacement(v10, snap, t, control=56, r8_max=2, category_fn=cat, jaccard_fn=jac,
                            candidate_indices=scfg["replacement_candidates"])
    audit = {a["libero_90_index"]: a["r8"] for a in p["retained_r8_audit"]}
    assert audit[28] == "fail" and len(p["retained"]) == 9 and 28 in [x["libero_90_index"] for x in p["retained"]]


def test_candidate_manifest_keeps_order_and_is_not_loadable_until_frozen(inputs, rules):
    v10, snap, scfg = inputs
    cat, jac = rules
    p = propose_replacement(v10, snap, table({55: 0}), control=56, r8_max=2, category_fn=cat, jaccard_fn=jac,
                            candidate_indices=[55])
    rep = p["replacement"]
    row = {"libero_90_index": rep["libero_90_index"], "name": rep["name"], "instruction": rep["instruction"],
           "scene": rep["scene"], "category": rep["category"],
           "bddl_file": f"libero_90/{rep['name']}.bddl", "init_states_file": f"libero_90/{rep['name']}.pruned_init",
           "demo_file": f"libero_90/{rep['name']}_demo.hdf5"}
    cand = candidate_manifest(v10, p, row, screening_ref="outputs/screening/x", r8_max=2, allow_category_change=False)
    assert [t["libero_90_index"] for t in cand["tasks"]] == [78, 84, 16, 55, 35, 42, 28, 3, 89, 63]
    assert cand["frozen"] is False and cand["control_tasks"][0]["libero_90_index"] == 56
    assert cand["supersedes"]["content_sha256"] == v10["content_sha256"]
    assert cand["content_sha256"] == manifest_content_sha256(cand)
    with pytest.raises(ManifestError, match="not frozen"):
        parse_manifest(cand)
    frozen = dict(cand, frozen=True, manifest_version="1.1.0")
    frozen["content_sha256"] = manifest_content_sha256(frozen)
    assert [t.libero_90_index for t in parse_manifest(frozen).tasks][3] == 55
    assert v10["manifest_version"] == "1.0.0"  # the input was not modified
