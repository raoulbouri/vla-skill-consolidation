import copy
import json
import re
import types

import pytest

from clvla.tasks import (
    ManifestError,
    check_against_snapshot,
    init_state_indices,
    manifest_content_sha256,
    parse_manifest,
    resolve_tasks,
)


def resign(raw):
    raw["content_sha256"] = manifest_content_sha256(raw)
    return raw


def test_committed_manifest_has_ten_unique_valid_libero90_tasks_in_fixed_order(manifest_raw, repo_root):
    m = parse_manifest(manifest_raw)
    assert m.benchmark == "libero_90" and len(m.tasks) == 10
    assert [t.position for t in m.tasks] == list(range(10))
    assert len({t.libero_90_index for t in m.tasks}) == 10
    assert len({t.name for t in m.tasks}) == 10
    assert all(0 <= t.libero_90_index < 90 for t in m.tasks)
    # the frozen order (any change here must be a deliberate manifest version bump)
    assert [t.libero_90_index for t in m.tasks] == [78, 84, 16, 56, 35, 42, 28, 3, 89, 63]
    check_against_snapshot(m, repo_root / "configs/libero90_tasks.json")


def test_manifest_tasks_avoid_known_pretraining_overlap(manifest_raw, repo_root):
    snap = json.loads((repo_root / "configs/libero90_tasks.json").read_text())
    pre = {r["language"] for s in ("libero_spatial", "libero_object", "libero_goal", "libero_10") for r in snap[s]}
    l10_scenes = {
        re.match(r"^((?:KITCHEN|LIVING_ROOM|STUDY)_SCENE\d+)_", r["name"]).group(1) for r in snap["libero_10"]
    }
    for t in parse_manifest(manifest_raw).tasks:
        assert t.instruction not in pre
        assert not any(t.instruction in p for p in pre)
        assert t.scene not in l10_scenes
    assert len({t.scene for t in parse_manifest(manifest_raw).tasks}) == 10


@pytest.mark.parametrize(
    "mutate,msg",
    [
        (lambda r: r["tasks"].pop(), "exactly 10"),
        (lambda r: r["tasks"].append(copy.deepcopy(r["tasks"][0])), "exactly 10"),
        (lambda r: r["tasks"][1].update(libero_90_index=r["tasks"][0]["libero_90_index"]), "duplicate libero_90_index"),
        (lambda r: r["tasks"][2].update(libero_90_index=90), "not a LIBERO-90 index"),
        (lambda r: r["tasks"][2].update(libero_90_index="3"), "not a LIBERO-90 index"),
        (
            lambda r: r["tasks"][3].update(name="pick_up_the_alphabet_soup_and_place_it_in_the_basket"),
            "not a LIBERO-90 task name",
        ),
        (lambda r: r["tasks"][4].update(bddl_file="libero_10/x.bddl"), "bddl_file"),
        (lambda r: r["tasks"][4].update(demo_file="libero_10/x_demo.hdf5"), "demo_file"),
        (lambda r: (r["tasks"][0].update(position=1), r["tasks"][1].update(position=0)), "positions must be"),
        (lambda r: r.update(benchmark="libero_10"), "benchmark must be"),
        (lambda r: r.update(frozen=False), "not frozen"),
        (lambda r: r["tasks"][0].pop("instruction"), "missing keys"),
    ],
)
def test_invalid_manifests_rejected(manifest_raw, mutate, msg):
    raw = copy.deepcopy(manifest_raw)
    mutate(raw)
    with pytest.raises(ManifestError, match=msg):
        parse_manifest(resign(raw))


def test_hand_edited_manifest_fails_checksum(manifest_raw):
    raw = copy.deepcopy(manifest_raw)
    raw["tasks"][0], raw["tasks"][1] = raw["tasks"][1], raw["tasks"][0]  # silent reorder
    raw["tasks"][0]["position"], raw["tasks"][1]["position"] = 0, 1
    with pytest.raises(ManifestError, match="checksum mismatch"):
        parse_manifest(raw)


def test_snapshot_mismatch_detected(manifest_raw, repo_root, tmp_path):
    snap = json.loads((repo_root / "configs/libero90_tasks.json").read_text())
    m = parse_manifest(manifest_raw)
    victim = m.tasks[0].libero_90_index
    for r in snap["libero_90"]:
        if r["idx"] == victim:
            r["language"] = "something else"
    p = tmp_path / "snap.json"
    p.write_text(json.dumps(snap))
    with pytest.raises(ManifestError, match="does not match LIBERO-90 snapshot"):
        check_against_snapshot(m, p)


def _fake_suite(snapshot):
    rows = snapshot["libero_90"]
    tasks = [
        types.SimpleNamespace(
            name=r["name"],
            language=r["language"],
            problem_folder=r["problem_folder"],
            bddl_file=r["bddl_file"],
            init_states_file=r["init_states_file"],
        )
        for r in rows
    ]
    return types.SimpleNamespace(n_tasks=len(tasks), get_task=lambda i: tasks[i])


def test_resolution_against_injected_suite(manifest_raw, repo_root, tmp_path):
    snap = json.loads((repo_root / "configs/libero90_tasks.json").read_text())
    m = parse_manifest(manifest_raw)
    for t in m.tasks:  # create the files resolution checks for
        for sub, rel in (("bddl", t.bddl_file), ("init", t.init_states_file)):
            p = tmp_path / sub / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("x")
    paths = {"bddl_files": str(tmp_path / "bddl"), "init_states": str(tmp_path / "init")}
    resolved = resolve_tasks(m, _fake_suite(snap), paths)
    assert [r.spec.libero_90_index for r in resolved] == [t.libero_90_index for t in m.tasks]

    snap["libero_90"][m.tasks[3].libero_90_index]["name"] = "KITCHEN_SCENE1_other"
    with pytest.raises(ManifestError, match="does not resolve"):
        resolve_tasks(m, _fake_suite(snap), paths)


def test_resolution_reports_missing_files(manifest_raw, repo_root, tmp_path):
    snap = json.loads((repo_root / "configs/libero90_tasks.json").read_text())
    with pytest.raises(ManifestError, match="bddl file missing"):
        resolve_tasks(
            parse_manifest(manifest_raw), _fake_suite(snap), {"bddl_files": str(tmp_path), "init_states": str(tmp_path)}
        )


def test_init_state_indices():
    assert init_state_indices(list(range(10, 50)), 3) == [10, 11, 12]
    with pytest.raises(ValueError):
        init_state_indices([1, 2], 3)
