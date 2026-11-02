"""Specialist runner: selection rules, serial execution, lock, GPU guard, summaries, matrix (no JAX, no GPU)."""

import json
import threading

import pytest

from clvla.specialists import (
    RunnerRefused,
    assemble_matrix,
    exclusive_training_lock,
    parse_selection,
    run_serial,
    summarize_runs,
    task_seeds,
    write_summary_files,
)
from clvla.tasks import parse_manifest


@pytest.fixture
def manifest(manifest_raw):
    return parse_manifest(manifest_raw)


def test_selection_never_defaults_to_all_tasks_when_executing(manifest):
    assert len(parse_selection(manifest, None, None, execute=False)) == 10  # plan may show everything
    with pytest.raises(RunnerRefused, match="explicit"):
        parse_selection(manifest, None, None, execute=True)
    assert [t.libero_90_index for t in parse_selection(manifest, "0,3", None, True)] == [78, 56]
    assert [t.libero_90_index for t in parse_selection(manifest, None, "89,78", True)] == [89, 78]
    with pytest.raises(RunnerRefused):
        parse_selection(manifest, "0", "78", True)
    with pytest.raises(KeyError):
        parse_selection(manifest, None, "1", True)  # not in the manifest


def test_task_seeds_recorded_per_task(manifest):
    s = task_seeds(7, manifest.tasks)
    assert set(s) == {t.libero_90_index for t in manifest.tasks} and len(set(s.values())) == 10
    assert s == task_seeds(7, list(reversed(manifest.tasks)))  # independent of order


def test_runs_are_strictly_serial_and_guarded(manifest, tmp_path):
    tasks = parse_selection(manifest, "0,1,2", None, True)
    active, max_active, order, guards = [0], [0], [], []
    lock = threading.Lock()

    def run_one(t):
        with lock:
            active[0] += 1
            max_active[0] = max(max_active[0], active[0])
        order.append(t.libero_90_index)
        assert (tmp_path / ".training.lock").exists()
        with lock:
            active[0] -= 1
        return {"run_dir": f"/r/{t.libero_90_index}"}

    res = run_serial(tasks, run_one, guard=lambda: guards.append(1), lock_root=tmp_path)
    assert order == [78, 84, 16] and max_active[0] == 1 and len(guards) == 3
    assert [r["status"] for r in res] == ["completed"] * 3
    assert not (tmp_path / ".training.lock").exists()


def test_gpu_guard_refusal_starts_nothing(manifest, tmp_path):
    started = []

    def guard():
        raise RunnerRefused("GPU busy")

    with pytest.raises(RunnerRefused, match="GPU busy"):
        run_serial(parse_selection(manifest, "0", None, True), lambda t: started.append(t), guard=guard, lock_root=tmp_path)
    assert started == [] and not (tmp_path / ".training.lock").exists()


def test_failure_stops_the_queue(manifest, tmp_path):
    def run_one(t):
        if t.libero_90_index == 84:
            raise RuntimeError("boom")
        return {}

    res = run_serial(parse_selection(manifest, "0,1,2", None, True), run_one, guard=lambda: None, lock_root=tmp_path)
    assert [r["status"] for r in res] == ["completed", "failed"] and "boom" in res[1]["error"]


def test_second_runner_is_refused_while_one_holds_the_lock(tmp_path):
    with exclusive_training_lock(tmp_path):
        with pytest.raises(RunnerRefused, match="serial"):
            with exclusive_training_lock(tmp_path):
                pass


def _fake_run(root, idx, status="completed", vals=(2.0, 1.5)):
    d = root / f"20260101T00000{idx % 10}Z_spec-v1-t{idx:03d}"
    d.mkdir(parents=True)
    (d / "run_metadata.json").write_text(json.dumps({"task": {"libero_90_index": idx, "name": f"T{idx}"}, "seed": idx}))
    (d / "summary.json").write_text(json.dumps({
        "status": status, "num_steps": 100, "final_train_loss": 0.3, "final_val_loss": vals[-1], "wall_time_s": 60,
        "mean_step_time_s": 0.5, "init": {"trainable_params": 1000}, "checkpoints": [{"step": 100, "dir": f"{d}/ck"}],
        "trained_checkpoint_fingerprint": "f" * 64, "memory": {"peak_bytes_in_use": 9}}))
    (d / "history.jsonl").write_text("\n".join(json.dumps({"step": i, "val_loss": v}) for i, v in enumerate(vals)))
    return d


def test_summary_rows_and_files(tmp_path):
    dirs = [_fake_run(tmp_path, 78), _fake_run(tmp_path, 56, vals=(1.0, 0.8, 0.9))]
    rows = summarize_runs(dirs, control_indices=[56])
    by = {r["libero_90_index"]: r for r in rows}
    assert by[56]["control_task"] is True and by[78]["control_task"] is False
    assert by[56]["best_val_loss"] == 0.8 and by[78]["trainable_params"] == 1000
    assert by[78]["checkpoint_fingerprint_sha256"] == "f" * 64
    write_summary_files(rows, tmp_path / "out")
    assert (tmp_path / "out/specialists_summary.csv").read_text().startswith("libero_90_index,task_name,run_dir,status")
    assert len(json.loads((tmp_path / "out/specialists_summary.json").read_text())) == 2


def test_expert_by_task_matrix():
    recs = [{"expert": 78, "task": 78, "successes": 3, "episodes": 5, "infra_errors": 0},
            {"expert": 78, "task": 84, "successes": 0, "episodes": 5, "infra_errors": 0},
            {"expert": 84, "task": 84, "successes": 2, "episodes": 4, "infra_errors": 1}]
    m = assemble_matrix(recs, [78, 84], [78, 84])
    assert m["success_rate"] == [[0.6, 0.0], [None, 0.5]]  # missing cell stays None (never imputed)
    assert m["diagonal"] == [0.6, 0.5] and m["infra_errors"] == 1 and m["counts"][1][0] is None
    with pytest.raises(ValueError, match="duplicate"):
        assemble_matrix(recs + [recs[0]], [78, 84], [78, 84])
