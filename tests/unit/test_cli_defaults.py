"""CLI safety defaults: dry run unless --execute; explicit selections; approved budgets; split guard; v1.1 config."""

import copy

import pytest
import yaml

from clvla.config import ConfigError, parse_experiment_config
from clvla.evaluation import check_split_allowed
from conftest import load_script


def test_rerender_defaults_to_plan_and_writes_nothing(project_env, tmp_path, capsys):
    mod = load_script("rerender_demos")
    assert mod.main(["--positions", "0"]) == 0
    out = capsys.readouterr().out
    assert "[plan]" in out and "nothing written" in out
    assert "STUDY_SCENE2_pick_up_the_book_and_place_it_in_the_front_compartment_of_the_caddy" in out  # position 0 = t78
    assert not (tmp_path / "processed").exists()


def test_rerender_execute_requires_explicit_selection(project_env):
    mod = load_script("rerender_demos")
    with pytest.raises(SystemExit, match="explicit selection"):
        mod.main(["--execute"])


def test_convert_defaults_to_plan(project_env, tmp_path, capsys):
    mod = load_script("convert_to_lerobot")
    assert mod.main(["--indices", "78"]) == 0
    assert "NOT RENDERED YET" in capsys.readouterr().out
    assert not (tmp_path / "lerobot").exists()


def test_specialists_execute_requires_selection(project_env):
    mod = load_script("train_specialists")
    with pytest.raises(SystemExit, match="explicit"):
        mod.main(["--execute"])


def test_specialists_execute_requires_approved_budget(project_env, monkeypatch, capsys):
    mod = load_script("train_specialists")
    calls = []

    def fake_prepare(**kw):
        calls.append(kw["libero_90_index"])
        return {"seed": mod.task_seeds(7, [type("T", (), {"libero_90_index": kw["libero_90_index"]})])[kw["libero_90_index"]],
                "task": {"libero_90_index": kw["libero_90_index"]}}

    monkeypatch.setattr(mod, "prepare_task_run", fake_prepare)
    monkeypatch.setattr(mod, "describe", lambda plan: "  (plan)")
    monkeypatch.setattr(mod, "execute_task_run", lambda *a, **k: pytest.fail("must not train"))
    with pytest.raises(SystemExit, match="budget_approved"):
        mod.main(["--indices", "78", "--execute"])
    assert calls == [78]
    assert mod.main(["--indices", "78"]) == 0  # plan is allowed without approval


def test_smoke_defaults_to_plan(project_env, monkeypatch):
    mod = load_script("train_smoke")
    monkeypatch.setattr(mod, "prepare_task_run", lambda **kw: {"plan": True})
    monkeypatch.setattr(mod, "describe", lambda plan: "  (plan)")
    monkeypatch.setattr(mod, "execute_task_run", lambda *a, **k: pytest.fail("must not train"))
    assert mod.main([]) == 0
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(SystemExit, match="CLVLA_USE_GPU"):
        mod.main(["--execute"])


@pytest.mark.parametrize("kind,split,conf,ok", [
    ("baseline", "test", False, True), ("smoke", "selection", False, True), ("smoke", "test", False, False),
    ("finetune", "test", True, False), ("specialist_eval", "test", False, False), ("specialist_eval", "test", True, True),
])
def test_evaluation_split_guard(kind, split, conf, ok):
    if ok:
        check_split_allowed(kind, split, conf)
    else:
        with pytest.raises(ValueError):
            check_split_allowed(kind, split, conf)


def test_proposed_v11_protocol_parses_but_is_not_frozen(repo_root, fake_env):
    raw = yaml.safe_load((repo_root / "configs/proposed/experiment_v1.1.yaml").read_text())
    with pytest.raises(ConfigError, match="frozen protocol"):
        parse_experiment_config(raw, env=fake_env)
    cfg = parse_experiment_config(raw, env=fake_env, check_frozen=False)
    assert cfg.splits.selection_init_states == list(range(10))
    assert cfg.splits.development_init_states == list(range(10, 20))
    assert cfg.splits.test_init_states == list(range(20, 50))
    bad = copy.deepcopy(raw)
    bad["protocol"]["splits"]["development_init_states"] = {"start": 5, "stop": 15}
    with pytest.raises(ConfigError, match="development"):
        parse_experiment_config(bad, env=fake_env, check_frozen=False)


def test_frozen_v10_protocol_unchanged(experiment_raw, fake_env):
    cfg = parse_experiment_config(experiment_raw, env=fake_env)
    assert cfg.protocol_version == "1.0.0" and cfg.splits.development_init_states is None
    assert cfg.protocol_sha256.startswith("a3d55cd8")
