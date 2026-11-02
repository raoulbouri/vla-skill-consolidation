"""Training loop gates, logging, checkpoint metadata, resume and overwrite rules with a fake backend (no JAX)."""

import copy
import json

import pytest
import yaml

from clvla.config import ConfigError
from clvla.experiment import make_run_dir
from clvla.training_core import (
    GateFailure,
    RunPaths,
    TrainSpec,
    open_run,
    read_history,
    run_loop,
    run_signature,
    task_seed,
)


@pytest.fixture
def smoke_raw(repo_root):
    return yaml.safe_load((repo_root / "configs/training/smoke_task78.yaml").read_text())


class FakeBackend:
    def __init__(self, losses=None, grads=None, val=1.0, trainable_changed=True, frozen_unchanged=True, start=0):
        self.losses, self.grads, self.val = losses, grads, val
        self.tc, self.fu, self._start = trainable_changed, frozen_unchanged, start
        self.saved, self.validated, self.steps = [], 0, []

    def init(self):
        return {"trainable_params": 123, "total_params": 1000}

    def start_step(self):
        return self._start

    def step(self, step):
        self.steps.append(step)
        loss = self.losses[step] if self.losses else 1.0 / (step + 1)
        grad = self.grads[step] if self.grads else 0.5
        return {"loss": loss, "grad_norm": grad, "param_norm": 10.0, "lr": 5e-5}

    def validate(self):
        self.validated += 1
        return {"val_loss": self.val, "val_batches": 4}

    def save(self, n):
        self.saved.append(n)
        return {"step": n, "dir": f"/ck/{n}"}

    def probe(self):
        return {"trainable_changed": {"a": self.tc}, "frozen_unchanged": {"b": self.fu}}

    def memory(self):
        return {"peak_bytes_in_use": 42}

    def finish(self):
        return {}


def spec_with(raw, **train):
    raw = copy.deepcopy(raw)
    raw["train"].update(train)
    return TrainSpec.from_dict(raw)


def paths(tmp_path):
    p = RunPaths(tmp_path / "run")
    p.root.mkdir()
    return p


def test_smoke_spec_is_valid_and_bounded(smoke_raw):
    s = TrainSpec.from_dict(smoke_raw)
    assert (s.num_steps, s.batch_size, s.save_interval, s.val_interval) == (100, 8, 100, 50)
    assert s.paligemma_variant == "gemma_2b_lora" and s.action_expert_variant == "gemma_300m_lora"
    assert s.lora_b_init == "zeros" and s.kind == "finetune"


@pytest.mark.parametrize("section,key,value,msg", [
    ("model", "mode", "full", "full fine-tuning"),
    ("model", "base_openpi_config", "pi05_base", "pi05_libero"),
    ("model", "paligemma_variant", "gemma_2b", "unsupported LoRA"),
    ("model", "lora_b_init", "ones", "lora_b_init"),
    ("train", "num_steps", 0, "num_steps"),
    ("train", "save_interval", 1000, "save_interval"),
    ("runtime", "xla_mem_fraction", 1.0, "xla_mem_fraction"),
])
def test_spec_rejects_invalid_settings(smoke_raw, section, key, value, msg):
    raw = copy.deepcopy(smoke_raw)
    raw[section][key] = value
    with pytest.raises(ConfigError, match=msg):
        TrainSpec.from_dict(raw)


def test_loop_logs_history_validation_and_bounded_checkpoints(smoke_raw, tmp_path):
    spec = spec_with(smoke_raw, num_steps=10, val_interval=5, save_interval=10)
    p, be = paths(tmp_path), FakeBackend()
    s = run_loop(be, spec, p)
    assert s["status"] == "completed" and be.steps == list(range(10))
    assert be.validated == 2 and be.saved == [10]
    hist = read_history(p.history)
    assert [h["completed_steps"] for h in hist] == list(range(1, 11))
    assert {"loss", "grad_norm", "param_norm", "lr", "step_time_s"} <= set(hist[0])
    assert "val_loss" in hist[4] and "val_loss" in hist[9] and "val_loss" not in hist[5]
    assert hist[9]["checkpoint"] == {"step": 10, "dir": "/ck/10"}
    saved = json.loads(p.summary.read_text())
    assert saved["checkpoints"] == [{"step": 10, "dir": "/ck/10"}] and saved["memory"]["peak_bytes_in_use"] == 42
    assert saved["init"]["trainable_params"] == 123 and saved["final_val_loss"] == 1.0


@pytest.mark.parametrize("losses,grads,msg", [
    ([1.0, float("nan")] + [1.0] * 8, None, "non-finite loss"),
    ([1.0, float("inf")] + [1.0] * 8, None, "non-finite loss"),
    (None, [0.1, 0.1, float("nan")] + [0.1] * 7, "non-finite grad_norm"),
    ([1.0, 1e9] + [1.0] * 8, None, "max_loss"),
])
def test_non_finite_or_exploding_training_aborts(smoke_raw, tmp_path, losses, grads, msg):
    p = paths(tmp_path)
    with pytest.raises(GateFailure, match=msg):
        run_loop(FakeBackend(losses=losses, grads=grads), spec_with(smoke_raw, num_steps=10, val_interval=5, save_interval=10), p)
    s = json.loads(p.summary.read_text())
    assert s["status"] == "failed" and msg.split()[0] in s["error"]


def test_non_finite_validation_aborts(smoke_raw, tmp_path):
    with pytest.raises(GateFailure, match="validation"):
        run_loop(FakeBackend(val=float("nan")), spec_with(smoke_raw, num_steps=10, val_interval=5, save_interval=10),
                 paths(tmp_path))


@pytest.mark.parametrize("tc,fu,msg", [(False, True, "did not change"), (True, False, "frozen parameters changed")])
def test_parameter_update_gate(smoke_raw, tmp_path, tc, fu, msg):
    with pytest.raises(GateFailure, match=msg):
        run_loop(FakeBackend(trainable_changed=tc, frozen_unchanged=fu),
                 spec_with(smoke_raw, num_steps=10, val_interval=5, save_interval=10), paths(tmp_path))


def test_resume_continues_from_start_step(smoke_raw, tmp_path):
    be = FakeBackend(start=6)
    s = run_loop(be, spec_with(smoke_raw, num_steps=10, val_interval=5, save_interval=10), paths(tmp_path))
    assert be.steps == [6, 7, 8, 9] and s["start_step"] == 6


def test_signature_changes_with_inputs_but_not_runtime(smoke_raw):
    spec = TrainSpec.from_dict(smoke_raw)
    ident = {"base_checkpoint_fingerprint": "cd22", "train_dataset": "aa", "seed": 1}
    sig = run_signature(spec, ident)
    assert run_signature(spec, dict(ident)) == sig
    assert run_signature(spec, dict(ident, train_dataset="bb")) != sig
    assert run_signature(spec_with(smoke_raw, num_steps=50, save_interval=50), ident) != sig
    raw = copy.deepcopy(smoke_raw)
    raw["runtime"]["xla_mem_fraction"] = 0.5
    assert run_signature(TrainSpec.from_dict(raw), ident) == sig


def test_new_runs_never_overwrite_and_resume_requires_matching_signature(tmp_path):
    p = open_run(lambda: make_run_dir(tmp_path, "finetune", "smoke-t078", stamp="20260101T000000Z"), "sig-A")
    assert json.loads(p.signature.read_text())["signature"] == "sig-A"
    with pytest.raises(FileExistsError):
        open_run(lambda: make_run_dir(tmp_path, "finetune", "smoke-t078", stamp="20260101T000000Z"), "sig-A")
    assert open_run(lambda: None, "sig-A", resume_dir=p.root).root == p.root
    with pytest.raises(GateFailure, match="differ"):
        open_run(lambda: None, "sig-B", resume_dir=p.root)
    p.summary.write_text(json.dumps({"status": "completed"}))
    with pytest.raises(GateFailure, match="already completed"):
        open_run(lambda: None, "sig-A", resume_dir=p.root)
    with pytest.raises(GateFailure, match="no signature"):
        open_run(lambda: None, "sig-A", resume_dir=tmp_path)


def test_task_seeds_are_deterministic_and_task_specific():
    assert task_seed(7, 78) == task_seed(7, 78)
    seeds = {task_seed(7, i) for i in (78, 84, 16, 56, 35, 42, 28, 3, 89, 63)}
    assert len(seeds) == 10
    assert task_seed(7, 78) != task_seed(8, 78)


def test_expert_mode_spec(repo_root):
    import copy

    raw = copy.deepcopy(yaml.safe_load((repo_root / "configs/training/smoke_task78.yaml").read_text()))
    raw["model"].update(mode="expert", paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
                        freeze_vision_encoder=True)
    raw["model"].pop("lora_b_init")
    spec = TrainSpec.from_dict(raw)
    assert spec.mode == "expert" and spec.lora_b_init == "none" and spec.freeze_vision_encoder
    for bad in ({"paligemma_variant": "gemma_2b_lora"}, {"freeze_vision_encoder": False}, {"lora_b_init": "zeros"},
                {"mode": "full"}):
        r = copy.deepcopy(raw)
        r["model"].update(bad)
        with pytest.raises(ConfigError):
            TrainSpec.from_dict(r)


def test_wandb_run_names_describe_the_run(repo_root):
    import copy

    from clvla.tracking import run_name, variant_name, wandb_settings

    raw = copy.deepcopy(yaml.safe_load((repo_root / "configs/training/smoke_task78.yaml").read_text()))
    lora = TrainSpec.from_dict(raw)
    assert run_name(lora, 25, 78) == "lora-25demos-s100-t078" and wandb_settings(lora) is None
    raw["model"].update(mode="expert", paligemma_variant="gemma_2b", action_expert_variant="gemma_300m",
                        freeze_vision_encoder=True)
    raw["model"].pop("lora_b_init")
    raw["wandb"] = {"entity": "e", "project": "p"}
    exp = TrainSpec.from_dict(raw)
    assert variant_name(exp, 10) == "expert300m-10demos-s100" and wandb_settings(exp) == {"entity": "e", "project": "p"}
