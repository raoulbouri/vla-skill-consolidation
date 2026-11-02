"""Framework-free training bookkeeping: spec validation, resume signatures, run directories, the step loop with safety
gates, history and summaries. The OpenPI/JAX backend lives in clvla.openpi_training; unit tests drive this loop with a
fake backend. Python-3.8 compatible; no JAX/OpenPI imports here.

Gates enforced by `run_loop` (a violated gate stops the run and is written to summary.json):
  * loss and gradient norm finite at every step (and loss <= max_loss)
  * validation loss finite whenever computed (validation demos only; see clvla.lerobot_convert)
  * after training: every probed trainable leaf changed, every probed frozen leaf unchanged
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol

import yaml

from clvla.config import ConfigError, canonical_sha256, derive_seed

LORA_VARIANTS = {"paligemma": ("gemma_2b_lora",), "action_expert": ("gemma_300m_lora",)}
LORA_B_INIT = ("zeros", "openpi_default")
# mode "expert": no LoRA; Gemma 2B + SigLIP frozen, the 300M action expert (+ action projections, time MLP) fully trained.
EXPERT_VARIANTS = {"paligemma": "gemma_2b", "action_expert": "gemma_300m"}


class GateFailure(RuntimeError):
    """A training safety gate failed; the run must not be used."""


def _req(d: Dict[str, Any], key: str, path: str, kind: Any = None) -> Any:
    if not isinstance(d, dict) or key not in d:
        raise ConfigError(f"missing required key: {path}.{key}" if path else f"missing required key: {key}")
    v = d[key]
    if kind is not None and not (isinstance(v, kind) and not (kind is int and isinstance(v, bool))):
        raise ConfigError(f"{path}.{key}: expected {kind}, got {v!r}")
    return v


@dataclasses.dataclass(frozen=True)
class TrainSpec:
    """Validated contents of a configs/training/*.yaml file (one task's run settings)."""

    name: str
    kind: str
    raw: Dict[str, Any]
    paligemma_variant: str
    action_expert_variant: str
    lora_b_init: str
    freeze_vision_encoder: bool
    num_steps: int
    batch_size: int
    num_workers: int
    log_interval: int
    val_interval: int
    val_max_batches: Optional[int]
    save_interval: int
    peak_lr: float
    warmup_steps: int
    decay_steps: int
    decay_lr: float
    clip_gradient_norm: float
    max_loss: float
    xla_mem_fraction: float

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "TrainSpec":
        raw = copy.deepcopy(raw)
        if _req(raw, "schema_version", "", int) != 1:
            raise ConfigError("schema_version must be 1")
        m, o, t, g, r = (_req(raw, k, "") for k in ("model", "optim", "train", "gates", "runtime"))
        if _req(m, "base_openpi_config", "model", str) != "pi05_libero":
            raise ConfigError("model.base_openpi_config must be pi05_libero (the official checkpoint's config)")
        mode = _req(m, "mode", "model", str)
        if mode == "expert":
            pv, av = _req(m, "paligemma_variant", "model", str), _req(m, "action_expert_variant", "model", str)
            if (pv, av) != (EXPERT_VARIANTS["paligemma"], EXPERT_VARIANTS["action_expert"]):
                raise ConfigError(f"mode expert needs paligemma_variant {EXPERT_VARIANTS['paligemma']!r} and "
                                  f"action_expert_variant {EXPERT_VARIANTS['action_expert']!r} (no LoRA)")
            if not _req(m, "freeze_vision_encoder", "model", bool):
                raise ConfigError("mode expert freezes the whole VLM: freeze_vision_encoder must be true")
            if m.get("lora_b_init", "none") != "none":
                raise ConfigError("mode expert has no LoRA: lora_b_init must be absent or 'none'")
            m["lora_b_init"] = "none"
        elif mode != "lora":
            raise ConfigError("model.mode must be 'lora' or 'expert': full fine-tuning of the whole model exceeds "
                              "the GPU budget (OpenPI README: > 70 GB) and is not silently allowed")
        pv, av = _req(m, "paligemma_variant", "model", str), _req(m, "action_expert_variant", "model", str)
        lbi = _req(m, "lora_b_init", "model", str)
        if mode == "lora":
            if pv not in LORA_VARIANTS["paligemma"] or av not in LORA_VARIANTS["action_expert"]:
                raise ConfigError(f"unsupported LoRA variants {pv!r}/{av!r}; supported: {LORA_VARIANTS}")
            if lbi not in LORA_B_INIT:
                raise ConfigError(f"model.lora_b_init must be one of {LORA_B_INIT}")
        spec = cls(
            name=_req(raw, "name", "", str), kind=_req(raw, "kind", "", str), raw=raw,
            paligemma_variant=pv, action_expert_variant=av, lora_b_init=lbi,
            freeze_vision_encoder=bool(_req(m, "freeze_vision_encoder", "model", bool)),
            num_steps=_req(t, "num_steps", "train", int), batch_size=_req(t, "batch_size", "train", int),
            num_workers=_req(t, "num_workers", "train", int), log_interval=_req(t, "log_interval", "train", int),
            val_interval=_req(t, "val_interval", "train", int), val_max_batches=t.get("val_max_batches"),
            save_interval=_req(t, "save_interval", "train", int),
            peak_lr=float(_req(o, "peak_lr", "optim")), warmup_steps=_req(o, "warmup_steps", "optim", int),
            decay_steps=_req(o, "decay_steps", "optim", int), decay_lr=float(_req(o, "decay_lr", "optim")),
            clip_gradient_norm=float(_req(o, "clip_gradient_norm", "optim")),
            max_loss=float(_req(g, "max_loss", "gates")), xla_mem_fraction=float(_req(r, "xla_mem_fraction", "runtime")),
        )
        if spec.kind not in ("finetune", "specialists"):
            raise ConfigError(f"kind must be finetune or specialists, got {spec.kind!r}")
        for f in ("num_steps", "batch_size", "log_interval", "val_interval", "save_interval"):
            if getattr(spec, f) < 1:
                raise ConfigError(f"train.{f} must be >= 1")
        if spec.num_workers < 0 or not (0 < spec.peak_lr < 1) or not (0 < spec.xla_mem_fraction <= 0.95):
            raise ConfigError("invalid num_workers / peak_lr / xla_mem_fraction")
        if spec.val_max_batches is not None and (not isinstance(spec.val_max_batches, int) or spec.val_max_batches < 1):
            raise ConfigError("train.val_max_batches must be null or a positive int")
        if spec.save_interval > spec.num_steps:
            raise ConfigError("train.save_interval > num_steps: no intermediate checkpoint; set it to num_steps")
        return spec

    @property
    def mode(self) -> str:
        return self.raw["model"]["mode"]

    @classmethod
    def from_yaml(cls, path: os.PathLike) -> "TrainSpec":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def with_overrides(self, **train_overrides: Any) -> "TrainSpec":
        raw = copy.deepcopy(self.raw)
        for k, v in train_overrides.items():
            if v is not None:
                raw["train"][k] = v
        return TrainSpec.from_dict(raw)


def task_seed(global_seed: int, libero_90_index: int) -> int:
    """Deterministic per-task training seed (same for smoke and specialists on the same task)."""
    return derive_seed(global_seed, "specialist", int(libero_90_index))


def run_signature(spec: TrainSpec, identity: Dict[str, Any]) -> str:
    """Everything that must match to resume: model/optim/train/gates settings + input identities.

    `identity` holds source checkpoint fingerprint, manifest hash, dataset fingerprints, task, seed, code revision.
    Runtime settings (memory fraction) are excluded."""
    body = {k: v for k, v in spec.raw.items() if k not in ("runtime",)}
    return canonical_sha256({"spec": body, "identity": identity})


# ---------------------------------------------------------------------------------------------------------------------
# Run directory


@dataclasses.dataclass(frozen=True)
class RunPaths:
    root: Path

    @property
    def checkpoints(self) -> Path:
        return self.root / "checkpoints"

    @property
    def history(self) -> Path:
        return self.root / "history.jsonl"

    @property
    def metadata(self) -> Path:
        return self.root / "run_metadata.json"

    @property
    def signature(self) -> Path:
        return self.root / "signature.json"

    @property
    def summary(self) -> Path:
        return self.root / "summary.json"

    @property
    def spec(self) -> Path:
        return self.root / "train_spec.yaml"


def open_run(make_dir: Callable[[], Path], signature: str, resume_dir: Optional[os.PathLike] = None) -> RunPaths:
    """New run: a fresh directory from `make_dir` (which must refuse existing paths). Resume: an existing directory
    whose recorded signature equals `signature`, and that is not already complete."""
    if resume_dir is None:
        paths = RunPaths(Path(make_dir()))
        paths.signature.write_text(json.dumps({"signature": signature}, indent=1))
        return paths
    paths = RunPaths(Path(resume_dir))
    if not paths.signature.is_file():
        raise GateFailure(f"cannot resume {paths.root}: no signature.json")
    old = json.loads(paths.signature.read_text())["signature"]
    if old != signature:
        raise GateFailure(f"cannot resume {paths.root}: model config, source checkpoint, dataset hashes or run "
                          f"settings differ (signature {old[:12]} != {signature[:12]})")
    if paths.summary.is_file() and json.loads(paths.summary.read_text()).get("status") == "completed":
        raise GateFailure(f"{paths.root} is already completed; start a new run instead of resuming")
    return paths


# ---------------------------------------------------------------------------------------------------------------------
# Loop


class Backend(Protocol):
    def init(self) -> Dict[str, Any]: ...  # counts, gate results; must set start_step
    def start_step(self) -> int: ...
    def step(self, step: int) -> Dict[str, float]: ...  # loss, grad_norm, param_norm, lr
    def validate(self) -> Dict[str, float]: ...  # val_loss, val_batches
    def save(self, completed_steps: int) -> Dict[str, Any]: ...  # checkpoint info
    def probe(self) -> Dict[str, Any]: ...  # trainable_changed: [...], frozen_unchanged: [...], deltas
    def memory(self) -> Dict[str, Any]: ...
    def finish(self) -> Dict[str, Any]: ...


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def run_loop(backend: Backend, spec: TrainSpec, paths: RunPaths, clock: Callable[[], float] = time.perf_counter,
             on_record: Optional[Callable[[Dict[str, Any]], None]] = None) -> Dict[str, Any]:
    """Run training with gates; always writes summary.json (status completed | failed)."""
    t_start = clock()
    summary: Dict[str, Any] = {"status": "running", "name": spec.name, "num_steps": spec.num_steps}
    history = paths.history.open("a")
    try:
        summary["init"] = backend.init()
        start = backend.start_step()
        summary["start_step"] = start
        if start >= spec.num_steps:
            raise GateFailure(f"start step {start} >= num_steps {spec.num_steps}")
        step_times: List[float] = []
        last_val: Optional[float] = None
        checkpoints = []
        for step in range(start, spec.num_steps):
            t0 = clock()
            info = {k: float(v) for k, v in backend.step(step).items()}
            dt = clock() - t0
            step_times.append(dt)
            for k in ("loss", "grad_norm"):
                if not _finite(info.get(k)):
                    raise GateFailure(f"non-finite {k} at step {step}: {info.get(k)}")
            if info["loss"] > spec.max_loss:
                raise GateFailure(f"loss {info['loss']} > max_loss {spec.max_loss} at step {step}")
            rec = {"step": step, "completed_steps": step + 1, "step_time_s": round(dt, 4), **info}
            last = step == spec.num_steps - 1
            if (step + 1) % spec.val_interval == 0 or last:
                val = {k: float(v) for k, v in backend.validate().items()}
                if not _finite(val.get("val_loss")):
                    raise GateFailure(f"non-finite validation loss at step {step}: {val}")
                last_val = val["val_loss"]
                rec.update(val)
            if (step + 1) % spec.save_interval == 0 or last:
                ck = backend.save(step + 1)
                checkpoints.append(ck)
                rec["checkpoint"] = ck
            history.write(json.dumps(rec) + "\n")
            history.flush()
            if on_record is not None:
                on_record(rec)
        probe = backend.probe()
        summary["param_probe"] = probe
        if not probe.get("trainable_changed") or not all(probe["trainable_changed"].values()):
            raise GateFailure(f"trainable parameters did not change: {probe}")
        if not probe.get("frozen_unchanged") or not all(probe["frozen_unchanged"].values()):
            raise GateFailure(f"frozen parameters changed: {probe}")
        summary.update({
            "status": "completed",
            "final_train_loss": info["loss"],
            "final_val_loss": last_val,
            "checkpoints": checkpoints,
            "mean_step_time_s": sum(step_times[1:]) / max(1, len(step_times) - 1) if len(step_times) > 1 else None,
            "first_step_time_s": step_times[0] if step_times else None,
        })
    except Exception as e:  # noqa: BLE001 - recorded, then re-raised
        summary.update({"status": "failed", "error": f"{type(e).__name__}: {e}"})
        raise
    finally:
        history.close()
        try:
            summary["memory"] = backend.memory()
            summary["finish"] = backend.finish()
        except Exception as e:  # noqa: BLE001
            summary["finish_error"] = f"{type(e).__name__}: {e}"
        summary["wall_time_s"] = round(clock() - t_start, 2)
        paths.summary.write_text(json.dumps(summary, indent=1, default=str))
    return summary


def read_history(path: os.PathLike) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
