"""OpenPI/JAX backend for clvla.training_core.run_loop (policy env only; nothing here runs at import time).

Reuses the pinned OpenPI training stack instead of reimplementing it:
  * openpi.training.config.TrainConfig / LeRobotLiberoDataConfig / AssetsConfig, Pi0Config LoRA variants and
    Pi0Config.get_freeze_filter (the stock LoRA recipe of `pi0_libero_low_mem_finetune`, applied to pi05)
  * init_train_state / train_step from third_party/openpi/scripts/train.py (imported from the file, unmodified)
  * openpi.training.data_loader.create_data_loader and openpi.training.checkpoints save/restore
Project-local additions (no upstream file is modified):
  * StrictCheckpointWeightLoader: OpenPI's CheckpointWeightLoader + refusal of dropped/missing keys and shape
    mismatches, a dtype-cast report, and optional zero init of LoRA B matrices (see docs/NEXT_EXPERIMENTS.md)
  * a validation pass with the model's own compute_loss(train=False) on the separate validation repo
  * parameter probes, norm-stat identity checks, memory/timing records

Checkpoint initialisation (the "checkpoint trap"): the upstream `pi05_libero` TrainConfig initialises from
gs://openpi-assets/checkpoints/pi05_base/params. Here the weight loader points at the LOCAL official pi05_libero
inference checkpoint's params/ directory (a "released checkpoint", which CheckpointWeightLoader documents as
supported), and norm stats come from that checkpoint's assets/ via AssetsConfig (OpenPI's documented mechanism).
"""

from __future__ import annotations

import dataclasses
import functools
import importlib.util
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from clvla.training_core import GateFailure, TrainSpec

LORA_B_RE = re.compile(r".*lora_b$")
LORA_ANY_RE = ".*lora.*"
ASSET_ID = "physical-intelligence/libero"
# Probed leaves (flattened pure-dict paths). Trainable: LoRA B (zeros at init -> must move) and the action head.
TRAINABLE_PROBES = ("PaliGemma/llm/layers/attn/q_einsum/lora_b", "action_out_proj/kernel")
FROZEN_PROBES = ("PaliGemma/llm/final_norm/scale", "PaliGemma/llm/layers/attn/q_einsum/w")
# mode "expert": an action-expert weight (suffix _1) must move; Gemma 2B and SigLIP weights must not.
EXPERT_TRAINABLE_PROBES = ("PaliGemma/llm/layers/attn/q_einsum_1/w", "action_out_proj/kernel")
EXPERT_FROZEN_PROBES = FROZEN_PROBES + ("PaliGemma/img/embedding/kernel",)


def probes_for(spec: TrainSpec):
    return (EXPERT_TRAINABLE_PROBES, EXPERT_FROZEN_PROBES) if spec.mode == "expert" else (TRAINABLE_PROBES, FROZEN_PROBES)


def load_openpi_train_module(openpi_dir: Path):
    """Import third_party/openpi/scripts/train.py as a module (unmodified) to reuse init_train_state / train_step."""
    path = Path(openpi_dir) / "scripts" / "train.py"
    spec = importlib.util.spec_from_file_location("clvla_openpi_train_script", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_strict_loader_cls():
    from flax import traverse_util

    import openpi.models.model as _model
    import openpi.shared.download as download
    import openpi.training.weight_loaders as wl

    @dataclasses.dataclass(frozen=True)
    class StrictCheckpointWeightLoader(wl.CheckpointWeightLoader):
        lora_b_init: str = "zeros"
        report_path: Optional[str] = None

        def load(self, params):
            loaded = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
            flat_ref = traverse_util.flatten_dict(params, sep="/")
            flat_loaded = traverse_util.flatten_dict(loaded, sep="/")
            lora_re = re.compile(LORA_ANY_RE)
            dropped = sorted(set(flat_loaded) - set(flat_ref))
            missing = sorted(k for k in flat_ref if k not in flat_loaded and not lora_re.fullmatch(k))
            shape_bad = sorted(k for k in flat_loaded if k in flat_ref
                               and tuple(flat_loaded[k].shape) != tuple(flat_ref[k].shape))
            if dropped or missing or shape_bad:
                raise GateFailure(
                    "checkpoint -> model parameter mapping is not exact: "
                    f"{len(dropped)} checkpoint keys absent from the model {dropped[:5]}, "
                    f"{len(missing)} non-LoRA model keys absent from the checkpoint {missing[:5]}, "
                    f"{len(shape_bad)} shape mismatches {shape_bad[:5]}")
            casts: Dict[str, int] = {}
            for k, v in flat_loaded.items():
                key = f"{v.dtype}->{flat_ref[k].dtype}"
                casts[key] = casts.get(key, 0) + 1
            lora_keys = sorted(k for k in flat_ref if lora_re.fullmatch(k))
            merged = wl._merge_params(loaded, params, missing_regex=LORA_ANY_RE)
            zeroed = []
            if self.lora_b_init == "zeros":
                flat = traverse_util.flatten_dict(merged, sep="/")
                for k in lora_keys:
                    if LORA_B_RE.fullmatch(k):
                        flat[k] = np.zeros(flat_ref[k].shape, dtype=flat_ref[k].dtype)
                        zeroed.append(k)
                merged = traverse_util.unflatten_dict(flat, sep="/")
            elif self.lora_b_init == "none":
                if lora_keys:
                    raise GateFailure(f"lora_b_init none but the model has LoRA leaves: {lora_keys[:3]}")
            elif self.lora_b_init != "openpi_default":
                raise GateFailure(f"unknown lora_b_init {self.lora_b_init!r}")
            if self.report_path:
                Path(self.report_path).write_text(json.dumps({
                    "params_path": str(self.params_path), "checkpoint_leaves": len(flat_loaded),
                    "model_leaves": len(flat_ref), "lora_leaves_from_init": len(lora_keys),
                    "lora_b_init": self.lora_b_init, "lora_b_zeroed": zeroed, "dtype_casts": casts,
                    "dropped": dropped, "missing_non_lora": missing, "shape_mismatches": shape_bad}, indent=1))
            return merged

    return StrictCheckpointWeightLoader


def build_train_config(spec: TrainSpec, *, base_ckpt_dir: str, train_repo: str, seed: int, checkpoint_base_dir: str,
                       report_path: Optional[str] = None, val_repo: Optional[str] = None):
    """TrainConfig derived from the pinned pi05_libero config with only the documented deviations."""
    import flax.nnx as nnx

    import openpi.models.pi0_config as pi0_config
    import openpi.shared.nnx_utils as nnx_utils
    import openpi.training.config as _config
    import openpi.training.optimizer as _optimizer

    base = _config.get_config("pi05_libero")
    bm = base.model
    model = pi0_config.Pi0Config(pi05=True, action_horizon=bm.action_horizon, discrete_state_input=bm.discrete_state_input,
                                 paligemma_variant=spec.paligemma_variant, action_expert_variant=spec.action_expert_variant)
    if (model.action_dim, model.max_token_len, model.action_horizon) != (bm.action_dim, bm.max_token_len, bm.action_horizon):
        raise GateFailure("LoRA model config diverges from pi05_libero in action_dim/max_token_len/action_horizon")
    if spec.mode == "expert":
        # Freeze Gemma 2B (every llm leaf that is not an action-expert `_1` leaf) and SigLIP; train the expert, the
        # action in/out projections and the time MLP. The prefix then has no trainable dependency, so autodiff only
        # backpropagates through the expert (equivalent to training on a stop-gradient prefix KV cache).
        freeze = nnx.All(nnx.Param, nnx.Any(
            nnx_utils.PathRegex(".*PaliGemma/img.*"),
            nnx.All(nnx_utils.PathRegex(".*llm.*"), nnx.Not(nnx_utils.PathRegex(".*llm.*_1.*")))))
    else:
        freeze = model.get_freeze_filter()
    if spec.mode != "expert" and spec.freeze_vision_encoder:
        freeze = nnx.Any(freeze, nnx.All(nnx.Param, nnx_utils.PathRegex(".*PaliGemma/img.*")))
    if base.data.extra_delta_transform:
        raise GateFailure("pinned pi05_libero unexpectedly uses extra_delta_transform")
    data = dataclasses.replace(
        base.data, repo_id=val_repo or train_repo,
        assets=_config.AssetsConfig(assets_dir=str(Path(base_ckpt_dir) / "assets"), asset_id=ASSET_ID),
        extra_delta_transform=False)
    loader_cls = make_strict_loader_cls()
    return dataclasses.replace(
        base,
        name=f"clvla_{spec.name}",
        exp_name="openpi",
        model=model,
        weight_loader=loader_cls(params_path=str(Path(base_ckpt_dir) / "params"), lora_b_init=spec.lora_b_init,
                                 report_path=report_path),
        freeze_filter=freeze,
        ema_decay=None,  # as OpenPI's LoRA configs
        data=data,
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=spec.warmup_steps, peak_lr=spec.peak_lr,
                                                   decay_steps=spec.decay_steps, decay_lr=spec.decay_lr),
        optimizer=_optimizer.AdamW(clip_gradient_norm=spec.clip_gradient_norm),
        seed=seed, batch_size=spec.batch_size, num_workers=spec.num_workers, num_train_steps=spec.num_steps,
        log_interval=spec.log_interval, save_interval=spec.save_interval, keep_period=None,
        checkpoint_base_dir=checkpoint_base_dir, overwrite=False, resume=False, wandb_enabled=False,
        policy_metadata={"clvla_run": spec.name, "base_checkpoint": str(base_ckpt_dir)},
    )


def norm_stats_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Compare two norm-stat mappings (NormStats objects or JSON dicts) at float32 precision."""
    def as_dict(x):
        return {f: getattr(x, f, None) if not isinstance(x, dict) else x.get(f) for f in ("mean", "std", "q01", "q99")}

    if set(a) != set(b):
        return False
    for k in a:
        da, db = as_dict(a[k]), as_dict(b[k])
        for f in da:
            if (da[f] is None) != (db[f] is None):
                return False
            if da[f] is not None and not np.array_equal(np.asarray(da[f], np.float32), np.asarray(db[f], np.float32)):
                return False
    return True


class OpenPIBackend:
    """Implements clvla.training_core.Backend with the pinned OpenPI trainer."""

    def __init__(self, spec: TrainSpec, *, run_dir: Path, openpi_dir: Path, base_ckpt_dir: str, train_repo: str,
                 val_repo: str, seed: int, resume: bool):
        self.spec, self.run_dir, self.openpi_dir = spec, Path(run_dir), Path(openpi_dir)
        self.base_ckpt_dir, self.train_repo, self.val_repo = base_ckpt_dir, train_repo, val_repo
        self.seed, self.resume = seed, resume
        self._start = 0

    # -- setup ---------------------------------------------------------------------------------------------------
    def init(self) -> Dict[str, Any]:
        import jax
        import jax.numpy as jnp
        from flax import traverse_util

        import openpi.training.checkpoints as _checkpoints
        import openpi.training.data_loader as _data_loader
        import openpi.training.sharding as sharding
        from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

        t0 = time.time()
        self.jax, self.jnp = jax, jnp
        self.train_mod = load_openpi_train_module(self.openpi_dir)
        report = self.run_dir / "weight_loading_report.json"
        self.tc = build_train_config(self.spec, base_ckpt_dir=self.base_ckpt_dir, train_repo=self.train_repo,
                                     seed=self.seed, checkpoint_base_dir=str(self.run_dir / "checkpoints"),
                                     report_path=str(report))
        self.val_tc = dataclasses.replace(self.tc, data=dataclasses.replace(self.tc.data, repo_id=self.val_repo))
        dc = self.tc.data.create(self.tc.assets_dirs, self.tc.model)
        bundled = json.loads((Path(self.base_ckpt_dir) / "assets" / ASSET_ID / "norm_stats.json").read_text())["norm_stats"]
        if dc.norm_stats is None or not norm_stats_equal(dc.norm_stats, bundled):
            raise GateFailure("training data config norm stats are missing or differ from the checkpoint's bundled stats")
        if dc.repo_id != self.train_repo or not dc.prompt_from_task or not dc.use_quantile_norm:
            raise GateFailure(f"unexpected data config: repo {dc.repo_id}, prompt_from_task {dc.prompt_from_task}")

        rng = jax.random.key(self.tc.seed)
        self.train_rng, init_rng = jax.random.split(rng)
        self.val_rng = jax.random.key(self.tc.seed + 1)
        self.mesh = sharding.make_mesh(self.tc.fsdp_devices)
        self.data_sharding = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
        self.replicated = jax.sharding.NamedSharding(self.mesh, jax.sharding.PartitionSpec())
        self.mngr, resuming = _checkpoints.initialize_checkpoint_dir(
            self.tc.checkpoint_dir, keep_period=None, overwrite=False, resume=self.resume)
        if self.resume and not resuming:
            raise GateFailure("resume requested but the checkpoint directory has no saved step")
        self.loader = _data_loader.create_data_loader(self.tc, sharding=self.data_sharding, shuffle=True)
        self.data_iter = iter(self.loader)
        n_val = LeRobotDatasetMetadata(self.val_repo).total_frames // self.tc.batch_size
        if self.spec.val_max_batches:
            n_val = min(n_val, self.spec.val_max_batches)
        if n_val < 1:
            raise GateFailure("validation set smaller than one batch")
        self.n_val = n_val
        self.val_loader = _data_loader.create_data_loader(self.val_tc, sharding=self.data_sharding, shuffle=False,
                                                          num_batches=n_val)

        state, self.state_sharding = self.train_mod.init_train_state(self.tc, init_rng, self.mesh, resume=resuming)
        if resuming:
            state = _checkpoints.restore_state(self.mngr, state, self.loader)
        self.state = state
        jax.block_until_ready(self.state)
        self._start = int(self.state.step)
        self.ptrain_step = jax.jit(functools.partial(self.train_mod.train_step, self.tc),
                                   in_shardings=(self.replicated, self.state_sharding, self.data_sharding),
                                   out_shardings=(self.state_sharding, self.replicated), donate_argnums=(1,))

        def _val(state, rng, batch):
            import flax.nnx as nnx

            model = nnx.merge(state.model_def, state.params)
            model.eval()
            obs, actions = batch
            return jnp.mean(model.compute_loss(rng, obs, actions, train=False))

        self.pval = jax.jit(_val, in_shardings=(self.state_sharding, self.replicated, self.data_sharding),
                            out_shardings=self.replicated)
        self.lr_fn = self.tc.lr_schedule.create()

        flat = traverse_util.flatten_dict(self.state.params.to_pure_dict(), sep="/")
        trainable = traverse_util.flatten_dict(self.state.params.filter(self.tc.trainable_filter).to_pure_dict(), sep="/")
        gates = {}
        trainable_probes, frozen_probes = probes_for(self.spec)
        if not resuming and self.spec.mode == "lora" and self.spec.lora_b_init == "zeros":
            nonzero = [k for k, v in flat.items() if LORA_B_RE.fullmatch(k) and bool(jnp.any(v != 0))]
            if nonzero:
                raise GateFailure(f"LoRA B not zero at initialisation: {nonzero[:5]}")
            gates["lora_b_zero_at_init"] = True
        missing = [p for p in trainable_probes + frozen_probes if p not in flat]
        if missing:
            raise GateFailure(f"probe parameters not found: {missing}")
        if any(p in trainable for p in frozen_probes) or not all(p in trainable for p in trainable_probes):
            raise GateFailure("probe leaves are not in the expected trainable/frozen partition")
        self.probe_init = {p: np.asarray(jax.device_get(flat[p])).copy() for p in trainable_probes + frozen_probes}
        self.trainable_probes, self.frozen_probes = trainable_probes, frozen_probes
        count = lambda d: int(sum(int(np.prod(v.shape)) for v in d.values()))  # noqa: E731
        lora_count = int(sum(int(np.prod(v.shape)) for k, v in trainable.items() if "lora" in k))
        return {
            "resumed": bool(resuming), "start_step": self._start, "total_params": count(flat),
            "trainable_params": count(trainable), "trainable_lora_params": lora_count,
            "trainable_non_lora_params": count(trainable) - lora_count,
            "lora": {"paligemma_variant": self.spec.paligemma_variant, "action_expert_variant": self.spec.action_expert_variant,
                     "paligemma_rank_alpha": [16, 16.0], "action_expert_rank_alpha": [32, 32.0],
                     "lora_b_init": self.spec.lora_b_init, "freeze_vision_encoder": self.spec.freeze_vision_encoder},
            "val_batches": n_val, "train_frames": LeRobotDatasetMetadata(self.train_repo).total_frames,
            "weight_loading_report": str(report) if report.exists() else None, "gates": gates,
            "init_s": round(time.time() - t0, 1),
        }

    def start_step(self) -> int:
        return self._start

    # -- loop ----------------------------------------------------------------------------------------------------
    def step(self, step: int) -> Dict[str, float]:
        import openpi.training.sharding as sharding

        batch = next(self.data_iter)
        with sharding.set_mesh(self.mesh):
            self.state, info = self.ptrain_step(self.train_rng, self.state, batch)
        info = self.jax.device_get(info)
        return {"loss": float(info["loss"]), "grad_norm": float(info["grad_norm"]),
                "param_norm": float(info["param_norm"]), "lr": float(self.lr_fn(step))}

    def validate(self) -> Dict[str, float]:
        import openpi.training.sharding as sharding

        losses = []
        with sharding.set_mesh(self.mesh):
            for i, batch in enumerate(self.val_loader):
                losses.append(float(self.pval(self.state, self.jax.random.fold_in(self.val_rng, i), batch)))
        return {"val_loss": float(np.mean(losses)), "val_batches": float(len(losses))}

    def save(self, completed_steps: int) -> Dict[str, Any]:
        import openpi.training.checkpoints as _checkpoints

        _checkpoints.save_state(self.mngr, self.state, self.loader, completed_steps)
        self.mngr.wait_until_finished()
        step_dir = Path(self.tc.checkpoint_dir) / str(completed_steps)
        saved = json.loads((step_dir / "assets" / ASSET_ID / "norm_stats.json").read_text())["norm_stats"]
        bundled = json.loads((Path(self.base_ckpt_dir) / "assets" / ASSET_ID / "norm_stats.json").read_text())["norm_stats"]
        if not norm_stats_equal(saved, bundled):
            raise GateFailure(f"norm stats saved with {step_dir} differ from the base checkpoint's")
        return {"step": completed_steps, "dir": str(step_dir)}

    def probe(self) -> Dict[str, Any]:
        from flax import traverse_util

        flat = traverse_util.flatten_dict(self.state.params.to_pure_dict(), sep="/")
        now = {p: np.asarray(self.jax.device_get(flat[p])) for p in self.probe_init}
        delta = {p: float(np.abs(now[p].astype(np.float64) - self.probe_init[p].astype(np.float64)).max()) for p in now}
        return {"max_abs_delta": delta,
                "trainable_changed": {p: delta[p] > 0 for p in self.trainable_probes},
                "frozen_unchanged": {p: delta[p] == 0 for p in self.frozen_probes}}

    def memory(self) -> Dict[str, Any]:
        try:
            stats = self.jax.local_devices()[0].memory_stats() or {}
        except Exception:  # noqa: BLE001
            stats = {}
        return {k: stats.get(k) for k in ("peak_bytes_in_use", "bytes_in_use", "bytes_limit")}

    def finish(self) -> Dict[str, Any]:
        self.mngr.wait_until_finished()
        return {"checkpoint_dir": str(self.tc.checkpoint_dir)}


# ---------------------------------------------------------------------------------------------------------------------
# Trained checkpoints -> inference


def trained_train_config(run_dir: Path, base_ckpt_dir: str):
    """Rebuild the exact TrainConfig of a finished run from its saved spec/metadata (needed to load its params)."""
    from clvla.training_core import TrainSpec as _Spec

    run_dir = Path(run_dir)
    spec = _Spec.from_yaml(run_dir / "train_spec.yaml")
    meta = json.loads((run_dir / "run_metadata.json").read_text())
    return build_train_config(spec, base_ckpt_dir=base_ckpt_dir, train_repo=meta["datasets"]["train"]["repo_id"],
                              seed=meta["seed"], checkpoint_base_dir=str(run_dir / "checkpoints"))


def load_trained_policy(run_dir: Path, base_ckpt_dir: str, step: Optional[int] = None):
    """OpenPI Policy for a trained run (norm stats from the trained checkpoint's assets = bundled values)."""
    from openpi.policies import policy_config

    tc = trained_train_config(run_dir, base_ckpt_dir)
    ckpt_root = Path(tc.checkpoint_dir)
    steps = sorted(int(p.name) for p in ckpt_root.iterdir() if p.name.isdigit())
    if not steps:
        raise FileNotFoundError(f"no saved checkpoints under {ckpt_root}")
    step = steps[-1] if step is None else step
    if step not in steps:
        raise FileNotFoundError(f"step {step} not saved (have {steps})")
    return policy_config.create_trained_policy(tc, ckpt_root / str(step)), step, ckpt_root / str(step)
