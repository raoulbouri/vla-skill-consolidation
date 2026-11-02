"""Training initialisation from the EXACT local pi05_libero checkpoint (policy env, checkpoint; CPU is enough).

Loads ~12 GB of params into host memory; no GPU and no training step. Verifies the static audit at runtime:
exact checkpoint->model key mapping and shapes, LoRA leaves from init with B = 0, bundled norm stats in the data config.
"""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("openpi", reason="policy env only")

from clvla.training_core import TrainSpec  # noqa: E402

pytestmark = [pytest.mark.policy_env, pytest.mark.checkpoint, pytest.mark.slow]
ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def tc(cfg, tmp_path_factory):
    from clvla.openpi_training import build_train_config

    if not Path(cfg.checkpoint.local_dir, "params").is_dir():
        pytest.skip("checkpoint not downloaded")
    spec = TrainSpec.from_yaml(ROOT / "configs/training/smoke_task78.yaml")
    report = tmp_path_factory.mktemp("init") / "weight_loading_report.json"
    return build_train_config(spec, base_ckpt_dir=cfg.checkpoint.local_dir, train_repo="clvla/unused-train", seed=1,
                              checkpoint_base_dir=str(report.parent / "ck"), report_path=str(report)), report


def test_config_matches_pi05_libero_except_documented_changes(tc, cfg):
    import openpi.training.config as _config

    t, _ = tc
    base = _config.get_config("pi05_libero")
    assert t.model.pi05 and t.model.action_horizon == base.model.action_horizon == 10
    assert t.model.action_dim == base.model.action_dim and t.model.max_token_len == base.model.max_token_len
    assert t.model.discrete_state_input == base.model.discrete_state_input
    assert (t.model.paligemma_variant, t.model.action_expert_variant) == ("gemma_2b_lora", "gemma_300m_lora")
    assert t.ema_decay is None and not t.data.extra_delta_transform and t.wandb_enabled is False
    assert t.weight_loader.params_path == str(Path(cfg.checkpoint.local_dir) / "params")  # NOT pi05_base
    assert "pi05_base" not in t.weight_loader.params_path


def test_data_config_uses_bundled_norm_stats(tc, cfg):
    from clvla.openpi_training import norm_stats_equal

    t, _ = tc
    dc = t.data.create(t.assets_dirs, t.model)
    bundled = json.loads((Path(cfg.checkpoint.local_dir) / "assets/physical-intelligence/libero/norm_stats.json")
                         .read_text())["norm_stats"]
    assert dc.asset_id == "physical-intelligence/libero" and dc.use_quantile_norm and dc.prompt_from_task
    assert norm_stats_equal(dc.norm_stats, bundled)


def test_strict_loader_maps_checkpoint_exactly(tc):
    import flax.nnx as nnx
    import jax
    from flax import traverse_util

    t, report = tc
    abstract = nnx.eval_shape(lambda: t.model.create(jax.random.key(0)))
    params_shape = nnx.state(abstract).to_pure_dict()
    merged = t.weight_loader.load(params_shape)  # raises GateFailure on any dropped/missing key or shape mismatch
    rep = json.loads(report.read_text())
    assert rep["dropped"] == [] and rep["missing_non_lora"] == [] and rep["shape_mismatches"] == []
    assert rep["lora_leaves_from_init"] > 0 and rep["lora_b_zeroed"]
    flat = traverse_util.flatten_dict(merged, sep="/")
    for k in rep["lora_b_zeroed"]:
        assert isinstance(flat[k], np.ndarray) and not flat[k].any()
    lora_a = [k for k in flat if k.endswith("lora_a")]
    assert lora_a and all(isinstance(flat[k], jax.ShapeDtypeStruct) for k in lora_a)  # left to OpenPI's init
    print("\n", {k: rep[k] for k in ("checkpoint_leaves", "model_leaves", "lora_leaves_from_init", "dtype_casts")})
