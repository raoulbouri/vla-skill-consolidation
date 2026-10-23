import types

import pytest

from clvla.policy import CompatibilityError, check_action_chunk, check_checkpoint_compat

ASSET = "physical-intelligence/libero"


def fake_configs(name="pi05_libero", horizon=10, action_dim=32, asset=ASSET, quantile=True):
    tc = types.SimpleNamespace(
        name=name,
        model=types.SimpleNamespace(action_horizon=horizon, action_dim=action_dim, model_type="ModelType.PI05"),
    )
    dc = types.SimpleNamespace(asset_id=asset, use_quantile_norm=quantile)
    return tc, dc


def check(tc, dc, ckpt):
    return check_checkpoint_compat(
        tc,
        dc,
        str(ckpt),
        expected_config="pi05_libero",
        expected_asset_id=ASSET,
        state_dim=8,
        action_dim=7,
        action_horizon=10,
    )


def test_compatible(fake_checkpoint):
    root, _ = fake_checkpoint
    rep = check(*fake_configs(), root)
    assert rep.config_name == "pi05_libero" and rep.asset_id == ASSET and rep.use_quantile_norm


@pytest.mark.parametrize(
    "kwargs,msg",
    [
        ({"name": "pi05_droid"}, "requires 'pi05_libero'.*do not substitute"),
        ({"horizon": 50}, "action_horizon 50 != protocol 10"),
        ({"action_dim": 6}, "action_dim 6 < LIBERO action dim 7"),
        ({"asset": "droid"}, "asset_id 'droid' != expected"),
    ],
)
def test_incompatible_config_messages(fake_checkpoint, kwargs, msg):
    root, _ = fake_checkpoint
    with pytest.raises(CompatibilityError, match=msg):
        check(*fake_configs(**kwargs), root)


def test_wrong_directory_and_missing_norm_stats(tmp_path, fake_checkpoint):
    with pytest.raises(CompatibilityError, match="not an OpenPI JAX checkpoint") as e:
        check(*fake_configs(), tmp_path)
    assert "bundled norm stats not found" in str(e.value) and "never recompute" in str(e.value)


def test_action_chunk_checks():
    import numpy as np

    assert check_action_chunk(np.zeros((10, 7)), 7, 5).shape == (10, 7)
    with pytest.raises(CompatibilityError, match=r"shape \(10, 8\)"):
        check_action_chunk(np.zeros((10, 8)), 7, 5)
    with pytest.raises(CompatibilityError, match="need >= replan_steps=5"):
        check_action_chunk(np.zeros((3, 7)), 7, 5)
    with pytest.raises(CompatibilityError, match="non-finite"):
        check_action_chunk(np.full((10, 7), np.nan), 7, 5)
