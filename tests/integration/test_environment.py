"""Stage 1: environment imports (run in each env; the other env's half skips)."""

import importlib
import sys

import pytest


@pytest.mark.policy_env
def test_policy_env_imports_and_versions():
    pytest.importorskip("openpi", reason="not the policy env")
    import flax
    import jax
    import numpy
    import orbax.checkpoint  # noqa: F401
    import torch
    from openpi.policies import policy_config  # noqa: F401
    from openpi.training import config as _config

    assert sys.version_info[:2] == (3, 11)
    assert jax.__version__ == "0.5.3" and flax.__version__ == "0.10.2" and torch.__version__.startswith("2.7.1")
    assert numpy.__version__.startswith("1.26")
    assert _config.get_config("pi05_libero").model.action_horizon == 10
    importlib.import_module("openpi_client.websocket_client_policy")


@pytest.mark.policy_env
@pytest.mark.gpu
def test_policy_env_sees_exactly_one_gpu():
    jax = pytest.importorskip("jax", reason="not the policy env")
    pytest.importorskip("openpi", reason="not the policy env")
    devs = [d for d in jax.devices() if d.platform == "gpu"]
    if not devs:
        pytest.skip("CPU-only run (set CLVLA_USE_GPU=1 to test the GPU)")
    assert len(devs) == 1, devs  # CUDA_VISIBLE_DEVICES=CLVLA_GPU


@pytest.mark.sim
def test_sim_env_imports_and_versions():
    pytest.importorskip("libero", reason="not the simulator env")
    import h5py  # noqa: F401
    import mujoco
    import numpy
    import robosuite
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv  # noqa: F401
    from openpi_client import image_tools, websocket_client_policy  # noqa: F401

    assert sys.version_info[:2] == (3, 8)
    assert robosuite.__version__ == "1.4.1" and mujoco.__version__ == "3.2.3" and numpy.__version__ == "1.22.4"
    assert torch.__version__.startswith("1.11.0")
    assert "libero_90" in benchmark.get_benchmark_dict()
    assert "IRL-project/third_party/openpi/third_party/libero" in get_libero_path("bddl_files")
