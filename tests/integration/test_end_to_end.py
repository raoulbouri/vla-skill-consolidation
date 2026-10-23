"""Stage 5: simulator -> official policy server -> simulator, one short capped episode (infrastructure, not performance)."""

import json
import time

import numpy as np
import pytest

pytest.importorskip("libero", reason="simulator env only")

from clvla.evaluation import make_libero_env, run_episode  # noqa: E402
from clvla.policy import RemotePolicy  # noqa: E402
from clvla.tasks import resolve_tasks  # noqa: E402

pytestmark = [pytest.mark.sim, pytest.mark.server, pytest.mark.gpu, pytest.mark.checkpoint, pytest.mark.slow]
SMOKE_MAX_STEPS = 60  # cap for the smoke episode (the frozen protocol uses 400); success is not required


def test_short_episode_end_to_end(policy_server, cfg, manifest, smoke_dir):
    import imageio
    import torch

    rt = resolve_tasks(manifest)[0]
    init_idx = cfg.splits.test_init_states[0]
    env = make_libero_env(rt.bddl_path, cfg.simulator.camera_resolution, cfg.simulator.env_seed)
    client = RemotePolicy(policy_server["host"], policy_server["port"], cfg.policy.action_dim, cfg.policy.replan_steps)
    try:
        t0 = time.time()
        out = run_episode(
            env,
            client,
            torch.load(str(rt.init_states_path))[init_idx],
            rt.spec.instruction,
            max_steps=SMOKE_MAX_STEPS,
            num_steps_wait=cfg.simulator.num_steps_wait,
            replan_steps=cfg.policy.replan_steps,
            resize_size=cfg.policy.resize_size,
            dummy_action=cfg.simulator.dummy_action,
            record_frames=True,
        )
        wall = time.time() - t0
    finally:
        env.close()
    video = smoke_dir / f"e2e_{rt.spec.libero_90_index:02d}_{'success' if out.success else 'failure'}.mp4"
    imageio.mimwrite(str(video), [np.asarray(f) for f in out.frames], fps=10)
    record = {
        "task_id": rt.spec.task_id,
        "instruction": rt.spec.instruction,
        "init_state_idx": init_idx,
        "max_steps": SMOKE_MAX_STEPS,
        "success": out.success,
        "steps": out.steps,
        "policy_calls": out.policy_calls,
        "error": out.error,
        "wall_time_s": round(wall, 2),
        "mean_inference_s": round(float(np.mean(out.inference_s)), 4) if out.inference_s else None,
        "video": str(video),
    }
    (smoke_dir / "e2e_episode.json").write_text(json.dumps(record, indent=2))
    print("\n" + json.dumps(record, indent=2))
    assert out.error == "", out.error  # infrastructure must work; the policy may or may not succeed in 60 steps
    assert out.policy_calls >= 1 and out.steps >= 1 and len(out.frames) == out.steps
