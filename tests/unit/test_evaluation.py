import json

import numpy as np
import pytest

from clvla.evaluation import (
    EpisodeResult,
    aggregate,
    append_episode_result,
    read_episode_results,
    run_episode,
    wilson_interval,
    write_episode_results,
    write_summary,
)

DUMMY = [0.0] * 6 + [-1.0]


class FakeEnv:
    """Records actions; reports success once `succeed_after` policy actions have been executed."""

    def __init__(self, succeed_after=None, raise_at=None):
        self.succeed_after, self.raise_at = succeed_after, raise_at
        self.actions, self.init_state = [], None

    def reset(self):
        self.actions = []

    def set_init_state(self, s):
        self.init_state = s
        return {"t": 0}

    def step(self, action):
        self.actions.append(list(action))
        n_policy = sum(a != DUMMY for a in self.actions)
        if self.raise_at is not None and len(self.actions) == self.raise_at:
            raise RuntimeError("simulator exploded")
        done = self.succeed_after is not None and n_policy >= self.succeed_after
        return {"t": len(self.actions)}, float(done), done, {}


class FakePolicy:
    def __init__(self, horizon=10):
        self.calls, self.horizon = 0, horizon

    def __call__(self, request):
        self.calls += 1
        a = np.zeros((self.horizon, 7))
        a[:, 0] = self.calls  # tag each chunk so we can see where replanning happened
        a[:, 1] = np.arange(self.horizon)
        return a


def fake_obs(obs, instruction, resize):
    return {"observation/image": np.zeros((resize, resize, 3), np.uint8), "prompt": instruction}


def run(env, policy, max_steps=40, **kw):
    return run_episode(
        env,
        policy,
        "INIT",
        "do it",
        max_steps=max_steps,
        num_steps_wait=10,
        replan_steps=5,
        resize_size=4,
        dummy_action=DUMMY,
        build_obs=fake_obs,
        **kw,
    )


def test_wait_steps_then_replan_every_five_actions():
    env, pol = FakeEnv(), FakePolicy()
    out = run(env, pol, max_steps=23)
    assert env.init_state == "INIT"
    assert env.actions[:10] == [DUMMY] * 10
    policy_actions = env.actions[10:]
    assert len(policy_actions) == 23 and out.steps == 23
    assert pol.calls == out.policy_calls == 5  # ceil(23 / 5)
    chunk_ids = [a[0] for a in policy_actions]
    assert chunk_ids[:5] == [1] * 5 and chunk_ids[5:10] == [2] * 5
    assert [a[1] for a in policy_actions[:6]] == [0, 1, 2, 3, 4, 0]  # first 5 of each chunk only
    assert not out.success and out.error == ""


def test_success_stops_episode():
    env = FakeEnv(succeed_after=7)
    out = run(env, FakePolicy())
    assert out.success and out.steps == 7 and len(env.actions) == 17


def test_exception_recorded_as_infra_error_not_hidden():
    out = run(FakeEnv(raise_at=12), FakePolicy())
    assert not out.success and out.error == "RuntimeError: simulator exploded"


def test_short_action_chunk_is_an_error():
    out = run(FakeEnv(), FakePolicy(horizon=3))
    assert "replan_steps" in out.error


def test_frames_recorded_per_control_step():
    out = run(FakeEnv(), FakePolicy(), max_steps=6, record_frames=True)
    assert len(out.frames) == 6 and len(out.inference_s) == 2


def ep(pos, i, success, infra=False, idx=None):
    return EpisodeResult(
        run_id="r",
        task_position=pos,
        libero_90_index=idx if idx is not None else pos * 10,
        task_name=f"T{pos}",
        instruction=f"do {pos}",
        episode_idx=i,
        init_state_idx=10 + i,
        env_seed=7,
        success=success,
        infra_error=infra,
        error="boom" if infra else "",
        steps=5,
        policy_calls=1,
        wall_time_s=1.0,
        mean_inference_s=0.1,
    )


def test_aggregate_success_rates_counts_and_task_means():
    results = [ep(0, i, s) for i, s in enumerate([1, 1, 0, 0])] + [ep(1, i, s) for i, s in enumerate([1, 0])]
    results.append(ep(1, 2, False, infra=True))
    s = aggregate(results)
    assert s["num_tasks"] == 2 and s["episodes"] == 7 and s["successes"] == 3 and s["infra_errors"] == 1
    assert s["success_rate_pooled"] == pytest.approx(3 / 7)
    t0, t1 = s["per_task"]
    assert (t0["episodes"], t0["successes"], t0["success_rate"]) == (4, 2, 0.5)
    assert (t1["episodes"], t1["successes"], t1["infra_errors"]) == (3, 1, 1)
    assert t1["success_rate"] == pytest.approx(1 / 3) and t1["success_rate_excl_infra"] == 0.5
    assert s["success_rate_task_mean"] == pytest.approx((0.5 + 1 / 3) / 2)


def test_aggregate_orders_by_manifest_position_and_handles_empty():
    s = aggregate([ep(3, 0, True, idx=89), ep(1, 0, False, idx=16)])
    assert [t["task_position"] for t in s["per_task"]] == [1, 3]
    empty = aggregate([])
    assert empty["episodes"] == 0 and np.isnan(empty["success_rate_pooled"])


def test_wilson_interval():
    lo, hi = wilson_interval(5, 10)
    assert lo < 0.5 < hi and lo >= 0 and hi <= 1
    assert wilson_interval(0, 10)[0] == 0.0 and wilson_interval(10, 10)[1] == pytest.approx(1.0)
    assert all(np.isnan(wilson_interval(0, 0)))


def test_result_files_roundtrip(tmp_path):
    results = [ep(0, 0, True), ep(0, 1, False), ep(1, 0, False, infra=True)]
    write_episode_results(tmp_path, results)
    for r in results:
        append_episode_result(tmp_path, r)
    assert read_episode_results(tmp_path / "episodes.jsonl") == results
    assert read_episode_results(tmp_path / "episodes.partial.jsonl") == results
    header = (tmp_path / "episodes.csv").read_text().splitlines()[0]
    assert header.startswith("run_id,task_position,libero_90_index")
    write_summary(tmp_path, aggregate(results), extra={"note": "x"})
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["note"] == "x" and summary["episodes"] == 3
    assert len((tmp_path / "per_task.csv").read_text().splitlines()) == 3
