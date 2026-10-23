"""In-distribution control: run clvla's evaluation loop on an official suite pi05_libero WAS fine-tuned on.

High success here (OpenPI reports 98.8% on libero_spatial) shows the client/server/preprocessing pipeline is correct,
so low success on the LIBERO-90 manifest reflects the policy, not the infrastructure. Results go to
outputs/smoke/<stamp>_control_<suite>/ and never into baseline results. Needs a running policy server.

  scripts/py-sim scripts/control_eval.py --port <port> [--suite libero_spatial] [--trials 2]
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np

from clvla.config import load_experiment_config
from clvla.evaluation import (
    EpisodeResult,
    aggregate,
    make_libero_env,
    run_episode,
    write_episode_results,
    write_summary,
)
from clvla.experiment import make_run_dir
from clvla.policy import RemotePolicy

ROOT = pathlib.Path(__file__).resolve().parents[1]
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}  # OpenPI main.py


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--suite", default="libero_spatial", choices=sorted(MAX_STEPS))
    ap.add_argument("--trials", type=int, default=2)
    args = ap.parse_args()
    from libero.libero import benchmark, get_libero_path

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    suite = benchmark.get_benchmark_dict()[args.suite]()
    run_dir = make_run_dir(cfg.output_root, "smoke", f"control_{args.suite}")
    client = RemotePolicy("127.0.0.1", args.port, cfg.policy.action_dim, cfg.policy.replan_steps)
    results = []
    for i in range(suite.n_tasks):
        task = suite.get_task(i)
        bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        env = make_libero_env(bddl, cfg.simulator.camera_resolution, cfg.simulator.env_seed)
        init_states = suite.get_task_init_states(i)
        try:
            for ep in range(args.trials):  # init states 0..trials-1, as OpenPI main.py
                t0 = time.time()
                out = run_episode(
                    env,
                    client,
                    init_states[ep],
                    task.language,
                    max_steps=MAX_STEPS[args.suite],
                    num_steps_wait=cfg.simulator.num_steps_wait,
                    replan_steps=cfg.policy.replan_steps,
                    resize_size=cfg.policy.resize_size,
                    dummy_action=cfg.simulator.dummy_action,
                )
                results.append(
                    EpisodeResult(
                        run_id=run_dir.name,
                        task_position=i,
                        libero_90_index=-1,
                        task_name=task.name,
                        instruction=task.language,
                        episode_idx=ep,
                        init_state_idx=ep,
                        env_seed=cfg.simulator.env_seed,
                        success=out.success,
                        infra_error=bool(out.error),
                        error=out.error,
                        steps=out.steps,
                        policy_calls=out.policy_calls,
                        wall_time_s=round(time.time() - t0, 2),
                        mean_inference_s=round(float(np.mean(out.inference_s)), 4) if out.inference_s else float("nan"),
                    )
                )
                print(
                    f"{args.suite} task {i} ep {ep}: {'SUCCESS' if out.success else 'fail'} steps={out.steps}",
                    flush=True,
                )
        finally:
            env.close()
    write_episode_results(run_dir, results)
    summary = aggregate(results)
    write_summary(
        run_dir,
        summary,
        extra={
            "suite": args.suite,
            "trials_per_task": args.trials,
            "purpose": "in-distribution infrastructure control (not a baseline)",
        },
    )
    print(json.dumps({k: summary[k] for k in ("episodes", "successes", "success_rate_pooled", "infra_errors")}))
    print(f"results: {run_dir}")


if __name__ == "__main__":
    main()
