"""R8 screening client: evaluate the OFFICIAL pi05_libero on arbitrary LIBERO-90 tasks, selection init states only.

Normally launched by scripts/screen_tasks.sh (starts/stops the official server). Same episode loop, preprocessing,
seed and 400-step cap as the baseline. Results: outputs/screening/<stamp>_<name>/{episodes.*,per_task.csv,summary.json}.

  scripts/py-sim scripts/screen_tasks.py --port <port>                     # tasks from configs/proposed/screening_v1.1.yaml
  scripts/py-sim scripts/screen_tasks.py --port <port> --indices 55,57
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from clvla.config import load_experiment_config
from clvla.evaluation import (
    EpisodeResult,
    aggregate,
    append_episode_result,
    make_libero_env,
    run_episode,
    write_episode_results,
    write_summary,
)
from clvla.experiment import environment_info, git_revision, make_run_dir, utc_stamp
from clvla.policy import RemotePolicy

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--config", default=str(ROOT / "configs/proposed/screening_v1.1.yaml"))
    ap.add_argument("--indices", help="override the task list (comma-separated LIBERO-90 indices)")
    ap.add_argument("--name", default="r8_screening_v1.1")
    args = ap.parse_args(argv)

    import torch
    from libero.libero import get_libero_path

    cfg = load_experiment_config(ROOT / "configs/experiment.yaml")
    scfg = yaml.safe_load(Path(args.config).read_text())
    init_states = list(range(scfg["init_states"]["start"], scfg["init_states"]["stop"]))
    if not set(init_states) <= set(cfg.splits.selection_init_states):
        raise SystemExit(f"screening init states {init_states} must lie in the selection split")
    n = scfg["episodes_per_task"]
    if n > len(init_states):
        raise SystemExit("episodes_per_task exceeds the screening init states")
    indices = ([int(i) for i in args.indices.split(",")] if args.indices else
               scfg["retained_tasks"] + scfg["control_tasks"] + scfg["replacement_candidates"])
    snap = {r["idx"]: r for r in json.loads((ROOT / "configs/libero90_tasks.json").read_text())["libero_90"]}
    run_dir = make_run_dir(cfg.output_root, "screening", args.name)
    (run_dir / "screening_metadata.json").write_text(json.dumps({
        "purpose": "R8 screening (official pi05_libero, selection init states only)", "config": scfg,
        "indices": indices, "init_states": init_states[:n], "env_seed": cfg.simulator.env_seed,
        "max_steps": cfg.simulator.max_steps, "code": git_revision(ROOT), "environment": environment_info(),
        "created_utc": utc_stamp()}, indent=1, default=str))
    client = RemotePolicy("127.0.0.1", args.port, cfg.policy.action_dim, cfg.policy.replan_steps)
    results = []
    for pos, idx in enumerate(indices):
        r = snap[idx]
        bddl = Path(get_libero_path("bddl_files")) / r["problem_folder"] / r["bddl_file"]
        states = torch.load(str(Path(get_libero_path("init_states")) / r["problem_folder"] / r["init_states_file"]))
        env = make_libero_env(bddl, cfg.simulator.camera_resolution, cfg.simulator.env_seed)
        try:
            for ep, init_idx in enumerate(init_states[:n]):
                t0 = time.time()
                out = run_episode(env, client, states[init_idx], r["language"], max_steps=cfg.simulator.max_steps,
                                  num_steps_wait=cfg.simulator.num_steps_wait, replan_steps=cfg.policy.replan_steps,
                                  resize_size=cfg.policy.resize_size, dummy_action=cfg.simulator.dummy_action)
                res = EpisodeResult(
                    run_id=run_dir.name, task_position=pos, libero_90_index=idx, task_name=r["name"],
                    instruction=r["language"], episode_idx=ep, init_state_idx=init_idx, env_seed=cfg.simulator.env_seed,
                    success=out.success, infra_error=bool(out.error), error=out.error, steps=out.steps,
                    policy_calls=out.policy_calls, wall_time_s=round(time.time() - t0, 2),
                    mean_inference_s=round(float(np.mean(out.inference_s)), 4) if out.inference_s else float("nan"))
                results.append(res)
                append_episode_result(run_dir, res)
                print(f"t{idx:02d} init {init_idx}: {'SUCCESS' if out.success else 'fail'} steps={out.steps}", flush=True)
        finally:
            env.close()
    write_episode_results(run_dir, results)
    summary = aggregate(results)
    write_summary(run_dir, summary, extra={"kind": "screening", "r8_max_successes": scfg["r8_max_successes"],
                                           "init_states": init_states[:n]})
    for t in summary["per_task"]:
        flag = "R8 ok" if t["successes"] <= scfg["r8_max_successes"] and t["infra_errors"] == 0 else "R8 FAIL/INVALID"
        print(f"  t{t['libero_90_index']:02d} {t['successes']}/{t['episodes']} infra {t['infra_errors']}  {flag}")
    print(f"results: {run_dir}")
    return 0 if summary["infra_errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
