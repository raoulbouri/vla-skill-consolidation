"""Evaluate a policy served by OpenPI's official websocket server on the frozen 10-task LIBERO-90 manifest.

Runs in the simulator env (client side). Normally launched by scripts/baseline_eval.sh, which starts/stops the server
with the pinned pi05_libero checkpoint. Results go to <output_root>/<kind>/<stamp>_<name>/ (kind=baseline here, so
checkpoint-baseline results never mix with fine-tuning / continual-learning runs).

  scripts/py-sim scripts/run_eval.py --port 8000 [--episodes-per-task 10] [--save-videos]
  scripts/py-sim scripts/run_eval.py --port 8000 --from-run outputs/baseline/<run>   # rerun from saved snapshots
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

from clvla.assets import load_lock, verify_checkpoint
from clvla.config import derive_seed, load_experiment_config, seed_everything
from clvla.evaluation import (
    EpisodeResult,
    aggregate,
    append_episode_result,
    check_split_allowed,
    make_libero_env,
    run_episode,
    write_episode_results,
    write_summary,
)
from clvla.experiment import build_run_metadata, checkpoint_identity, make_run_dir, utc_stamp, write_run_metadata
from clvla.policy import RemotePolicy
from clvla.tasks import check_against_snapshot, init_state_indices, load_manifest, resolve_tasks

ROOT = Path(__file__).resolve().parents[1]

DEVIATIONS_FROM_OPENPI = [
    "Task set: custom frozen 10-task subset of LIBERO-90 (configs/task_manifest.yaml), not an official suite; results "
    "are NOT comparable to published LIBERO-Spatial/Object/Goal/10 numbers.",
    "Episodes: evaluation.episodes_per_task per task (OpenPI main.py default: 50), using the first N init states of the "
    "configured split (default 'test' = init states 10-49; OpenPI uses 0-49).",
    "Exceptions inside an episode are recorded as infra_error (OpenPI logs them and counts a failure); "
    "success_rate counts them as failures (OpenPI-compatible), success_rate_excl_infra excludes them.",
    "Python/NumPy are reseeded per episode with derive_seed(global, eval_seed, task_position, episode) (main.py seeds "
    "NumPy once); this does not affect the simulator or the policy.",
    "Unchanged from main.py: 256px rendering, 180-degree image rotation, resize_with_pad 224, 8-D state, replan every 5 "
    "actions, 10 settle steps with the dummy action, max_steps=400 (libero_90), env.seed(7) per task env, fresh env "
    "per task, success = env done.",
]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "configs/experiment.yaml"))
    ap.add_argument("--from-run", help="rerun using <run>/config.snapshot.yaml + task_manifest.snapshot.yaml")
    ap.add_argument("--episodes-per-task", type=int)
    ap.add_argument("--split", choices=["selection", "test"])
    ap.add_argument("--save-videos", action="store_true", default=None)
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--kind", default="baseline",
                    choices=["baseline", "smoke", "finetune", "specialist_eval", "teacher_eval"])
    ap.add_argument("--init-source", default="official",
                    help="'official' (LIBERO init files, --split) or 'demos:<profile>' (initial states of that split "
                         "profile's HELD-OUT demos, e.g. demos:n25 = demo_25..49, read from the processed HDF5)")
    ap.add_argument("--data-config", default=str(ROOT / "configs/datasets/processed_v1.yaml"))
    ap.add_argument("--confirmatory", action="store_true",
                    help="allow the test split for kind specialist_eval (reporting runs only; never for smoke decisions)")
    ap.add_argument("--name", default="pi05_libero_official")
    ap.add_argument("--positions", help="comma-separated manifest positions (debugging only; recorded)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    overrides = {}
    config_path, manifest_path = Path(args.config), None
    if args.from_run:
        run = Path(args.from_run)
        config_path, manifest_path = run / "config.snapshot.yaml", run / "task_manifest.snapshot.yaml"
        prev = json.loads((run / "run_metadata.json").read_text())
        if not (config_path.is_file() and manifest_path.is_file()):
            sys.exit(f"{run} lacks config/manifest snapshots")
        overrides.update({f"evaluation.{k}": v for k, v in prev["evaluation_settings"].items()})
    for key, val in (
        ("episodes_per_task", args.episodes_per_task),
        ("init_state_split", args.split),
        ("save_videos", args.save_videos),
        ("host", args.host),
        ("port", args.port),
    ):
        if val is not None:
            overrides[f"evaluation.{key}"] = val
    cfg = load_experiment_config(config_path, overrides=overrides)
    manifest = load_manifest(manifest_path or cfg.resolve(cfg.task_manifest))
    check_against_snapshot(manifest, ROOT / "configs/libero90_tasks.json")
    ev = cfg.evaluation

    lock = load_lock(ROOT / cfg.checkpoint.lock_file)  # repo-relative, also when the config is a run snapshot
    if args.from_run and prev["checkpoint"]["fingerprint_sha256"] != lock["fingerprint_sha256"]:
        sys.exit("checkpoint fingerprint differs from the run being reproduced; refusing to rerun")
    rep = verify_checkpoint(
        cfg.checkpoint.local_dir,
        lock["files"],
        cfg.checkpoint.norm_stats_asset_id,
        cfg.policy.state_dim,
        cfg.policy.action_dim,
    )
    rep.raise_if_failed()
    resolved = resolve_tasks(manifest)
    positions = [int(p) for p in args.positions.split(",")] if args.positions else list(range(len(resolved)))
    # Smoke / fine-tune decisions only ever look at selection init states (0-9). Test states (10-49 in protocol v1.0;
    # 10-19 already inspected, 20-49 proposed reporting set in v1.1) need an explicit confirmatory specialist run.
    demo_profile = args.init_source.split(":", 1)[1] if args.init_source.startswith("demos:") else None
    if args.init_source != "official" and not demo_profile:
        sys.exit(f"--init-source {args.init_source!r}: use 'official' or 'demos:<profile>'")
    demo_source = {}
    if demo_profile:
        # Held-out demonstrations' initial states (never official LIBERO init states, never a training demo).
        import yaml

        from clvla.lerobot_convert import resolve_split

        dcfg = yaml.safe_load(Path(args.data_config).read_text())
        _, held_out = resolve_split(dcfg, cfg, demo_profile)
        indices = held_out[: ev.episodes_per_task]
        proot = Path(os.environ[dcfg["output"]["processed_root_env"]]) / dcfg["version"] / "hdf5"
        demo_source = {"profile": demo_profile, "processed_version": dcfg["version"], "processed_dir": str(proot)}
    else:
        try:
            check_split_allowed(args.kind, ev.init_state_split, args.confirmatory)
        except ValueError as e:
            sys.exit(str(e))
        split = cfg.splits.test_init_states if ev.init_state_split == "test" else cfg.splits.selection_init_states
        indices = init_state_indices(split, ev.episodes_per_task)

    run_dir = make_run_dir(cfg.output_root, args.kind, args.name)
    run_id = run_dir.name
    server_info = json.loads(os.environ.get("CLVLA_SERVER_INFO", "{}"))
    meta = build_run_metadata(
        run_id=run_id,
        kind=args.kind,
        config=cfg,
        manifest=manifest,
        checkpoint=checkpoint_identity(cfg, lock["files"], lock["fingerprint_sha256"], "sizes+norm_stats"),
        repo_root=ROOT,
        extra={
            "evaluation_settings": {
                "episodes_per_task": ev.episodes_per_task,
                "init_state_split": ev.init_state_split,
                "save_videos": ev.save_videos,
                "host": ev.host,
                "port": ev.port,
            },
            "init_state_indices": indices,
            "init_source": args.init_source,
            "init_source_detail": demo_source or None,
            "task_positions": positions,
            "rerun_of": args.from_run,
            "policy_server": server_info,
            "deviations_from_openpi_protocol": DEVIATIONS_FROM_OPENPI,
        },
    )
    write_run_metadata(run_dir, meta, str(config_path), manifest.path)
    print(f"[run_eval] {run_dir}\n[run_eval] tasks {positions}, init states {indices}, seeds {cfg.eval_seeds}")

    client = RemotePolicy(ev.host, ev.port, cfg.policy.action_dim, cfg.policy.replan_steps)
    (run_dir / "server_metadata.json").write_text(json.dumps(client.server_metadata, indent=1, default=str))
    results = []
    t_run = time.time()
    for seed in cfg.eval_seeds:
        for pos in positions:
            rt = resolved[pos]
            import torch

            if demo_profile:
                import h5py

                with h5py.File(str(Path(demo_source["processed_dir"]) / f"{rt.spec.name}_demo.hdf5"), "r") as f:
                    init_states = {i: np.asarray(f["data"][f"demo_{i}"]["states"][0]) for i in indices}
            else:
                init_states = torch.load(str(rt.init_states_path))
            env = make_libero_env(rt.bddl_path, cfg.simulator.camera_resolution, seed)
            try:
                for ep_i, init_idx in enumerate(indices):
                    seed_everything(derive_seed(cfg.global_seed, seed, pos, ep_i))
                    t0 = time.time()
                    out = run_episode(
                        env,
                        client,
                        init_states[init_idx],
                        rt.spec.instruction,
                        max_steps=cfg.simulator.max_steps,
                        num_steps_wait=cfg.simulator.num_steps_wait,
                        replan_steps=cfg.policy.replan_steps,
                        resize_size=cfg.policy.resize_size,
                        dummy_action=cfg.simulator.dummy_action,
                        record_frames=ev.save_videos,
                    )
                    video = ""
                    if ev.save_videos:
                        import imageio

                        vdir = run_dir / "videos"
                        vdir.mkdir(exist_ok=True)
                        video = str(
                            vdir / f"p{pos}_t{rt.spec.libero_90_index:02d}_{'demo' if demo_profile else 'init'}{init_idx:02d}_s{seed}_"
                            f"{'success' if out.success else 'failure'}.mp4"
                        )
                        imageio.mimwrite(video, [np.asarray(f) for f in out.frames], fps=10)
                    r = EpisodeResult(
                        run_id=run_id,
                        task_position=pos,
                        libero_90_index=rt.spec.libero_90_index,
                        task_name=rt.spec.name,
                        instruction=rt.spec.instruction,
                        episode_idx=ep_i,
                        init_state_idx=init_idx,
                        env_seed=seed,
                        success=out.success,
                        infra_error=bool(out.error),
                        error=out.error,
                        steps=out.steps,
                        policy_calls=out.policy_calls,
                        wall_time_s=round(time.time() - t0, 3),
                        mean_inference_s=round(float(np.mean(out.inference_s)), 4) if out.inference_s else float("nan"),
                        video_path=video,
                    )
                    results.append(r)
                    append_episode_result(run_dir, r)
                    print(
                        f"[run_eval] p{pos} {rt.spec.libero_90_index:02d} init {init_idx:02d}: "
                        f"{'SUCCESS' if out.success else 'fail'} steps={out.steps} {r.wall_time_s:.1f}s"
                        f"{' ERROR ' + out.error if out.error else ''}",
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
            "run_id": run_id,
            "kind": args.kind,
            "finished_utc": utc_stamp(),
            "wall_time_s": round(time.time() - t_run, 1),
            "checkpoint_fingerprint_sha256": lock["fingerprint_sha256"],
            "task_manifest_sha256": manifest.content_sha256,
            "protocol_sha256": cfg.protocol_sha256,
            "comparability": "custom LIBERO-90 subset; not an official benchmark",
        },
    )
    print(
        f"\n[run_eval] success (pooled) {summary['success_rate_pooled']:.3f}; task mean "
        f"{summary['success_rate_task_mean']:.3f}; infra errors {summary['infra_errors']}"
    )
    for t in summary["per_task"]:
        print(f"  p{t['task_position']} {t['libero_90_index']:02d} {t['successes']}/{t['episodes']}  {t['task_name']}")
    print(f"[run_eval] results: {run_dir}")
    return 0 if summary["infra_errors"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
