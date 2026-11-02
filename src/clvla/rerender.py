"""Re-render official LIBERO demos with deployment-aligned (pre-action) observations at 256x256.

Why (verified, see docs/EXPERIMENT_PROTOCOL.md section 6 and tests/integration/test_simulator.py): LIBERO's
scripts/create_dataset.py replays each raw demo, DROPS the first `cap_index = 5` steps, and stores
    states[k]  = raw pre-action state at raw step j = k + 5
    actions[k] = raw action at j
    obs[k]     = observation returned by env.step(actions[k])  ->  the state AFTER actions[k]  ~= sim(states[k+1])
at 128x128. The evaluation loop observes first and then acts, at 256x256. The aligned observation for actions[k]
is therefore render(states[k]); for k = 0 it does not exist in the HDF5 at all.

This module produces, per demo:
    obs[k] = render(states[k]) at 256x256 in the official LIBERO env (the same BDDL-built OffScreenRenderEnv the
             evaluation uses), paired with the ORIGINAL actions[k]; states, actions, rewards, dones copied unchanged.
and audits, without assuming the offset from array indices:
    * initial state: demo attrs["init_state"] == states[0]
    * offsets d in {-1, 0, +1}: residual between the 8-D state of render(states[k]) and HDF5 obs[k - d];
      the conversion proceeds only if d = +1 is clearly best and below threshold (else AlignmentAmbiguity)
    * replay (RECORDED, not gated by default): stepping actions[k] from the restored states[k] and comparing with
      states[k+1] gives qpos/qvel errors that are NOT a deterministic function of the demo. Measured on one task-35 demo:
      the same code gave a median qpos error of 0.0015 / 0.0055 / 0.0038 / 0.0010 on repeated passes, because the contact
      solver's warm start and the randomised reset are history that a (qpos, qvel) restore cannot reproduce (zeroing
      the warm start made it worse, 0.0085-0.014); qvel spikes to 5-10 during grasps; LIBERO's own continuous replay
      diverges more. The processed data never depends on replay (each frame is an exact restore of a recorded state),
      so a hard gate on it only rejects demos by chance. Demos with a median qpos error above `max_replay_qpos_median`
      are flagged (`replay.suspect`); `gate_on_replay: true` makes that a stop condition.
    * images: a 128 px render of states[k] must match HDF5 agentview[k-1] in the same (OpenGL) orientation

Pure helpers (planning, offset choice, manifests, paths) are unit-tested; the simulator paths need the sim env.
Must stay Python-3.8 compatible.
"""

from __future__ import annotations

import dataclasses
import hashlib
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from clvla.config import canonical_sha256

PROCESSED_FORMAT = "clvla-aligned-hdf5/1"
OFFSETS = (-1, 0, 1)
EXPECTED_OFFSET = 1  # HDF5 obs[k - 1] ~= render(states[k])
VOLATILE_KEYS = ("created_utc", "elapsed_s", "hostname")


class AlignmentAmbiguity(RuntimeError):
    """The observation/action offset could not be established; conversion must stop."""


class ReconstructionError(RuntimeError):
    """Reconstructed states/images disagree with the recorded demo beyond tolerance."""


@dataclasses.dataclass(frozen=True)
class Thresholds:
    max_state_residual: float = 3e-3  # observed <= 1.7e-3 for the aligned offset (convention test)
    offset_margin: float = 3.0  # every other offset's median residual must be >= margin x the aligned one
    max_replay_qpos_median: float = 0.01  # a demo whose MEDIAN one-step qpos replay error exceeds this is FLAGGED
    gate_on_replay: bool = False  # opt-in: stop on a flagged demo (off: replay is a noisy diagnostic, see docstring)
    max_image_mae: float = 12.0  # mean abs pixel error (0-255) of 128 px re-render vs HDF5 image
    image_check_every: int = 10  # compare images on every n-th frame (128 px render is extra work)
    require_final_success: bool = False  # informational by default (single-step replay of the last action)

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Thresholds":
        d = dict(d or {})
        unknown = set(d) - {f.name for f in dataclasses.fields(cls)}
        if unknown:
            raise ValueError(f"unknown threshold keys: {sorted(unknown)}")
        return cls(**d)


# ---------------------------------------------------------------------------------------------------------------------
# Pure helpers


def alignment_plan(T: int) -> Dict[str, List[Tuple[int, int]]]:
    """Index plan for one demo of length T (explicit so it can be reviewed and tested).

    pairs:       (action index k, MuJoCo state index rendered for its observation)  -> (k, k)
    validation:  (state index rendered, HDF5 obs index it must match)                -> (k, k - 1) for k >= 1
    replay:      (state index, action index, expected next state index)              -> (k, k, k + 1) for k < T-1
    """
    if T < 2:
        raise ValueError(f"demo too short to audit alignment (T={T})")
    return {
        "pairs": [(k, k) for k in range(T)],
        "validation": [(k, k - EXPECTED_OFFSET) for k in range(EXPECTED_OFFSET, T)],
        "replay": [(k, k, k + 1) for k in range(T - 1)],
    }


def offset_residuals(rendered_state8: np.ndarray, hdf5_state8: np.ndarray) -> Dict[int, float]:
    """Median over frames of max-abs 8-D state residual between render(states[k]) and HDF5 obs[k - d]."""
    T = len(rendered_state8)
    if hdf5_state8.shape != rendered_state8.shape:
        raise ValueError(f"shape mismatch {rendered_state8.shape} vs {hdf5_state8.shape}")
    out = {}
    for d in OFFSETS:
        ks = [k for k in range(T) if 0 <= k - d < T]
        if not ks:
            continue
        diffs = np.abs(rendered_state8[ks] - hdf5_state8[[k - d for k in ks]]).max(axis=1)
        out[d] = float(np.median(diffs))
    return out


def choose_offset(medians: Dict[int, float], th: Thresholds) -> int:
    """Return EXPECTED_OFFSET only if the data unambiguously supports it; otherwise raise AlignmentAmbiguity."""
    if EXPECTED_OFFSET not in medians:
        raise AlignmentAmbiguity(f"offset {EXPECTED_OFFSET} not measurable: {medians}")
    best = min(medians, key=lambda d: medians[d])
    aligned = medians[EXPECTED_OFFSET]
    if best != EXPECTED_OFFSET:
        raise AlignmentAmbiguity(f"best offset is {best}, not {EXPECTED_OFFSET}: medians {medians}")
    if aligned > th.max_state_residual:
        raise AlignmentAmbiguity(f"aligned residual {aligned:.5f} > {th.max_state_residual}: medians {medians}")
    others = [v for d, v in medians.items() if d != EXPECTED_OFFSET]
    if others and min(others) < th.offset_margin * max(aligned, 1e-9):
        raise AlignmentAmbiguity(f"offsets not separated by margin {th.offset_margin}: medians {medians}")
    return EXPECTED_OFFSET


def replay_summary(qpos_err, qvel_err):
    """Median / 95th percentile / max of the one-step replay errors (informational except the qpos median gate)."""
    out = {}
    for name, e in (("qpos", qpos_err), ("qvel", qvel_err)):
        a = np.asarray(e, dtype=np.float64)
        out.update(
            {
                f"{name}_median": float(np.median(a)) if a.size else 0.0,
                f"{name}_p95": float(np.percentile(a, 95)) if a.size else 0.0,
                f"{name}_max": float(a.max()) if a.size else 0.0,
            }
        )
    return out


def estimate_disk_bytes(n_frames: int, resolution: int = 256, cameras: int = 2) -> int:
    """Uncompressed upper bound for images (+~1 KB/frame low-dim). LeRobot PNG/HDF5 gzip are usually smaller."""
    return int(n_frames * (cameras * resolution * resolution * 3 + 1024))


def refuse_existing(*paths: os.PathLike) -> None:
    for p in paths:
        p = Path(p)
        if p.exists() or Path(str(p) + ".partial").exists():
            raise FileExistsError(f"refusing to overwrite existing output (or leftover partial): {p}")


def processed_paths(processed_root: os.PathLike, version: str, task_name: str) -> Dict[str, Path]:
    base = Path(processed_root) / version
    return {
        "version_dir": base,
        "hdf5": base / "hdf5" / f"{task_name}_demo.hdf5",
        "report": base / "hdf5" / f"{task_name}.report.json",
        "manifest": base / "manifest.json",
    }


def sha256_file(path: os.PathLike, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stable_content(obj: Any) -> Any:
    """Drop volatile keys (timestamps, durations, host) recursively so equal data hash equally."""
    if isinstance(obj, dict):
        return {k: stable_content(v) for k, v in obj.items() if k not in VOLATILE_KEYS}
    if isinstance(obj, list):
        return [stable_content(v) for v in obj]
    return obj


def build_processed_manifest(
    version: str,
    task_reports: Iterable[Dict[str, Any]],
    source: Dict[str, Any],
    thresholds: Thresholds,
    code: Dict[str, Any],
) -> Dict[str, Any]:
    """Deterministic aggregate manifest: tasks sorted by LIBERO-90 index; content hash excludes volatile keys."""
    tasks = sorted(task_reports, key=lambda r: r["libero_90_index"])
    idx = [t["libero_90_index"] for t in tasks]
    if len(set(idx)) != len(idx):
        raise ValueError(f"duplicate tasks in processed manifest: {idx}")
    m = {
        "format": PROCESSED_FORMAT,
        "version": version,
        "source": source,
        "transformations": [
            "observations re-rendered from recorded MuJoCo states[k] in the official LIBERO OffScreenRenderEnv "
            "(BDDL-built, as used for evaluation) at 256x256, raw OpenGL orientation (rotate 180 for OpenPI)",
            "obs[k] = pre-action observation for actions[k] (deployment alignment); HDF5 obs were post-action",
            "8-D state = eef_pos ++ quat2axisangle(eef_quat) ++ gripper_qpos computed from the re-rendered obs "
            "with the OpenPI client formula (clvla.policy.state_from_env_obs)",
            "actions, states, rewards, dones, demo ids, ordering and instruction copied unchanged",
            "no normalization statistics computed or copied",
        ],
        "thresholds": dataclasses.asdict(thresholds),
        "code": code,
        "tasks": tasks,
    }
    m["content_sha256"] = canonical_sha256(stable_content({k: v for k, v in m.items() if k != "content_sha256"}))
    return m


def check_split_isolation(train_keys: Sequence[str], val_keys: Sequence[str], all_keys: Sequence[str]) -> None:
    """Train and validation demos must be disjoint, non-empty and drawn from the demo file."""
    tr, va = set(train_keys), set(val_keys)
    if not tr or not va:
        raise ValueError("train and validation splits must both be non-empty")
    if tr & va:
        raise ValueError(f"data leakage: demos in both train and validation: {sorted(tr & va)}")
    missing = (tr | va) - set(all_keys)
    if missing:
        raise ValueError(f"split references demos not in the file: {sorted(missing)}")
    if len(tr) != len(train_keys) or len(va) != len(val_keys):
        raise ValueError("duplicate demo ids within a split")


# ---------------------------------------------------------------------------------------------------------------------
# Simulator paths (sim env: libero, robosuite, h5py)


def _state8(obs: Dict[str, np.ndarray]) -> np.ndarray:
    from clvla.policy import state_from_env_obs

    return state_from_env_obs(obs).astype(np.float64)


def rerender_episode(
    env: Any, env_small: Optional[Any], ep: Dict[str, np.ndarray], init_state: Optional[np.ndarray], th: Thresholds
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    """Re-render one demo. `ep` holds the HDF5 arrays (clvla.data.DEMO_SCHEMA keys). Returns (arrays, report).

    Raises AlignmentAmbiguity / ReconstructionError instead of writing doubtful data.
    """
    from clvla.data import episode_state

    states, actions = ep["states"], ep["actions"]
    T = len(actions)
    plan = alignment_plan(T)
    if init_state is not None and not np.allclose(np.asarray(init_state), states[0], atol=1e-9, rtol=0):
        raise ReconstructionError("demo attrs['init_state'] differs from states[0]")

    env.reset()
    nq, nv = env.sim.model.nq, env.sim.model.nv
    av, wr, s8, eef_pos, eef_quat, grip, joint = [], [], [], [], [], [], []
    replay_qpos, replay_qvel, final = [], [], {}
    for k, state_idx in plan["pairs"]:
        obs = env.set_init_state(states[state_idx])
        av.append(np.asarray(obs["agentview_image"]).copy())
        wr.append(np.asarray(obs["robot0_eye_in_hand_image"]).copy())
        s8.append(_state8(obs))
        eef_pos.append(np.asarray(obs["robot0_eef_pos"], np.float64))
        eef_quat.append(np.asarray(obs["robot0_eef_quat"], np.float64))
        grip.append(np.asarray(obs["robot0_gripper_qpos"], np.float64))
        joint.append(np.asarray(obs["robot0_joint_pos"], np.float64))
        post, _, done, _ = env.step(np.asarray(actions[k]).tolist())  # replay audit from the restored state
        if k < T - 1:
            e = env.sim.get_state().flatten() - states[k + 1]  # flat MuJoCo state: [time, qpos(nq), qvel(nv)]
            replay_qpos.append(float(np.linalg.norm(e[1 : 1 + nq])))
            replay_qvel.append(float(np.linalg.norm(e[1 + nq : 1 + nq + nv])))
        else:
            final = {
                "success": bool(env.check_success()) or bool(done),
                "state_residual_vs_hdf5_last": float(np.abs(_state8(post) - episode_state(ep)[T - 1]).max()),
            }

    rendered = np.stack(s8)
    medians = offset_residuals(rendered, episode_state(ep).astype(np.float64))
    offset = choose_offset(medians, th)
    aligned = np.abs(rendered[1:] - episode_state(ep)[:-1]).max(axis=1)
    replay = replay_summary(replay_qpos, replay_qvel)
    replay["suspect"] = bool(replay_qpos) and replay["qpos_median"] > th.max_replay_qpos_median
    problems = []
    if aligned.max() > th.max_state_residual:
        problems.append(f"aligned state residual max {aligned.max():.5f} > {th.max_state_residual}")
    if th.gate_on_replay and replay["suspect"]:
        problems.append(f"median one-step qpos replay error {replay['qpos_median']:.4f} > {th.max_replay_qpos_median}")
    if th.require_final_success and not final["success"]:
        problems.append("replaying the last action does not end in task success")

    image_report = {}
    if env_small is not None:
        env_small.reset()
        maes, maes_rot = [], []
        for k in range(1, T, max(1, th.image_check_every)):
            img = np.asarray(env_small.set_init_state(states[k])["agentview_image"]).astype(np.int32)
            ref = ep["obs/agentview_rgb"][k - 1].astype(np.int32)
            maes.append(float(np.abs(img - ref).mean()))
            maes_rot.append(float(np.abs(img[::-1, ::-1] - ref).mean()))
        image_report = {
            "frames_checked": len(maes),
            "mae_max": max(maes),
            "mae_mean": float(np.mean(maes)),
            "mae_mean_rotated": float(np.mean(maes_rot)),
        }
        if max(maes) > th.max_image_mae or np.mean(maes) >= 0.5 * np.mean(maes_rot):
            problems.append(f"image reconstruction check failed: {image_report}")
    if problems:
        raise ReconstructionError("; ".join(problems))

    arrays = {
        "actions": np.asarray(actions),
        "states": np.asarray(states),
        "rewards": np.asarray(ep["rewards"]),
        "dones": np.asarray(ep["dones"]),
        "obs/agentview_rgb": np.stack(av).astype(np.uint8),
        "obs/eye_in_hand_rgb": np.stack(wr).astype(np.uint8),
        "obs/state": rendered.astype(np.float32),
        "obs/ee_pos": np.stack(eef_pos),
        "obs/ee_quat": np.stack(eef_quat),
        "obs/gripper_qpos": np.stack(grip),
        "obs/joint_pos": np.stack(joint),
    }
    validate_rerendered(arrays, T, env_resolution(env))
    report = {
        "T": T,
        "offset": offset,
        "offset_median_residuals": {str(k): v for k, v in sorted(medians.items())},
        "aligned_state_residual_max": float(aligned.max()),
        "replay": replay,
        "final": final,
        "images": image_report,
    }
    return arrays, report


def env_resolution(env: Any) -> Optional[int]:
    try:
        return int(env.env.camera_heights[0])
    except Exception:  # noqa: BLE001 - informational only; validate_rerendered falls back to the array shape
        return None


def validate_rerendered(arrays: Dict[str, np.ndarray], T: int, resolution: Optional[int]) -> None:
    """Shapes, dtypes, finiteness of a re-rendered episode."""
    expect = {
        "actions": (T, 7),
        "states": None,
        "rewards": (T,),
        "dones": (T,),
        "obs/state": (T, 8),
        "obs/ee_pos": (T, 3),
        "obs/ee_quat": (T, 4),
        "obs/gripper_qpos": (T, 2),
        "obs/joint_pos": (T, 7),
    }
    for k, shape in expect.items():
        a = arrays[k]
        if len(a) != T or (shape is not None and a.shape != shape):
            raise ReconstructionError(f"{k}: shape {a.shape}, expected {shape or (T, '...')}")
        if a.dtype.kind == "f" and not np.all(np.isfinite(a)):
            raise ReconstructionError(f"{k}: non-finite values")
    for k in ("obs/agentview_rgb", "obs/eye_in_hand_rgb"):
        a = arrays[k]
        res = resolution or a.shape[1]
        if a.shape != (T, res, res, 3) or a.dtype != np.uint8:
            raise ReconstructionError(f"{k}: {a.shape} {a.dtype}, expected ({T}, {res}, {res}, 3) uint8")
