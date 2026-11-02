"""Weights & Biases logging for training runs (optional; enabled by a `wandb:` block in the training spec).

Run names describe the run instead of numbering it: <method>-<N>demos-s<steps>-t<idx>, e.g.
`expert300m-10demos-s250-t042` (action expert fully trained) or `lora-25demos-s500-t078` (LoRA). The group is the
same string without the task, so all tasks of one variant sit together. The API key is read from the environment
(WANDB_API_KEY) only; it is never written to a spec, a log or the run directory.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Dict, Optional

LOGGED_KEYS = ("loss", "grad_norm", "param_norm", "lr", "step_time_s", "val_loss")


def method_tag(spec: Any) -> str:
    return "expert300m" if spec.mode == "expert" else "lora"


def variant_name(spec: Any, num_demos: int) -> str:
    return f"{method_tag(spec)}-{num_demos}demos-s{spec.num_steps}"


def run_name(spec: Any, num_demos: int, libero_90_index: int) -> str:
    return f"{variant_name(spec, num_demos)}-t{libero_90_index:03d}"


def wandb_settings(spec: Any) -> Optional[Dict[str, str]]:
    """{'entity', 'project'} from the spec, or None when the spec does not ask for W&B."""
    w = spec.raw.get("wandb")
    if not w:
        return None
    if not w.get("entity") or not w.get("project"):
        raise ValueError("training spec `wandb:` block needs `entity` and `project`")
    return {"entity": str(w["entity"]), "project": str(w["project"])}


def require_api_key(spec: Any) -> None:
    """Fail BEFORE training when W&B is requested but no key is available (no silently unlogged runs)."""
    if wandb_settings(spec) and not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "offline":
        raise RuntimeError("spec requests W&B logging but WANDB_API_KEY is not set (launch via the W&B launcher)")


class Tracker:
    """No-op unless the spec has a `wandb:` block."""

    def __init__(self, plan: Dict[str, Any], run_dir: str):
        spec = plan["spec"]
        self.settings = wandb_settings(spec)
        self.run = None
        if self.settings is None:
            return
        import wandb

        n_demos = int(plan["datasets"]["train"]["num_episodes"])
        idx = int(plan["task"]["libero_90_index"])
        self.name = run_name(spec, n_demos, idx)
        os.makedirs(os.path.join(run_dir, "wandb"), exist_ok=True)
        self.run = wandb.init(
            entity=self.settings["entity"], project=self.settings["project"], name=self.name,
            id=self.name, resume="allow",  # one W&B run per variant+task: a retry resumes it, never duplicates
            group=variant_name(spec, n_demos), job_type="train",
            tags=[method_tag(spec), f"{n_demos}demos", f"s{spec.num_steps}", f"t{idx:03d}"],
            dir=run_dir, settings=wandb.Settings(console="off"),
            config={"train_spec": spec.raw, "task": plan["task"], "seed": plan["seed"],
                    "signature": plan["signature"], "run_dir": run_dir,
                    "train_dataset": plan["datasets"]["train"], "val_dataset": plan["datasets"]["val"],
                    "base_checkpoint": plan["base"].get("fingerprint_sha256")},
        )

    def on_record(self, rec: Dict[str, Any]) -> None:
        if self.run is None:
            return
        data = {f"train/{k}" if k != "val_loss" else "val/loss": rec[k] for k in LOGGED_KEYS if k in rec}
        try:
            self.run.log(data, step=int(rec["completed_steps"]))
        except Exception as e:  # noqa: BLE001 - logging must never stop training
            print(f"[wandb] log failed at step {rec.get('completed_steps')}: {type(e).__name__}: {e}", flush=True)

    def finish(self, summary: Dict[str, Any]) -> None:
        if self.run is None:
            return
        try:
            ck = (summary.get("checkpoints") or [{}])[-1].get("dir")
            self.run.summary.update({"status": summary.get("status"), "final_val_loss": summary.get("final_val_loss"),
                                     "final_train_loss": summary.get("final_train_loss"), "checkpoint_dir": ck,
                                     "trained_checkpoint_fingerprint": summary.get("trained_checkpoint_fingerprint"),
                                     "peak_gpu_bytes": (summary.get("memory") or {}).get("peak_bytes_in_use")})
            self.run.finish(exit_code=0 if summary.get("status") == "completed" else 1)
        except Exception as e:  # noqa: BLE001
            print(f"[wandb] finish failed: {type(e).__name__}: {e}", flush=True)


def record_hook(tracker: Tracker) -> Callable[[Dict[str, Any]], None]:
    return tracker.on_record
