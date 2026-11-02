"""Build an orderly, rerunnable index of finished task checkpoints (CPU, read-only on the runs themselves).

  outputs/checkpoints/<variant>/t<idx>_<task name>  ->  symlink to the run's final OpenPI checkpoint step dir
  outputs/checkpoints/INDEX.json / INDEX.md          ->  variant, task, demos, steps, val loss, run dir, fingerprint

<variant> is <method>-<N>demos-s<steps> (lora | expert300m), the same string as the W&B group. Only runs with
status `completed` and an existing final checkpoint are indexed; runs whose checkpoints were deleted are skipped.
Nothing is copied or moved: run directories keep their full provenance.

  scripts/py-policy scripts/index_checkpoints.py [--runs-root outputs/specialists]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def variant_of(spec_raw: dict, num_demos: int) -> str:
    method = "expert300m" if spec_raw["model"]["mode"] == "expert" else "lora"
    return f"{method}-{num_demos}demos-s{spec_raw['train']['num_steps']}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default=str(ROOT / "outputs/specialists"))
    ap.add_argument("--out", default=str(ROOT / "outputs/checkpoints"))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], []
    for summ in sorted(Path(args.runs_root).glob("*/summary.json")):
        run = summ.parent
        s = json.loads(summ.read_text())
        meta_p = run / "run_metadata.json"
        if s.get("status") != "completed" or not meta_p.is_file():
            skipped.append((run.name, f"status {s.get('status')}"))
            continue
        meta = json.loads(meta_p.read_text())
        ck = Path((s.get("checkpoints") or [{}])[-1].get("dir", ""))
        if not ck.is_dir():
            skipped.append((run.name, "final checkpoint missing"))
            continue
        spec_raw = yaml.safe_load((run / "train_spec.yaml").read_text())
        task = meta["task"]
        n = int(meta["datasets"]["train"]["num_episodes"])
        variant = variant_of(spec_raw, n)
        link_dir = out / variant
        link_dir.mkdir(exist_ok=True)
        link = link_dir / f"t{task['libero_90_index']:03d}_{task['name']}"
        rows.append({"variant": variant, "libero_90_index": task["libero_90_index"], "task": task["name"],
                     "instruction": task["instruction"], "method": spec_raw["model"]["mode"], "train_demos": n,
                     "held_out_demos": meta["datasets"]["val"]["demo_keys"], "steps": spec_raw["train"]["num_steps"],
                     "final_val_loss": s.get("final_val_loss"), "run_dir": str(run), "checkpoint_dir": str(ck),
                     "checkpoint_fingerprint": s.get("trained_checkpoint_fingerprint"), "link": str(link)})
    by_key = {}
    for r in rows:  # newest run wins if a variant/task was trained twice
        by_key[(r["variant"], r["libero_90_index"])] = r
    rows = sorted(by_key.values(), key=lambda r: (r["variant"], r["libero_90_index"]))
    for r in rows:
        link = Path(r["link"])
        if link.is_symlink() or link.exists():
            link.unlink()
        link.symlink_to(r["checkpoint_dir"])
    (out / "INDEX.json").write_text(json.dumps(rows, indent=1))
    lines = ["| variant | task | demos | steps | final val loss | checkpoint |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['variant']} | t{r['libero_90_index']:03d} {r['instruction']} | {r['train_demos']} | "
                     f"{r['steps']} | {r['final_val_loss']:.4f} | `{Path(r['link']).relative_to(out)}` |")
    (out / "INDEX.md").write_text("\n".join(lines) + "\n")
    counts = {}
    for r in rows:
        counts[r["variant"]] = counts.get(r["variant"], 0) + 1
    print(f"indexed {len(rows)} checkpoints in {out}: {counts}; skipped {len(skipped)} runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
