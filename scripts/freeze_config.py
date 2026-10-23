"""Recompute and store protocol_sha256 in configs/experiment.yaml. A deliberate act: run only after an intended
protocol change (with protocol_version bumped and the change logged in docs/EXPERIMENT_PROTOCOL.md).

Usage: scripts/py-policy scripts/freeze_config.py [--check]
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from clvla.config import compute_protocol_sha256, load_yaml, parse_experiment_config

PATH = Path(__file__).resolve().parents[1] / "configs/experiment.yaml"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="only verify; exit 1 if the stored hash is stale")
    args = ap.parse_args()
    raw = load_yaml(PATH)
    new = compute_protocol_sha256(raw)
    if raw.get("protocol_sha256") == new:
        print(f"protocol_sha256 up to date: {new}")
        return
    if args.check:
        sys.exit(f"protocol_sha256 is stale: stored {raw.get('protocol_sha256')} != {new}")
    text = PATH.read_text()
    text, n = re.subn(r"^protocol_sha256: .*$", f"protocol_sha256: {new}", text, flags=re.M)
    assert n == 1, "expected exactly one protocol_sha256 line"
    PATH.write_text(text)
    parse_experiment_config(
        load_yaml(PATH), str(PATH), env={"OPENPI_DATA_HOME": "x", "LIBERO_DATA_ROOT": "x", "CLVLA_OUTPUT_ROOT": "x"}
    )
    print(f"protocol frozen: protocol_version={raw['protocol']['protocol_version']} protocol_sha256={new}")


if __name__ == "__main__":
    main()
