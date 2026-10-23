"""Integration fixtures: the real config/manifest/assets and (session-scoped) the official OpenPI policy server.

Run through the wrappers so paths/env vars are set:  scripts/py-sim -m pytest ...  /  scripts/py-policy -m pytest ...
Tests needing the other environment skip with an explicit reason; see scripts/run_tests.sh for the staged order.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

import pytest

from clvla.config import load_experiment_config
from clvla.experiment import utc_stamp
from clvla.tasks import load_manifest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="session")
def cfg():
    if "CLVLA_ROOT" not in os.environ:
        pytest.skip("integration tests must run via scripts/py-sim or scripts/py-policy (env.sh not sourced)")
    return load_experiment_config(ROOT / "configs/experiment.yaml")


@pytest.fixture(scope="session")
def manifest(cfg):
    return load_manifest(cfg.resolve(cfg.task_manifest))


@pytest.fixture(scope="session")
def smoke_dir(cfg):
    d = Path(cfg.output_root) / "smoke" / f"{utc_stamp()}_pytest"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _healthy(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=2) as r:
            return r.status == 200
    except OSError:
        return False


@pytest.fixture(scope="session")
def policy_server(cfg, smoke_dir):
    """Start scripts/serve_policy.sh (official OpenPI server, pinned checkpoint, CLVLA_GPU) and wait for /healthz."""
    guard = subprocess.run([str(ROOT / "scripts/gpu_guard.sh")], capture_output=True, text=True)
    if guard.returncode != 0:
        pytest.skip(f"GPU not available for the policy server: {guard.stdout.strip()}")
    port = _free_port()
    log = smoke_dir / f"policy_server_{port}.log"
    with log.open("w") as f:
        proc = subprocess.Popen(
            [str(ROOT / "scripts/serve_policy.sh"), str(port)],
            stdout=f,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    t0 = time.time()
    try:
        while not _healthy(port):
            if proc.poll() is not None:
                pytest.fail(f"policy server exited with code {proc.returncode}; log: {log}\n{log.read_text()[-3000:]}")
            if time.time() - t0 > 900:
                pytest.fail(f"policy server not healthy after 900 s; log: {log}")
            time.sleep(3)
        yield {"host": "127.0.0.1", "port": port, "log": log, "startup_s": round(time.time() - t0, 1)}
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
