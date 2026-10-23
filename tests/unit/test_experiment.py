import json

import pytest

from clvla.config import parse_experiment_config
from clvla.experiment import (
    REQUIRED_METADATA,
    build_run_metadata,
    checkpoint_identity,
    git_revision,
    make_run_dir,
    package_versions,
    write_run_metadata,
)
from clvla.tasks import parse_manifest


@pytest.fixture
def cfg_and_manifest(experiment_raw, manifest_raw, fake_env, repo_root):
    cfg = parse_experiment_config(experiment_raw, str(repo_root / "configs/experiment.yaml"), env=fake_env)
    m = parse_manifest(manifest_raw, str(repo_root / "configs/task_manifest.yaml"))
    return cfg, m


def test_run_dirs_are_separated_by_kind_and_never_overwritten(tmp_path):
    a = make_run_dir(tmp_path, "baseline", "pi05", stamp="20260101T000000Z")
    b = make_run_dir(tmp_path, "continual", "pi05", stamp="20260101T000000Z")
    assert a.parent.name == "baseline" and b.parent.name == "continual" and a != b
    with pytest.raises(FileExistsError):
        make_run_dir(tmp_path, "baseline", "pi05", stamp="20260101T000000Z")
    with pytest.raises(ValueError, match="kind"):
        make_run_dir(tmp_path, "misc", "x")
    with pytest.raises(ValueError, match="name"):
        make_run_dir(tmp_path, "baseline", "../escape")


def test_metadata_contains_everything_needed_to_reproduce(cfg_and_manifest, tmp_path, repo_root):
    cfg, m = cfg_and_manifest
    lock = {"params/_METADATA": {"size": 2, "sha256": "a" * 64}}
    ck = checkpoint_identity(cfg, lock, "f" * 64, "sizes")
    meta = build_run_metadata(
        run_id="r1",
        kind="baseline",
        config=cfg,
        manifest=m,
        checkpoint=ck,
        repo_root=repo_root,
        env_info={"python": "3.x", "packages": {"numpy": "1"}},
        code={"project": {"commit": "abc", "dirty": False}},
    )
    for k in REQUIRED_METADATA:
        assert meta[k], k
    assert meta["protocol_sha256"] == cfg.protocol_sha256
    assert meta["task_manifest"]["content_sha256"] == m.content_sha256
    assert meta["task_manifest"]["task_ids"][0] == m.tasks[0].task_id
    assert meta["seeds"] == {"global": 7, "eval": [7], "env_seed": 7}
    assert meta["checkpoint"]["fingerprint_sha256"] == "f" * 64 and meta["checkpoint"]["openpi_config"] == "pi05_libero"
    assert meta["checkpoint"]["uri"] == "gs://openpi-assets/checkpoints/pi05_libero"

    run_dir = make_run_dir(tmp_path, "baseline", "x")
    write_run_metadata(run_dir, meta, str(repo_root / "configs/experiment.yaml"), m.path)
    saved = json.loads((run_dir / "run_metadata.json").read_text())
    assert saved["run_id"] == "r1"
    assert (run_dir / "config.snapshot.yaml").read_bytes() == (repo_root / "configs/experiment.yaml").read_bytes()
    assert (run_dir / "task_manifest.snapshot.yaml").read_bytes() == (
        repo_root / "configs/task_manifest.yaml"
    ).read_bytes()


def test_incomplete_metadata_rejected(cfg_and_manifest, repo_root):
    cfg, m = cfg_and_manifest
    with pytest.raises(ValueError, match="checkpoint"):
        build_run_metadata(
            run_id="r",
            kind="baseline",
            config=cfg,
            manifest=m,
            checkpoint={},
            repo_root=repo_root,
            env_info={"python": "3"},
            code={"x": 1},
        )


def test_real_environment_probes(tmp_path):
    v = package_versions(["numpy", "definitely-not-a-package"])
    assert v["numpy"] and v["definitely-not-a-package"] is None
    rev = git_revision(tmp_path)  # not a repo: reported, not raised
    assert rev["commit"] is None and "error" in rev
