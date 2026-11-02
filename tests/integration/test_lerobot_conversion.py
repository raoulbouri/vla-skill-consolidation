"""LeRobot conversion end to end on a synthetic aligned file, read back by LeRobot and OpenPI (policy env; CPU)."""

import json

import numpy as np
import pytest

pytest.importorskip("lerobot", reason="policy env only")

from clvla.lerobot_convert import (  # noqa: E402
    dataset_fingerprint,
    repo_ids,
    split_demo_keys,
    verify_lerobot_dataset,
    write_lerobot_dataset,
)

pytestmark = [pytest.mark.policy_env, pytest.mark.slow]
INSTR = "pick up the book and place it in the front compartment of the caddy"


def _processed_file(path, n_demos=6, T=12):
    import h5py

    rng = np.random.default_rng(0)
    with h5py.File(str(path), "w") as f:
        d = f.create_group("data")
        d.attrs["alignment"] = "pre_action: obs[k] = render(states[k]) paired with actions[k]"
        d.attrs["problem_info"] = json.dumps({"language_instruction": INSTR})
        for i in range(n_demos):
            g = d.create_group(f"demo_{i}")
            g.create_dataset("obs/agentview_rgb", data=rng.integers(0, 255, (T, 256, 256, 3), dtype=np.uint8))
            g.create_dataset("obs/eye_in_hand_rgb", data=rng.integers(0, 255, (T, 256, 256, 3), dtype=np.uint8))
            g.create_dataset("obs/state", data=rng.normal(size=(T, 8)).astype(np.float32))
            g.create_dataset("actions", data=rng.uniform(-1, 1, (T, 7)))
    return path


def test_conversion_creates_isolated_train_val_datasets(tmp_path):
    import h5py
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    src = _processed_file(tmp_path / "t078_demo.hdf5")
    rid = repo_ids("clvla", "testver", 78)
    home = tmp_path / "lerobot"
    with h5py.File(str(src), "r") as f:
        split = split_demo_keys(list(f["data"].keys()), [0, 1, 2, 3], [4, 5])
        info = {}
        for part in ("train", "val"):
            root = home / rid[part]
            info[part] = write_lerobot_dataset(rid[part], root, [f["data"][k] for k in split[part]], INSTR, fps=10,
                                               robot_type="panda", threads=2, processes=0)
            verify_lerobot_dataset(rid[part], root, info[part]["num_episodes"], info[part]["num_frames"], INSTR)
        with pytest.raises(FileExistsError):
            write_lerobot_dataset(rid["train"], home / rid["train"], [], INSTR, fps=10, robot_type="panda",
                                  threads=0, processes=0)
        first_train_action = f["data"]["demo_0"]["actions"][0].astype(np.float32)
        first_train_image = f["data"]["demo_0"]["obs/agentview_rgb"][0]
    assert info["train"]["num_episodes"] == 4 and info["val"]["num_episodes"] == 2
    ds = LeRobotDataset(rid["train"], root=home / rid["train"], delta_timestamps={"actions": [t / 10 for t in range(10)]})
    item = ds[0]
    assert np.allclose(np.asarray(item["actions"][0]), first_train_action, atol=1e-6)
    img = (np.asarray(item["image"]).transpose(1, 2, 0) * 255).round().astype(np.uint8)  # LeRobot gives float CHW
    assert np.array_equal(img, first_train_image[::-1, ::-1])  # stored rotated 180, lossless PNG
    assert item["task"] == INSTR
    assert dataset_fingerprint(home / rid["train"]) == dataset_fingerprint(home / rid["train"])


def test_converted_sample_passes_openpi_training_transforms(tmp_path, cfg):
    """The converted format feeds OpenPI's pi05_libero chain (repack -> LiberoInputs -> Normalize -> model)."""
    pytest.importorskip("openpi")
    import h5py
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    from clvla.policy import openpi_training_transforms

    src = _processed_file(tmp_path / "t_demo.hdf5", n_demos=1)
    rid = repo_ids("clvla", "chain", 78)["train"]
    root = tmp_path / "lerobot" / rid
    with h5py.File(str(src), "r") as f:
        write_lerobot_dataset(rid, root, [f["data"]["demo_0"]], INSTR, fps=10, robot_type="panda", threads=0, processes=0)
    item = LeRobotDataset(rid, root=root, delta_timestamps={"actions": [t / 10 for t in range(10)]})[0]
    transform, dc, _ = openpi_training_transforms(cfg.checkpoint.openpi_config, cfg.checkpoint.local_dir)
    sample = {"image": item["image"], "wrist_image": item["wrist_image"], "state": item["state"],
              "actions": item["actions"], "prompt": item["task"]}
    out = transform({k: np.asarray(v) if k != "prompt" else v for k, v in sample.items()})
    assert out["image"]["base_0_rgb"].shape == (224, 224, 3) and out["actions"].shape == (10, 32)
