from tools.moge3.prepare_combined_hypersim import merge_manifests


def _sample(identifier: str, split: str, group: str):
    return {
        "id": identifier,
        "split": split,
        "scene": f"{group}_001",
        "scene_group": group,
        "frame": 0,
        "rgb": {"file": f"{identifier}.jpg", "sha256": "a"},
        "depth": {"file": f"{identifier}.hdf5", "sha256": "b"},
    }


def test_exp10_manifest_combines_two_training_sets_without_tartanair():
    exp2 = {
        "format": "moge3-hypersim-smallset-v1",
        "samples": [
            {
                key: value
                for key, value in _sample(
                    f"exp2-{index}", "train", f"ai_{index:03d}"
                ).items()
                if key != "scene_group"
            }
            for index in range(24)
        ],
    }
    exp7 = {
        "format": "moge3-hypersim-generalization-v1",
        "samples": [
            *[
                _sample(f"exp7-{index}", "train", f"ai_{index + 24:03d}")
                for index in range(24)
            ],
            *[
                _sample(f"val-{index}", "val", f"ai_{index + 48:03d}")
                for index in range(16)
            ],
            *[
                _sample(f"test-{index}", "test", f"ai_{index + 64:03d}")
                for index in range(16)
            ],
        ],
    }
    merged = merge_manifests(exp2, exp7)
    assert merged["counts"] == {"train": 48, "val": 16, "test": 16}
    assert merged["provenance"]["tartanair_used"] is False
    assert sum(
        sample.get("training_source") == "exp2_trajectory_24"
        for sample in merged["samples"]
    ) == 24
    assert sum(
        sample.get("training_source") == "exp7_curated_24"
        for sample in merged["samples"]
    ) == 24
    assert all("scene_group" in sample for sample in merged["samples"])
