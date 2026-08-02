from __future__ import annotations

import pytest

from tools.moge3.prepare_exp11_hypersim import (
    approved_candidates,
    assemble_descriptors,
)


def _sample(index: int, split: str, group: int) -> dict[str, object]:
    scene = f"ai_{group:03d}_{index:03d}"
    return {
        "id": f"{scene}_cam_00_frame.0000",
        "split": split,
        "scene": scene,
        "camera": "cam_00",
        "frame": 0,
    }


def _manifest() -> dict[str, object]:
    samples = [
        *[_sample(index, "train", 200 + index) for index in range(1, 49)],
        *[_sample(index, "val", 43) for index in range(49, 65)],
        *[_sample(index, "test", 50) for index in range(65, 81)],
    ]
    return {
        "format": "moge3-hypersim-generalization-v1",
        "counts": {"train": 48, "val": 16, "test": 16},
        "samples": samples,
    }


def _candidates() -> list[dict[str, object]]:
    candidates = []
    for offset in range(52):
        scene = f"ai_{100 + offset:03d}_001"
        candidates.append(
            {
                "id": f"{scene}_cam_00_frame.0000",
                "scene": scene,
                "camera": "cam_00",
                "frame": "0",
                "display_id": f"N{offset + 1:02d}",
                "source_rank": offset + 1,
                "crop_x0": "0",
                "crop_y0": "0",
                "crop_x1": "96",
                "crop_y1": "96",
            }
        )
    return candidates


def test_approval_is_bound_to_exact_candidate_manifest() -> None:
    candidates = approved_candidates(
        {"candidates": _candidates()},
        {
            "status": "approved",
            "candidate_count": 52,
            "candidate_sha256": "abc",
        },
        candidate_sha256="abc",
    )
    assert len(candidates) == 52

    with pytest.raises(ValueError, match="does not match"):
        approved_candidates(
            {"candidates": _candidates()},
            {
                "status": "approved",
                "candidate_count": 52,
                "candidate_sha256": "old",
            },
            candidate_sha256="new",
        )


def test_exp11_descriptors_are_100_16_16_without_group_leakage() -> None:
    descriptors, groups = assemble_descriptors(_manifest(), _candidates())
    assert sum(item["split"] == "train" for item in descriptors) == 100
    assert sum(item["split"] == "val" for item in descriptors) == 16
    assert sum(item["split"] == "test" for item in descriptors) == 16
    assert set(groups["train"]).isdisjoint(groups["val"])
    assert set(groups["train"]).isdisjoint(groups["test"])


def test_exp11_rejects_candidate_in_held_out_scene_group() -> None:
    candidates = _candidates()
    candidates[0] = {
        **candidates[0],
        "id": "ai_043_999_cam_00_frame.0000",
        "scene": "ai_043_999",
    }
    with pytest.raises(ValueError, match="leaks"):
        assemble_descriptors(_manifest(), candidates)
