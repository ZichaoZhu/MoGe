from __future__ import annotations

from tools.moge3.prepare_exp11_candidate_review import (
    apply_curation,
    eligible_candidates,
    select_candidates,
)


def _row(rank: int, group: int, scene: int) -> dict[str, str]:
    scene_name = f"ai_{group:03d}_{scene:03d}"
    return {
        "id": f"{scene_name}_cam_00_frame.{rank:04d}",
        "scene": scene_name,
        "camera": "cam_00",
        "frame": str(rank),
        "crop_x0": "0",
        "crop_y0": "0",
        "crop_x1": "96",
        "crop_y1": "96",
        "crop_thin_pixels": str(1000 - rank),
        "crop_thin_near_pixels": "500",
        "crop_thin_far_pixels": "500",
        "rgb_source": f"/nas1/datasets/hypersim/raw/{scene_name}/rgb.jpg",
        "depth_source": f"/nas1/datasets/hypersim/raw/{scene_name}/depth.hdf5",
    }


def test_selection_excludes_existing_scenes_and_held_out_groups() -> None:
    rows = [
        _row(1, 1, 1),
        _row(2, 43, 1),
        _row(3, 2, 1),
        _row(4, 3, 1),
    ]
    manifest = {
        "samples": [{"id": rows[0]["id"], "scene": "ai_001_001"}],
        "scene_groups": {"val": ["ai_043"], "test": ["ai_050"]},
    }

    selected, exclusions = select_candidates(
        rows,
        manifest,
        count=2,
        max_per_scene_group=1,
    )

    assert [candidate["scene"] for candidate in selected] == [
        "ai_002_001",
        "ai_003_001",
    ]
    assert exclusions["held_out_scene_groups"] == ["ai_043", "ai_050"]


def test_selection_first_maximizes_group_diversity() -> None:
    rows = [
        _row(1, 1, 1),
        _row(2, 1, 2),
        _row(3, 2, 1),
        _row(4, 3, 1),
    ]
    manifest = {
        "samples": [],
        "scene_groups": {"val": [], "test": []},
    }

    selected, _ = select_candidates(
        rows,
        manifest,
        count=4,
        max_per_scene_group=2,
    )

    assert [candidate["scene_group"] for candidate in selected[:3]] == [
        "ai_001",
        "ai_002",
        "ai_003",
    ]
    assert selected[3]["scene"] == "ai_001_002"
    assert len({candidate["scene"] for candidate in selected}) == 4


def test_manual_curation_replaces_only_with_eligible_distinct_scene() -> None:
    rows = [
        _row(1, 1, 1),
        _row(2, 2, 1),
        _row(3, 3, 1),
    ]
    manifest = {
        "samples": [],
        "scene_groups": {"val": [], "test": []},
    }
    selected, _ = select_candidates(
        rows,
        manifest,
        count=2,
        max_per_scene_group=1,
    )
    eligible, _ = eligible_candidates(rows, manifest)

    curated = apply_curation(
        selected,
        eligible,
        {
            "replacements": [
                {
                    "remove": rows[0]["id"],
                    "add": rows[2]["id"],
                    "reason": "more explicit geometry",
                }
            ]
        },
        max_per_scene_group=1,
    )

    assert [candidate["id"] for candidate in curated] == [
        rows[2]["id"],
        rows[1]["id"],
    ]
    assert curated[0]["curation_replaced"] == rows[0]["id"]
