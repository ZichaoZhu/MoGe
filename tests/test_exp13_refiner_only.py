from __future__ import annotations

from tools.moge3.prepare_exp13_fine_structure_rois import (
    build_manifest,
    scaled_crop,
)


def test_scaled_crop_doubles_hypersim_ranking_coordinates():
    assert scaled_crop(
        {
            "crop_x0": "16",
            "crop_y0": "8",
            "crop_x1": "112",
            "crop_y1": "104",
        },
        source_width=256,
        source_height=192,
        width=512,
        height=384,
    ) == [32, 16, 224, 208]


def test_manifest_preserves_approved_train_and_locked_held_out_rois():
    samples = [
        {"id": "train-a", "split": "train"},
        {"id": "train-b", "split": "train"},
        {"id": "val-a", "split": "val"},
        {"id": "test-a", "split": "test"},
    ]
    candidates = {
        "candidates": [
            {
                "id": "train-a",
                "crop_x0": "0",
                "crop_y0": "0",
                "crop_x1": "96",
                "crop_y1": "96",
            }
        ]
    }
    held_out = {
        "train": {
            "id": "train-b",
            "crop_x0": 10,
            "crop_y0": 20,
            "crop_x1": 100,
            "crop_y1": 120,
        },
        "val": {
            "id": "val-a",
            "crop_x0": 10,
            "crop_y0": 20,
            "crop_x1": 100,
            "crop_y1": 120,
        },
        "test": {
            "id": "test-a",
            "crop_x0": 10,
            "crop_y0": 20,
            "crop_x1": 100,
            "crop_y1": 120,
        },
    }
    result = build_manifest(
        data_manifest={"samples": samples},
        candidates=candidates,
        approval={"status": "approved", "candidate_count": 1},
        held_out=held_out,
        width=512,
        height=384,
    )
    assert result["selection_inputs"] == "RGB and ground truth only; no predictions"
    assert result["counts"] == {"train": 2, "val": 1, "test": 1}
    assert len(result["entries"]) == 4
