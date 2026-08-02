import numpy as np
import pytest

from tools.moge3.visualize_split_rods import (
    build_display_mask,
    masked_relative_metrics,
    normalize_manifest,
    parse_series,
    selection_entries,
    source_rgb_with_crop,
)


def test_normalize_legacy_single_batch_manifest():
    manifest = {
        "scene": "ai_001_001",
        "camera": "cam_00",
        "M_cam_from_uv": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
        "samples": [{"frame": 0, "rgb": {}, "depth": {}}],
    }
    normalized = normalize_manifest(manifest)
    sample = normalized["samples"][0]
    assert sample["id"] == "ai_001_001_cam_00_frame.0000"
    assert sample["split"] == "train"
    assert sample["scene"] == "ai_001_001"


def test_selection_entries_reject_cross_split_sample():
    manifest = {
        "samples": [
            {
                "id": "sample",
                "split": "val",
                "scene": "scene",
                "M_cam_from_uv": [],
            }
        ]
    }
    selection = {
        "status": "locked-before-rendering",
        "selection_inputs": "RGB and ground truth only; no predictions",
        "splits": {"test": {"id": "sample", "crop_xyxy": [0, 0, 4, 4]}},
    }
    with pytest.raises(ValueError, match="belongs to val"):
        selection_entries(selection, manifest, height=4, width=4)


def test_parse_series_requires_unique_labels_and_valid_k():
    parsed = parse_series([["A", "/checkpoint-a.pt", "1"]])
    assert parsed[0].label == "A"
    assert parsed[0].refinement_steps == 1
    with pytest.raises(ValueError, match="unique"):
        parse_series(
            [
                ["A", "/checkpoint-a.pt", "1"],
                ["A", "/checkpoint-b.pt", "3"],
            ]
        )


def test_masked_metrics_ignore_unselected_error():
    gt = np.ones((2, 2, 3), dtype=np.float32)
    pred = gt.copy()
    pred[0, 0, 2] = 1.5
    pred[1, 1] = 100
    mask = np.asarray([[True, False], [False, False]])
    metrics = masked_relative_metrics(pred, gt, mask)
    assert metrics["pixels"] == 1
    assert metrics["point_rel"] == pytest.approx(0.5)
    assert metrics["depth_rel"] == pytest.approx(0.5)


def test_near_quantile_mask_is_derived_from_ground_truth_only():
    gt = np.zeros((2, 2, 3), dtype=np.float32)
    gt[..., 2] = np.asarray([[1.0, 2.0], [3.0, 4.0]])
    mask, resolved = build_display_mask(
        gt,
        {"type": "near_quantile", "quantile": 0.5},
    )
    assert mask.tolist() == [[True, True], [False, False]]
    assert resolved["resolved_maximum_depth_m"] == pytest.approx(2.5)


def test_source_rgb_with_crop_draws_only_the_border():
    image = np.zeros((12, 16, 3), dtype=np.float32)
    marked = source_rgb_with_crop(image, (3, 2, 13, 10))
    assert np.allclose(marked[2, 3], np.asarray([1.0, 0.08, 0.03]))
    assert np.allclose(marked[9, 12], np.asarray([1.0, 0.08, 0.03]))
    assert np.array_equal(marked[5, 8], np.zeros(3, dtype=np.float32))
    assert np.array_equal(marked[0, 0], np.zeros(3, dtype=np.float32))
