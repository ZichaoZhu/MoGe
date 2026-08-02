import numpy as np

from tools.moge3.visualize_fine_structure import (
    paper_coarse_fine_mask,
    select_crop,
    selection_description,
)


def test_paper_coarse_mask_detects_thin_near_structure():
    depth = np.full((64, 64), 10.0, dtype=np.float32)
    depth[:, 30:33] = 2.0
    valid = np.ones_like(depth, dtype=bool)
    mask = paper_coarse_fine_mask(depth, valid)
    assert mask[:, 29:34].sum() > 64
    assert mask[:, :10].mean() < 0.1


def test_improvement_crop_selects_changed_fine_region():
    fine = np.zeros((64, 64), dtype=bool)
    fine[32:48, 32:48] = True
    before = np.zeros((64, 64), dtype=np.float32)
    after = np.zeros((64, 64), dtype=np.float32)
    before[32:48, 32:48] = 0.2
    after[32:48, 32:48] = 0.05
    crop, score, count = select_crop(
        fine,
        before,
        after,
        crop_height=32,
        crop_width=32,
        stride=16,
        min_fine_pixels=16,
        mode="improvement",
    )
    x0, y0, x1, y1 = crop
    assert x0 <= 32 < x1
    assert y0 <= 32 < y1
    assert score > 0
    assert count == 256


def test_posthoc_rgb_gt_selection_is_not_reported_as_pretraining_locked():
    description = selection_description(
        "test",
        has_fixed_sample=True,
        provenance="posthoc_rgb_gt_only",
    )
    assert "post-hoc" in description
    assert "RGB/ground truth only" in description
    assert "predictions were not used" in description
    assert "before training" not in description


def test_training_visualization_discloses_prediction_based_crop():
    description = selection_description(
        "train",
        has_fixed_sample=True,
        provenance="posthoc_rgb_gt_only",
    )
    assert "training illustration" in description
    assert "K=0 to K=3 point-Rel reduction" in description
