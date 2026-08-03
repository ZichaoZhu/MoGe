import torch

from tools.moge3.prepare_exp30_rois import (
    depth_boundary_score,
    select_crop,
)


def test_depth_boundary_crop_finds_a_discontinuity():
    points = torch.zeros(64, 96, 3)
    points[..., 2] = 1.0
    points[:, 64:, 2] = 3.0
    crop, score = select_crop(points, crop_size=32, grid_stride=8)
    assert crop[0] <= 64 <= crop[2]
    assert score > 0


def test_invalid_depth_does_not_create_false_edges():
    points = torch.zeros(32, 32, 3)
    points[..., 2] = float("nan")
    score = depth_boundary_score(points)
    assert torch.equal(score, torch.zeros_like(score))
