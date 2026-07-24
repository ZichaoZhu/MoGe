import torch

from moge.train.losses_v3 import (
    affine_invariant_global_loss_v3,
    grouped_weighted_median,
    radial_partition_local_loss,
)


def _points(height: int = 8, width: int = 8) -> torch.Tensor:
    rows, cols = torch.meshgrid(
        torch.linspace(-0.3, 0.3, height),
        torch.linspace(-0.4, 0.4, width),
        indexing="ij",
    )
    depth = 2.0 + 0.2 * rows + 0.1 * cols
    return torch.stack((cols * depth, rows * depth, depth), dim=-1).unsqueeze(0)


def test_grouped_weighted_median():
    values = torch.tensor([[0.0], [10.0], [20.0], [1.0], [3.0]])
    weights = torch.tensor([1.0, 5.0, 1.0, 1.0, 1.0])
    groups = torch.tensor([0, 0, 0, 1, 1])
    medians = grouped_weighted_median(values, weights, groups)
    torch.testing.assert_close(medians[:, 0], torch.tensor([10.0, 1.0]))


def test_global_loss_is_optical_axis_affine_invariant():
    gt = _points()
    transformed = gt * 1.7
    transformed[..., 2] += 0.4
    loss, _ = affine_invariant_global_loss_v3(transformed, gt, align_resolution=8)
    assert loss.item() < 1e-5


def test_local_loss_is_zero_for_identical_geometry():
    points = _points()
    _, alignment = affine_invariant_global_loss_v3(points, points, align_resolution=8)
    generator = torch.Generator().manual_seed(7)
    loss = radial_partition_local_loss(
        points, points, alignment, scales=(4,), generator=generator
    )
    assert loss.item() < 1e-6


def test_empty_ground_truth_is_finite():
    points = _points()
    gt = torch.full_like(points, torch.inf)
    loss, alignment = affine_invariant_global_loss_v3(points, gt, align_resolution=8)
    assert torch.isfinite(loss).all()
    assert not alignment.valid.any()
    local = radial_partition_local_loss(points, gt, alignment, scales=(4,))
    assert torch.isfinite(local).all()
    assert local.item() == 0
