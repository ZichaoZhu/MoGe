import torch

from moge.train.losses import edge_loss
from moge.train.losses_v3 import (
    affine_invariant_global_loss_v3,
    edge_angle_loss_v3,
    grouped_weighted_median,
    radial_partition_local_loss,
)
from moge.utils.alignment import align_points_scale_z_shift


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


def test_grouped_weighted_median_handles_large_prefix_and_tiny_later_group():
    values = torch.cat(
        (
            torch.arange(1000, dtype=torch.float32),
            torch.tensor([10.0, 20.0, 30.0]),
        )
    )
    weights = torch.cat(
        (
            torch.full((1000,), 1e5, dtype=torch.float32),
            torch.ones(3),
        )
    )
    groups = torch.cat(
        (
            torch.zeros(1000, dtype=torch.long),
            torch.ones(3, dtype=torch.long),
        )
    )
    medians = grouped_weighted_median(values, weights, groups)
    assert medians[1, 0].item() == 20.0


def test_global_loss_is_optical_axis_affine_invariant():
    gt = _points()
    transformed = gt * 1.7
    transformed[..., 2] += 0.4
    loss, _ = affine_invariant_global_loss_v3(transformed, gt, align_resolution=8)
    assert loss.item() < 1e-5


def test_scale_z_shift_alignment_is_invariant_to_anchor_chunk_size():
    generator = torch.Generator().manual_seed(19)
    source = torch.randn((2, 16, 3), generator=generator)
    source[..., 2] += 3.0
    target = torch.stack((1.4 * source[0], 0.8 * source[1]))
    target[0, :, 2] += 0.35
    target[1, :, 2] -= 0.2
    weight = torch.ones((2, 16))

    scale_chunked, shift_chunked = align_points_scale_z_shift(
        source,
        target,
        weight,
        trunc=1.0,
        max_anchor_elements=3 * source.shape[1],
    )
    scale_full, shift_full = align_points_scale_z_shift(
        source,
        target,
        weight,
        trunc=1.0,
        max_anchor_elements=10_000,
    )

    torch.testing.assert_close(scale_chunked, scale_full)
    torch.testing.assert_close(shift_chunked, shift_full)


def test_global_loss_does_not_differentiate_through_solved_alignment():
    gt = _points()
    pred = (1.7 * gt).clone()
    pred[..., 2] += 0.4
    pred.requires_grad_()
    loss, alignment = affine_invariant_global_loss_v3(
        pred,
        gt,
        align_resolution=8,
    )
    alignment_gradient = torch.autograd.grad(
        loss.sum(),
        alignment.scale,
        retain_graph=True,
        allow_unused=True,
    )[0]
    assert alignment_gradient is None
    pred_gradient = torch.autograd.grad(loss.sum(), pred)[0]
    assert torch.isfinite(pred_gradient).all()


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


def test_v3_edge_loss_uses_paper_short_dimension_normalization():
    generator = torch.Generator().manual_seed(101)
    pred = torch.randn(1, 12, 20, 3, generator=generator)
    gt = torch.randn(1, 12, 20, 3, generator=generator)
    gt[..., 2].abs_().add_(1)

    legacy, _ = edge_loss(pred, gt)
    paper, _ = edge_angle_loss_v3(pred, gt)

    torch.testing.assert_close(paper, legacy * (20 / 12))
