import pytest
import torch

from moge.model.ssr import (
    SelfGuidedSparseRefiner,
    SpconvSparseUNet,
    VoxelDepthSpanError,
    effective_voxel_depth_limit,
    factorize_points,
    gather_features_at_coordinates,
    sample_visual_features_for_voxels,
    smooth_bound_log_depth_residual,
    unfactorize_points,
    voxelize_factorized,
)


def test_smooth_log_depth_residual_bound_is_identity_near_zero_and_bounded():
    raw = torch.tensor(
        [-float("inf"), -1.0, -1e-6, 0.0, 1e-6, 1.0, float("inf")],
        requires_grad=True,
    )
    bounded = smooth_bound_log_depth_residual(raw, 0.1)
    assert torch.isneginf(bounded[0])
    assert torch.isposinf(bounded[-1])
    assert bounded[1:-1].abs().max() <= 0.1
    torch.testing.assert_close(
        bounded[2:5],
        raw[2:5],
        atol=1e-10,
        rtol=1e-6,
    )
    bounded[3].backward()
    assert raw.grad[3].item() == pytest.approx(1.0)


def test_disabled_smooth_log_depth_residual_bound_preserves_tensor():
    raw = torch.randn(2, 3)
    assert smooth_bound_log_depth_residual(raw, 0.0) is raw
    assert smooth_bound_log_depth_residual(raw, None) is raw
    with pytest.raises(ValueError, match="finite and non-negative"):
        smooth_bound_log_depth_residual(raw, float("nan"))


def _plane_points(batch: int = 1, height: int = 4, width: int = 5) -> torch.Tensor:
    rows, cols = torch.meshgrid(
        torch.linspace(-0.4, 0.4, height),
        torch.linspace(-0.5, 0.5, width),
        indexing="ij",
    )
    depth = torch.ones_like(rows) * 2.0
    points = torch.stack((cols * depth, rows * depth, depth), dim=-1)
    return points.unsqueeze(0).expand(batch, -1, -1, -1).clone()


def test_factorized_round_trip():
    points = _plane_points()
    reconstructed = unfactorize_points(factorize_points(points))
    torch.testing.assert_close(reconstructed, points)


def test_voxelization_has_one_input_voxel_per_pixel_and_round_trip_mapping():
    q = factorize_points(_plane_points(batch=2))
    shell = voxelize_factorized(q, voxel_resolution=200, num_downsamples=2)
    assert shell.coordinates.shape == (2 * 4 * 5, 4)
    assert torch.unique(shell.coordinates, dim=0).shape[0] == 2 * 4 * 5

    source_coordinates = shell.coordinates.flip(0)
    source_features = torch.arange(source_coordinates.shape[0])[:, None].float()
    reordered = gather_features_at_coordinates(
        source_features, source_coordinates, shell.coordinates, shell.spatial_shape
    )
    expected_hash = {
        tuple(coordinate.tolist()): value
        for coordinate, value in zip(source_coordinates, source_features)
    }
    expected = torch.stack(
        [expected_hash[tuple(coordinate.tolist())] for coordinate in shell.coordinates]
    )
    torch.testing.assert_close(reordered, expected)


def test_voxel_depth_span_is_rejected_before_sparse_construction():
    factorized = torch.zeros(1, 2, 2, 3)
    factorized[0, 1, 1, 2] = 3.0
    with pytest.raises(VoxelDepthSpanError, match="before sparse tensor"):
        voxelize_factorized(
            factorized,
            voxel_resolution=200,
            max_depth_span=512,
        )


def test_effective_depth_limit_never_rejects_the_base_shell():
    assert effective_voxel_depth_limit(
        512,
        torch.tensor([120, 908]),
        maximum_expansion=122,
    ) == 1030
    assert effective_voxel_depth_limit(
        512,
        torch.tensor([120, 300]),
        maximum_expansion=122,
    ) == 512


def test_depth_discontinuity_is_separated_in_voxel_space():
    points = _plane_points(height=2, width=2)
    points[:, :, 1, 2] *= 2.0
    points[:, :, 1, :2] *= 2.0
    shell = voxelize_factorized(factorize_points(points), voxel_resolution=200)
    logical = shell.logical_depth
    assert (logical[:, :, 1] - logical[:, :, 0]).abs().min() > 1


def test_reference_backend_is_identity_when_output_layer_is_zero():
    points = _plane_points(height=4, width=4)
    q = factorize_points(points)
    visual = torch.randn(1, 6, 2, 2)
    refiner = SelfGuidedSparseRefiner(
        visual_dim=6,
        voxel_resolution=20,
        channels=[4, 8],
        visual_channels=4,
        blocks_per_level=1,
        backend="reference",
    )
    residual, stats = refiner(q, visual)
    torch.testing.assert_close(residual, torch.zeros_like(residual))
    assert stats["active_voxels"].item() == 16
    assert stats["active_voxels_per_level"][0] == 16


def test_updated_depth_changes_next_shell():
    q = factorize_points(_plane_points(height=2, width=3))
    first = voxelize_factorized(q, voxel_resolution=20)
    update = torch.zeros_like(q[..., 2])
    update[..., 1] = 0.2
    updated_q = torch.cat((q[..., :2], q[..., 2:3] + update[..., None]), dim=-1)
    second = voxelize_factorized(updated_q, voxel_resolution=20)
    assert not torch.equal(first.logical_depth, second.logical_depth)


def test_visual_sampling_ignores_voxel_depth():
    visual = torch.arange(16).float().reshape(1, 1, 4, 4)
    coordinates = torch.tensor([[0, 0, 1, 2], [0, 9, 1, 2]], dtype=torch.int32)
    sampled = sample_visual_features_for_voxels(
        visual, coordinates, image_size=(4, 4), level_scale=1
    )
    torch.testing.assert_close(sampled[0], sampled[1])


def test_production_spconv_uses_mask_implicit_gemm_algorithm():
    pytest.importorskip("spconv.pytorch")
    from spconv.core import ConvAlgo

    model = SpconvSparseUNet(
        visual_dim=8,
        channels=(4, 8),
        visual_channels=4,
        blocks_per_level=1,
    )
    algorithms = {
        module.algo for module in model.modules() if hasattr(module, "algo")
    }
    assert algorithms == {ConvAlgo.MaskImplicitGemm}
