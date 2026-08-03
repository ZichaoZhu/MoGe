import numpy as np
import pytest

from tools.moge3.export_exp9_pointclouds import (
    apply_scale_z_shift,
    read_binary_ply,
    resolve_asset,
    voxel_depth_bins,
    write_binary_ply,
)


def test_binary_ply_round_trip_preserves_raster_order_and_rgb(tmp_path):
    height, width = 3, 4
    points = np.arange(height * width * 3, dtype=np.float32).reshape(
        height, width, 3
    )
    points[..., 2] += 1.0
    colors = np.arange(height * width * 3, dtype=np.uint8).reshape(
        height, width, 3
    )
    output = tmp_path / "pointmap.ply"
    write_binary_ply(
        output,
        points,
        colors,
        width=width,
        height=height,
    )
    restored_points, restored_colors, comments = read_binary_ply(output)
    np.testing.assert_array_equal(restored_points, points.reshape(-1, 3))
    np.testing.assert_array_equal(restored_colors, colors.reshape(-1, 3))
    assert comments["raster_width"] == str(width)
    assert comments["raster_height"] == str(height)
    assert comments["vertex_order"] == "row_major_one_vertex_per_pixel"


def test_binary_ply_rejects_nonfinite_points(tmp_path):
    points = np.ones((2, 2, 3), dtype=np.float32)
    points[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="NaN or infinity"):
        write_binary_ply(
            tmp_path / "invalid.ply",
            points,
            np.zeros((2, 2, 3), dtype=np.uint8),
            width=2,
            height=2,
        )


def test_binary_ply_can_preserve_sparse_invalid_gt_pixels(tmp_path):
    points = np.ones((2, 2, 3), dtype=np.float32)
    points[0, 1] = np.nan
    output = tmp_path / "ground_truth.ply"
    write_binary_ply(
        output,
        points,
        np.zeros((2, 2, 3), dtype=np.uint8),
        width=2,
        height=2,
        allow_nonfinite=True,
    )
    restored, _, _ = read_binary_ply(output)
    np.testing.assert_allclose(
        restored,
        points.reshape(-1, 3),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )


def test_alignment_matches_paper_scale_and_optical_axis_shift():
    points = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    aligned = apply_scale_z_shift(points, scale=2.0, z_shift=-0.5)
    np.testing.assert_allclose(aligned, [[[2.0, 4.0, 5.5]]])


def test_voxel_depth_bins_use_round_D_log_Z():
    depths = np.exp(np.array([0.0, 0.5, -0.5], dtype=np.float32))
    points = np.zeros((1, 3, 3), dtype=np.float32)
    points[..., 2] = depths
    np.testing.assert_array_equal(voxel_depth_bins(points), [[0, 100, -100]])


def test_voxel_depth_bins_reject_nonpositive_depth():
    points = np.ones((1, 1, 3), dtype=np.float32)
    points[..., 2] = 0.0
    with pytest.raises(ValueError, match="positive Z"):
        voxel_depth_bins(points)


def test_initial_k_aliases_resolve_to_the_single_k0_asset():
    assets = {
        "0": {"url": "/data/sample/initial_k0.ply", "pointCount": 8},
        "1": {"alias": "initial.0"},
        "3": {"alias": "initial.0"},
        "5": {"alias": "initial.0"},
    }
    assert resolve_asset(assets, 0)["pointCount"] == 8
    for step in (1, 3, 5):
        assert resolve_asset(assets, step)["url"].endswith("initial_k0.ply")
