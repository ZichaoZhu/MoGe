import pytest
import torch

from moge.scripts.train_hypersim_smallset_v3 import (
    aggregate_metrics,
    boundary_f1,
    depth_edge_map,
)


def test_depth_edge_map_detects_depth_discontinuity():
    depth = torch.ones(8, 8)
    depth[:, 4:] = 2
    valid = torch.ones_like(depth, dtype=torch.bool)
    edges = depth_edge_map(depth, valid, threshold=0.03)
    assert edges[:, 3:5].all()
    assert not edges[:, :2].any()
    assert not edges[:, 6:].any()


def test_boundary_f1_is_one_for_identical_depth():
    depth = torch.ones(8, 8)
    depth[:, 4:] = 2
    valid = torch.ones_like(depth, dtype=torch.bool)
    assert boundary_f1(depth, depth, valid, threshold=0.03) == 1.0


def test_boundary_f1_is_one_when_both_maps_have_no_edges():
    depth = torch.ones(8, 8)
    valid = torch.ones_like(depth, dtype=torch.bool)
    assert boundary_f1(depth, depth, valid, threshold=0.03) == 1.0


def test_aggregate_metrics_averages_records():
    records = [
        {
            "point_rel": 0.1,
            "depth_rel": 0.2,
            "depth_delta_1.01": 0.3,
            "depth_delta_1.25": 0.4,
            "boundary_f1": 0.5,
        },
        {
            "point_rel": 0.3,
            "depth_rel": 0.4,
            "depth_delta_1.01": 0.5,
            "depth_delta_1.25": 0.6,
            "boundary_f1": 0.7,
        },
    ]
    result = aggregate_metrics(records)
    assert result == pytest.approx({
        "point_rel": 0.2,
        "depth_rel": 0.3,
        "depth_delta_1.01": 0.4,
        "depth_delta_1.25": 0.5,
        "boundary_f1": 0.6,
    })
