from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

from tools.moge3.export_viewer_experiment import (
    batch_norm_state_sha256,
    batch_norm_states_equal,
    depth_edge_crop,
    state_dict_sha256,
    validate_selection,
)


def test_selection_requires_five_unique_samples_per_split():
    selection = {
        "version": 1,
        "experiment": "exp20_stateless_ssr_batch_statistics",
        "splits": {
            split: [
                {
                    "id": f"{split}-{index}",
                    "picture": index,
                    "rank": index,
                    "total": 16,
                }
                for index in range(1, 6)
            ]
            for split in ("train", "val", "test")
        },
    }
    validate_selection(
        selection,
        experiment="exp20_stateless_ssr_batch_statistics",
        splits=("train", "val", "test"),
    )
    selection["splits"]["test"][4]["id"] = "train-1"
    with pytest.raises(ValueError, match="Duplicate"):
        validate_selection(
            selection,
            experiment="exp20_stateless_ssr_batch_statistics",
            splits=("train", "val", "test"),
        )


def test_depth_edge_crop_finds_densest_valid_window_deterministically():
    points = np.zeros((384, 512, 3), dtype=np.float32)
    points[..., 2] = 2.0
    points[120:280, 300:304, 2] = 4.0
    crop = depth_edge_crop(points, size=192, stride=16)
    assert crop == [128, 96, 320, 288]


def test_depth_edge_crop_avoids_mostly_invalid_regions():
    points = np.full((64, 96, 3), np.nan, dtype=np.float32)
    points[16:, 32:, :] = 0.0
    points[16:, 32:, 2] = 2.0
    points[24:48, 55:57, 2] = 4.0
    crop = depth_edge_crop(points, size=32, stride=8)
    x0, y0, x1, y1 = crop
    valid = np.isfinite(points[y0:y1, x0:x1]).all(axis=-1)
    assert valid.mean() >= 0.75


def test_batch_norm_state_hash_detects_mutation():
    state = {
        "block": {
            "running_mean": torch.zeros(3),
            "running_var": torch.ones(3),
            "num_batches_tracked": torch.tensor(2),
        }
    }
    same = copy.deepcopy(state)
    changed = copy.deepcopy(state)
    changed["block"]["running_mean"][0] = 1
    assert batch_norm_states_equal(state, same)
    assert batch_norm_state_sha256(state) == batch_norm_state_sha256(same)
    assert not batch_norm_states_equal(state, changed)
    assert batch_norm_state_sha256(state) != batch_norm_state_sha256(changed)


def test_model_state_hash_is_order_independent_and_detects_mutation():
    state = {
        "weight": torch.arange(6, dtype=torch.float32).reshape(2, 3),
        "counter": torch.tensor(2, dtype=torch.int64),
    }
    reordered = {
        "counter": state["counter"].clone(),
        "weight": state["weight"].clone(),
    }
    changed = copy.deepcopy(state)
    changed["weight"][0, 0] = -1
    assert state_dict_sha256(state) == state_dict_sha256(reordered)
    assert state_dict_sha256(state) != state_dict_sha256(changed)
