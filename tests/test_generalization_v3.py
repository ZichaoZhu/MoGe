from types import SimpleNamespace

import pytest
import torch

from moge.scripts.train_hypersim_generalization_v3 import (
    base_state_digest,
    changes_from_k0,
    configure_ssr_only,
    stratified_subset,
    verify_cached_base_detached,
    verify_optimizer_ssr_only,
    validate_refinement_configuration,
)
from tools.moge3.prepare_hypersim_generalization import (
    scene_group,
    validate_disjoint_scene_groups,
)
from tools.moge3.summarize_ssr_stability_ablation import (
    build_row,
    verify_common_baseline,
)


def test_scene_group_and_split_leakage_detection():
    assert scene_group("ai_036_002") == "ai_036"
    valid = {
        "splits": {
            "train": [{"scene": "ai_001_001"}],
            "val": [{"scene": "ai_002_001"}],
            "test": [{"scene": "ai_003_001"}],
        }
    }
    validate_disjoint_scene_groups(valid)

    same_split_group = {
        "splits": {
            "train": [
                {"scene": "ai_001_001"},
                {"scene": "ai_001_002"},
            ],
            "val": [{"scene": "ai_002_001"}],
            "test": [{"scene": "ai_003_001"}],
        }
    }
    validate_disjoint_scene_groups(same_split_group)

    leaked = {
        "splits": {
            "train": [{"scene": "ai_001_001"}],
            "val": [{"scene": "ai_001_002"}],
            "test": [{"scene": "ai_003_001"}],
        }
    }
    with pytest.raises(ValueError, match="leaks"):
        validate_disjoint_scene_groups(leaked)


def test_periodic_train_subset_is_scene_balanced_and_temporally_spread():
    samples = [
        SimpleNamespace(scene=scene, frame=frame)
        for scene in ("a", "b", "c")
        for frame in range(8)
    ]
    selected = stratified_subset(samples, 6)
    assert len(selected) == 6
    for scene in ("a", "b", "c"):
        frames = [sample.frame for sample in selected if sample.scene == scene]
        assert frames == [0, 7]


def test_k3_changes_use_error_reduction_and_accuracy_difference():
    metrics = {
        "0": {
            split: {
                "point_rel": 0.10,
                "depth_rel": 0.20,
                "depth_delta_1.01": 0.30,
                "depth_delta_1.25": 0.80,
                "boundary_f1": 0.60,
            }
            for split in ("train", "val", "test")
        },
        "3": {
            split: {
                "point_rel": 0.08,
                "depth_rel": 0.10,
                "depth_delta_1.01": 0.40,
                "depth_delta_1.25": 0.85,
                "boundary_f1": 0.63,
            }
            for split in ("train", "val", "test")
        },
    }
    changes = changes_from_k0(metrics, 3)
    assert changes["test"] == pytest.approx(
        {
            "point_rel_reduction": 0.2,
            "depth_rel_reduction": 0.5,
            "depth_delta_1.01_change": 0.1,
            "depth_delta_1.25_change": 0.05,
            "boundary_f1_change": 0.03,
        }
    )


def test_refinement_configuration_supports_k1_and_paper_k3_training():
    validate_refinement_configuration(1, [0, 1, 3, 5])
    validate_refinement_configuration(3, [0, 1, 3, 5])

    with pytest.raises(ValueError, match="training refinement step"):
        validate_refinement_configuration(2, [0, 1, 3, 5])
    with pytest.raises(ValueError, match="sorted and unique"):
        validate_refinement_configuration(1, [0, 3, 1, 5])


class _TinySSRModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = torch.nn.Linear(3, 4)
        self.head = torch.nn.Linear(4, 3)
        self.ssr = torch.nn.Sequential(
            torch.nn.Linear(3, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
        )


def test_ssr_only_guard_freezes_base_and_optimizer_matches_ssr():
    model = _TinySSRModel()
    guard = configure_ssr_only(model)
    assert guard["base_trainable_parameter_tensors"] == 0
    assert guard["trainable_parameter_tensors"] == guard["ssr_parameter_tensors"]
    assert all(parameter.requires_grad for parameter in model.ssr.parameters())
    assert all(
        not parameter.requires_grad
        for name, parameter in model.named_parameters()
        if not name.startswith("ssr.")
    )

    optimizer = torch.optim.AdamW(model.ssr.parameters(), lr=1e-3)
    assert verify_optimizer_ssr_only(optimizer, model.ssr) == len(
        list(model.ssr.parameters())
    )

    wrong_optimizer = torch.optim.AdamW(
        list(model.ssr.parameters()) + list(model.head.parameters()),
        lr=1e-3,
    )
    with pytest.raises(RuntimeError, match="not exactly"):
        verify_optimizer_ssr_only(wrong_optimizer, model.ssr)


def test_base_digest_excludes_ssr_and_detects_base_changes():
    model = _TinySSRModel()
    initial = base_state_digest(model)
    with torch.no_grad():
        next(model.ssr.parameters()).add_(1)
    assert base_state_digest(model) == initial
    with torch.no_grad():
        next(model.encoder.parameters()).add_(1)
    assert base_state_digest(model) != initial


def test_cached_base_guard_rejects_autograd_tensors():
    detached = SimpleNamespace(
        sample_id="detached",
        base_points=torch.zeros(1),
        visual_features=torch.zeros(1),
    )
    verify_cached_base_detached([detached])

    attached = SimpleNamespace(
        sample_id="attached",
        base_points=torch.zeros(1, requires_grad=True),
        visual_features=torch.zeros(1),
    )
    with pytest.raises(RuntimeError, match="retains autograd"):
        verify_cached_base_detached([attached])


def _ablation_report(*, point_rel: float, boundary_f1: float):
    metrics = {
        "0": {
            split: {"point_rel": 0.10, "depth_rel": 0.08, "boundary_f1": 0.80}
            for split in ("train", "val", "test")
        },
        "1": {
            split: {
                "point_rel": point_rel,
                "depth_rel": point_rel - 0.01,
                "boundary_f1": boundary_f1,
            }
            for split in ("train", "val", "test")
        },
    }
    return {
        "paper_alignment": {"training_refinement_steps": 1},
        "loss_weights": {"edge": 4.0},
        "best_checkpoint_step": 250,
        "training_seconds_including_periodic_eval": 12.0,
        "peak_memory_bytes": 1024,
        "metrics_by_k": metrics,
    }


def test_ablation_summary_uses_training_k_and_checks_common_baseline():
    arm = {
        "id": "arm_b_k1_edge4",
        "training_refinement_steps": 1,
        "edge_weight": 4.0,
        "steps": 2000,
    }
    best = _ablation_report(point_rel=0.09, boundary_f1=0.798)
    latest = {
        **_ablation_report(point_rel=0.085, boundary_f1=0.79),
        "checkpoint_step": 2000,
    }
    row = build_row(arm, best, latest)
    assert row["best_val_point_rel_reduction"] == pytest.approx(0.1)
    assert row["latest_val_boundary_f1_change"] == pytest.approx(-0.01)
    verify_common_baseline([row, dict(row)])

    changed = dict(row)
    changed["latest_val_point_rel_k0"] = 0.11
    with pytest.raises(ValueError, match="differs across arms"):
        verify_common_baseline([row, changed])
