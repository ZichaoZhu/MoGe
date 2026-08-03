import argparse

import pytest
import torch

from moge.scripts.train_hypersim_joint_v3 import (
    aggregate_evaluation,
    backbone_learning_rate,
    base_parameters_are_frozen,
    clear_backbone_gradients,
    effective_training_stage,
    flatten_periodic_evaluation,
    learning_rate_scale,
    periodic_selection_score,
    preclip_gradient_action,
    refinement_has_collapsed,
    save_training_plot,
    selection_metric_key,
    set_refiner_only_trainable,
    shard_batch_indices,
    skipped_preclip_state,
    training_stage,
    validate_distributed_batch,
    validate_joint_schedule,
    validate_resume_histories,
)


def _args(**overrides):
    values = {
        "steps": 2500,
        "batch_size": 2,
        "microbatch_size": 1,
        "refinement_steps": 3,
        "backbone_freeze_steps": 100,
        "backbone_warmup_end": 200,
        "refiner_detach_steps": 500,
        "eval_every": 250,
        "full_eval_every": 0,
        "periodic_train_samples": 64,
        "ssr_learning_rate": 2e-5,
        "head_learning_rate": 1e-5,
        "backbone_learning_rate": 5e-7,
        "learning_rate_schedule": "constant",
        "learning_rate_decay_start_step": 0,
        "learning_rate_decay_end_step": 0,
        "learning_rate_final_scale": 1.0,
        "freeze_backbone": False,
        "train_refiner_only": False,
        "fine_structure_rois": None,
        "selection_scope": "full",
        "max_preclip_grad_norm": 0.0,
        "max_abs_log_depth_residual": 0.0,
        "smooth_log_depth_residual_bound": 0.0,
        "max_abs_raw_log_depth_residual": 0.0,
        "raw_residual_warning_threshold": 0.0,
        "raw_residual_tail_threshold": 0.0,
        "raw_residual_tail_weight": 0.0,
        "max_bound_saturation_fraction": 0.0,
        "max_consecutive_saturated_steps": 0,
        "max_voxel_depth_span": 0,
        "max_refined_point_rel": 0.0,
        "max_refined_to_base_ratio": 0.0,
        "max_base_to_best_ratio": 0.0,
        "max_skipped_preclip_steps": 0,
        "max_consecutive_skipped_preclip_steps": 0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_scaled_two_stage_schedule_is_valid():
    validate_joint_schedule(_args())
    with pytest.raises(ValueError, match="freeze < warmup"):
        validate_joint_schedule(_args(backbone_warmup_end=600))
    validate_joint_schedule(
        _args(
            steps=3004,
            backbone_freeze_steps=1000,
            backbone_warmup_end=2000,
            refiner_detach_steps=5000,
        )
    )
    with pytest.raises(ValueError, match="Microbatch"):
        validate_joint_schedule(_args(microbatch_size=3))
    validate_joint_schedule(
        _args(freeze_backbone=True, backbone_learning_rate=0)
    )
    with pytest.raises(ValueError, match="Frozen backbone"):
        validate_joint_schedule(_args(freeze_backbone=True))
    validate_joint_schedule(
        _args(
            train_refiner_only=True,
            head_learning_rate=0,
            backbone_learning_rate=0,
        )
    )
    with pytest.raises(ValueError, match="head and backbone"):
        validate_joint_schedule(_args(train_refiner_only=True))
    with pytest.raises(ValueError, match="fine-structure"):
        validate_joint_schedule(
            _args(
                train_refiner_only=True,
                head_learning_rate=0,
                backbone_learning_rate=0,
                selection_scope="structure",
            )
        )
    validate_joint_schedule(
        _args(
            max_skipped_preclip_steps=5,
            max_consecutive_skipped_preclip_steps=2,
        )
    )
    with pytest.raises(ValueError, match="Consecutive"):
        validate_joint_schedule(
            _args(max_consecutive_skipped_preclip_steps=1)
        )
    with pytest.raises(ValueError, match="between 1"):
        validate_joint_schedule(
            _args(
                max_skipped_preclip_steps=2,
                max_consecutive_skipped_preclip_steps=3,
            )
        )
    validate_joint_schedule(
        _args(
            learning_rate_schedule="cosine",
            learning_rate_decay_start_step=500,
            learning_rate_decay_end_step=2500,
            learning_rate_final_scale=0.1,
        )
    )
    with pytest.raises(ValueError, match="Cosine decay"):
        validate_joint_schedule(
            _args(
                learning_rate_schedule="cosine",
                learning_rate_decay_start_step=2500,
                learning_rate_decay_end_step=2500,
            )
        )
    with pytest.raises(ValueError, match="Final learning-rate scale"):
        validate_joint_schedule(
            _args(learning_rate_final_scale=0.0)
        )
    validate_joint_schedule(
        _args(smooth_log_depth_residual_bound=0.1)
    )
    with pytest.raises(ValueError, match="negative"):
        validate_joint_schedule(
            _args(smooth_log_depth_residual_bound=-0.1)
        )


def test_global_batch_is_evenly_sharded_across_two_processes():
    assert validate_distributed_batch(2, 1, 2) == 1
    assert shard_batch_indices([7, 11], 0, 2) == [7]
    assert shard_batch_indices([7, 11], 1, 2) == [11]
    with pytest.raises(ValueError, match="divisible"):
        validate_distributed_batch(3, 1, 2)
    with pytest.raises(ValueError, match="per-process"):
        validate_distributed_batch(2, 2, 2)


def test_backbone_freeze_and_linear_warmup():
    assert backbone_learning_rate(
        100, freeze_steps=100, warmup_end=200, peak_lr=5e-7
    ) == 0
    assert backbone_learning_rate(
        150, freeze_steps=100, warmup_end=200, peak_lr=5e-7
    ) == pytest.approx(2.5e-7)
    assert backbone_learning_rate(
        200, freeze_steps=100, warmup_end=200, peak_lr=5e-7
    ) == pytest.approx(5e-7)


def test_cosine_learning_rate_scale_respects_endpoints():
    common = {
        "schedule": "cosine",
        "decay_start_step": 3000,
        "decay_end_step": 5000,
        "final_scale": 0.1,
    }
    assert learning_rate_scale(3000, **common) == pytest.approx(1.0)
    assert learning_rate_scale(4000, **common) == pytest.approx(0.55)
    assert learning_rate_scale(5000, **common) == pytest.approx(0.1)
    assert learning_rate_scale(5500, **common) == pytest.approx(0.1)
    assert learning_rate_scale(
        4000,
        schedule="constant",
        decay_start_step=0,
        decay_end_step=0,
        final_scale=1.0,
    ) == pytest.approx(1.0)


def test_frozen_ddp_backbone_does_not_affect_global_gradient_clipping():
    class Model:
        def __init__(self):
            self.encoder = type("Encoder", (), {})()
            self.encoder.backbone = torch.nn.Linear(2, 2)

    model = Model()
    for parameter in model.encoder.backbone.parameters():
        parameter.grad = torch.ones_like(parameter)
    clear_backbone_gradients(model)
    assert all(
        parameter.grad is None
        for parameter in model.encoder.backbone.parameters()
    )


def test_refiner_stage_switches_after_detach_boundary():
    assert training_stage(500, 500) == "detached_warmup"
    assert training_stage(501, 500) == "joint"
    assert (
        effective_training_stage(
            501,
            500,
            train_refiner_only=True,
        )
        == "refiner_only"
    )


def test_refiner_only_freezes_every_base_parameter():
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(2, 2)
    model.ssr = torch.nn.Linear(2, 2)
    set_refiner_only_trainable(model)
    assert base_parameters_are_frozen(model)
    assert all(parameter.requires_grad for parameter in model.ssr.parameters())
    assert not any(
        parameter.requires_grad for parameter in model.encoder.parameters()
    )


def test_structure_selection_metric_and_flattening():
    assert selection_metric_key("full") == "point_rel"
    assert selection_metric_key("crop") == "crop_point_rel"
    assert selection_metric_key("structure") == "structure_point_rel"
    periodic = {
        "0": {
            "train": {
                "point_rel": 0.2,
                "structure_point_rel": 0.3,
            }
        }
    }
    assert flatten_periodic_evaluation(10, periodic) == {
        "step": 10,
        "train/k0_point_rel": 0.2,
        "train/k0_structure_point_rel": 0.3,
    }
    assert periodic_selection_score(
        periodic,
        refinement_step=0,
        split="train",
        scope="composite",
    ) == pytest.approx(0.5)


def test_composite_selection_requires_full_train_evaluation():
    validate_joint_schedule(
        _args(
            selection_scope="composite",
            selection_split="train",
            fine_structure_rois="locked.json",
            full_eval_every=2500,
        )
    )
    with pytest.raises(ValueError, match="Composite selection"):
        validate_joint_schedule(
            _args(
                selection_scope="composite",
                selection_split="train",
                fine_structure_rois="locked.json",
            )
        )


def test_refinement_collapse_requires_all_enabled_thresholds():
    assert refinement_has_collapsed(
        base_point_rel=0.02,
        refined_point_rel=1.4,
        absolute_threshold=0.1,
        ratio_threshold=3.0,
    )
    assert not refinement_has_collapsed(
        base_point_rel=0.04,
        refined_point_rel=0.11,
        absolute_threshold=0.1,
        ratio_threshold=3.0,
    )
    assert not refinement_has_collapsed(
        base_point_rel=0.02,
        refined_point_rel=1.4,
        absolute_threshold=0.0,
        ratio_threshold=0.0,
    )


def test_preclip_gradient_guard_skips_only_within_both_budgets():
    common = {
        "threshold": 100.0,
        "max_skipped_total": 5,
        "max_skipped_consecutive": 2,
    }
    assert preclip_gradient_action(
        **common,
        grad_norm=449.0,
        skipped_total=0,
        skipped_consecutive=0,
    ) == "skip"
    assert preclip_gradient_action(
        **common,
        grad_norm=449.0,
        skipped_total=5,
        skipped_consecutive=0,
    ) == "abort"
    assert preclip_gradient_action(
        **common,
        grad_norm=449.0,
        skipped_total=1,
        skipped_consecutive=2,
    ) == "abort"
    assert preclip_gradient_action(
        **common,
        grad_norm=99.0,
        skipped_total=5,
        skipped_consecutive=2,
    ) == "apply"


def test_skipped_preclip_state_restores_total_and_consecutive_counts():
    events = [
        {"event": "preclip_gradient_limit_skipped", "step": 4165},
        {"event": "preclip_gradient_limit_skipped", "step": 4199},
        {"event": "preclip_gradient_limit_skipped", "step": 4200},
        {"event": "another_event", "step": 4200},
    ]
    assert skipped_preclip_state(events, start_step=4200) == (3, 2)
    assert skipped_preclip_state(events, start_step=4300) == (3, 0)


def test_aggregate_evaluation_supports_train_and_validation_only():
    records = {
        0: [
            {
                "split": "train",
                "point_rel": 0.2,
                "depth_rel": 0.1,
                "depth_delta_1.01": 0.3,
                "depth_delta_1.25": 0.9,
                "boundary_f1": 0.7,
            },
            {
                "split": "val",
                "point_rel": 0.4,
                "depth_rel": 0.2,
                "depth_delta_1.01": 0.2,
                "depth_delta_1.25": 0.8,
                "boundary_f1": 0.6,
            },
        ]
    }
    aggregated = aggregate_evaluation(records)
    assert set(aggregated["0"]) == {"train", "val"}
    assert aggregated["0"]["val"]["point_rel"] == pytest.approx(0.4)


def test_resume_history_supports_warm_start_origin():
    validate_resume_histories(
        start_step=3500,
        training_history=[{"step": step} for step in range(3001, 3501)],
        evaluation_history=[{"step": 3000}, {"step": 3500}],
    )
    with pytest.raises(ValueError, match="history origin"):
        validate_resume_histories(
            start_step=3500,
            training_history=[{"step": 3500}],
            evaluation_history=[{"step": 3000}, {"step": 3500}],
        )


def test_training_plot_uses_configured_refinement_step(tmp_path):
    training = [
        {"step": 1, "loss": 0.2},
        {"step": 2, "loss": 0.1},
    ]
    evaluation = [
        {
            "step": step,
            **{
                f"{split}/k{k}_{metric}": value
                for split in ("train", "val")
                for k in (0, 1)
                for metric, value in (
                    ("point_rel", 0.2),
                    ("boundary_f1", 0.7),
                )
            },
        }
        for step in (0, 2)
    ]
    save_training_plot(
        tmp_path,
        training,
        evaluation,
        smoothing_window=2,
        detach_step=1,
        refinement_step=1,
    )
    assert (tmp_path / "training_curves.png").is_file()
    assert (tmp_path / "training_curves.pdf").is_file()
