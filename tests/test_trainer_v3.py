import torch
import torch.nn as nn

from moge.train.trainer_v3 import (
    TrainingScheduleV3,
    build_v3_optimizer,
    build_v3_scheduler,
    compute_v3_training_loss,
)


class _TinyRoutedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssr = nn.Linear(2, 2)
        self.encoder = nn.Module()
        self.encoder.backbone = nn.Linear(2, 2)
        self.points_head = nn.Linear(2, 2)


class _GradientRoutingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Parameter(torch.tensor(0.5))
        self.ssr = nn.Parameter(torch.tensor(0.25))

    def forward(
        self,
        image,
        num_tokens,
        num_refinement_steps,
        return_intermediates=False,
        detach_base_from_refiner=False,
    ):
        base_points = self.base * torch.ones(
            image.shape[0], image.shape[2], image.shape[3], 3, device=image.device
        )
        output = {"points": base_points}
        if return_intermediates:
            refined_base = base_points.detach() if detach_base_from_refiner else base_points
            sequence = [base_points]
            current = refined_base
            for _ in range(num_refinement_steps):
                current = current + self.ssr
                sequence.append(current)
            output["points"] = sequence[-1]
            output["points_sequence"] = sequence
        return output


def _simple_geometry_loss(points_sequence, gt_points, **kwargs):
    loss = sum((points - gt_points).square().mean() for points in points_sequence)
    return loss, {"geometry/total": loss.detach()}


def _routing_batch(is_synthetic):
    batch_size = len(is_synthetic)
    return {
        "image": torch.zeros(batch_size, 3, 2, 2),
        "points": torch.zeros(batch_size, 2, 2, 3),
        "is_synthetic": torch.tensor(is_synthetic),
    }


def test_optimizer_groups_and_learning_rates():
    model = _TinyRoutedModel()
    schedule = TrainingScheduleV3()
    optimizer = build_v3_optimizer(model, schedule)
    assert [group["name"] for group in optimizer.param_groups] == [
        "ssr",
        "heads",
        "backbone",
    ]
    assert [group["lr"] for group in optimizer.param_groups] == [
        schedule.ssr_lr,
        schedule.head_lr,
        schedule.backbone_lr,
    ]
    build_v3_scheduler(optimizer, schedule)
    assert optimizer.param_groups[2]["lr"] == 0.0


def test_backbone_freeze_warmup_and_decay():
    model = _TinyRoutedModel()
    schedule = TrainingScheduleV3()
    optimizer = build_v3_optimizer(model, schedule)
    scheduler = build_v3_scheduler(optimizer, schedule)
    backbone_base_lr = schedule.backbone_lr

    scheduler.step(999)
    assert optimizer.param_groups[2]["lr"] == 0.0
    scheduler.step(1_500)
    assert abs(optimizer.param_groups[2]["lr"] - 0.5 * backbone_base_lr) < 1e-12
    scheduler.step(2_000)
    assert abs(optimizer.param_groups[2]["lr"] - backbone_base_lr) < 1e-12
    scheduler.step(27_000)
    assert abs(optimizer.param_groups[2]["lr"] - 0.5 * backbone_base_lr) < 1e-12


def test_real_only_batch_never_updates_ssr(monkeypatch):
    monkeypatch.setattr(
        "moge.train.trainer_v3.geometric_loss_sequence", _simple_geometry_loss
    )
    model = _GradientRoutingModel()
    loss, _ = compute_v3_training_loss(model, _routing_batch([False, False]), step=6_000)
    loss.backward()
    assert model.base.grad is not None
    assert model.ssr.grad is None


def test_synthetic_batch_updates_ssr_but_detaches_base_during_warmup(monkeypatch):
    monkeypatch.setattr(
        "moge.train.trainer_v3.geometric_loss_sequence", _simple_geometry_loss
    )
    early_model = _GradientRoutingModel()
    early_loss, early_records = compute_v3_training_loss(
        early_model, _routing_batch([True, False]), step=4_999, num_refinement_steps=1
    )
    early_loss.backward()
    early_base_grad = early_model.base.grad.detach().clone()
    assert early_model.ssr.grad is not None
    assert early_records["routing/refiner_detached"].item() == 1.0

    joint_model = _GradientRoutingModel()
    joint_loss, joint_records = compute_v3_training_loss(
        joint_model, _routing_batch([True, False]), step=5_000, num_refinement_steps=1
    )
    joint_loss.backward()
    assert joint_model.ssr.grad is not None
    assert joint_records["routing/refiner_detached"].item() == 0.0
    assert joint_model.base.grad.abs() > early_base_grad.abs()
