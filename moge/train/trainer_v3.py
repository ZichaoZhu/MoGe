from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, MutableMapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from .losses import mask_bce_loss, metric_scale_loss
from .losses_v3 import geometric_loss_sequence


@dataclass(frozen=True)
class TrainingScheduleV3:
    """Training constants reported in the MoGe-3 appendix."""

    refiner_detach_steps: int = 5_000
    backbone_freeze_steps: int = 1_000
    backbone_warmup_end: int = 2_000
    ssr_decay_steps: int = 10_000
    base_decay_steps: int = 25_000
    ssr_lr: float = 2e-4
    head_lr: float = 1e-4
    backbone_lr: float = 5e-6
    weight_decay: float = 1e-2
    gradient_clip_norm: float = 1.0


def _parameter_groups(model: nn.Module) -> Tuple[list[nn.Parameter], ...]:
    ssr: list[nn.Parameter] = []
    backbone: list[nn.Parameter] = []
    heads: list[nn.Parameter] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        if name.startswith("ssr."):
            ssr.append(parameter)
        elif name.startswith("encoder.backbone."):
            backbone.append(parameter)
        else:
            heads.append(parameter)
    if not ssr:
        raise ValueError("MoGe-3 optimizer found no SSR parameters")
    if not backbone:
        raise ValueError("MoGe-3 optimizer found no DINO backbone parameters")
    return ssr, heads, backbone


def build_v3_optimizer(
    model: nn.Module,
    schedule: TrainingScheduleV3 = TrainingScheduleV3(),
) -> torch.optim.AdamW:
    ssr, heads, backbone = _parameter_groups(model)
    return torch.optim.AdamW(
        [
            {"name": "ssr", "params": ssr, "lr": schedule.ssr_lr},
            {"name": "heads", "params": heads, "lr": schedule.head_lr},
            {"name": "backbone", "params": backbone, "lr": schedule.backbone_lr},
        ],
        betas=(0.9, 0.999),
        weight_decay=schedule.weight_decay,
    )


def _backbone_multiplier(step: int, schedule: TrainingScheduleV3) -> float:
    if step < schedule.backbone_freeze_steps:
        return 0.0
    if step < schedule.backbone_warmup_end:
        span = schedule.backbone_warmup_end - schedule.backbone_freeze_steps
        return float(step - schedule.backbone_freeze_steps) / max(1, span)
    return 0.5 ** ((step - schedule.backbone_warmup_end) // schedule.base_decay_steps)


def build_v3_scheduler(
    optimizer: torch.optim.Optimizer,
    schedule: TrainingScheduleV3 = TrainingScheduleV3(),
) -> torch.optim.lr_scheduler.LambdaLR:
    functions = []
    for group in optimizer.param_groups:
        name = group.get("name")
        if name == "ssr":
            functions.append(lambda step: 0.5 ** (step // schedule.ssr_decay_steps))
        elif name == "heads":
            functions.append(lambda step: 0.5 ** (step // schedule.base_decay_steps))
        elif name == "backbone":
            functions.append(lambda step: _backbone_multiplier(step, schedule))
        else:
            raise ValueError(f"Unknown MoGe-3 optimizer group: {name!r}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, functions)


def clip_v3_gradients(
    model: nn.Module,
    schedule: TrainingScheduleV3 = TrainingScheduleV3(),
) -> torch.Tensor:
    return torch.nn.utils.clip_grad_norm_(
        (parameter for parameter in model.parameters() if parameter.grad is not None),
        schedule.gradient_clip_norm,
    )


def compute_v3_training_loss(
    model: nn.Module,
    batch: Mapping[str, torch.Tensor],
    step: int,
    num_tokens: int = 2_500,
    num_refinement_steps: int = 3,
    schedule: TrainingScheduleV3 = TrainingScheduleV3(),
    local_scales: Sequence[int] = (4, 16, 64),
    mask_weight: float = 1.0,
    metric_scale_weight: float = 1.0,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
    Apply the paper's component-wise data routing.

    The base model receives every sample. A second forward only selects
    synthetic samples, so real-world samples can never update the SSR module.
    No normal prediction or normal supervision is used.
    """
    required = {"image", "points", "is_synthetic"}
    missing = required.difference(batch)
    if missing:
        raise KeyError(f"Missing MoGe-3 batch fields: {sorted(missing)}")

    image = batch["image"]
    gt_points = batch["points"]
    is_synthetic = batch["is_synthetic"].to(device=image.device, dtype=torch.bool)
    if is_synthetic.shape != (image.shape[0],):
        raise ValueError("is_synthetic must have shape [batch]")

    records: MutableMapping[str, torch.Tensor] = {}
    base_output = model(image, num_tokens=num_tokens, num_refinement_steps=0)
    base_loss, base_records = geometric_loss_sequence(
        [base_output["points"]],
        gt_points,
        local_scales=local_scales,
        generator=generator,
    )
    total = base_loss
    records.update({f"base/{key}": value for key, value in base_records.items()})

    synthetic_indices = torch.where(is_synthetic)[0]
    if synthetic_indices.numel() and num_refinement_steps:
        refined_output = model(
            image[synthetic_indices],
            num_tokens=num_tokens,
            num_refinement_steps=num_refinement_steps,
            return_intermediates=True,
            detach_base_from_refiner=step < schedule.refiner_detach_steps,
        )
        # k=0 is already supervised by base_output; this contributes k=1..K.
        refined_loss, refined_records = geometric_loss_sequence(
            refined_output["points_sequence"][1:],
            gt_points[synthetic_indices],
            local_scales=local_scales,
            generator=generator,
        )
        total = total + refined_loss
        records.update({f"refined/{key}": value for key, value in refined_records.items()})

    if "mask_positive" in batch and "mask_negative" in batch and "mask" in base_output:
        mask_loss, _ = mask_bce_loss(
            base_output["mask"], batch["mask_positive"], batch["mask_negative"]
        )
        mask_mean = mask_loss.mean()
        total = total + mask_weight * mask_mean
        records["aux/mask"] = mask_mean.detach()

    if "metric_scale" in batch and "metric_scale" in base_output:
        scale_loss, _ = metric_scale_loss(
            base_output["metric_scale"], batch["metric_scale"]
        )
        scale_mean = scale_loss.mean()
        total = total + metric_scale_weight * scale_mean
        records["aux/metric_scale"] = scale_mean.detach()

    records["total"] = total.detach()
    records["routing/synthetic_count"] = synthetic_indices.numel() * total.new_ones(())
    records["routing/refiner_detached"] = total.new_tensor(
        float(step < schedule.refiner_detach_steps)
    )
    return total, dict(records)
