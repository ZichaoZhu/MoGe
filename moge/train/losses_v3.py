from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import torch
import utils3d

from ..utils.alignment import align_points_scale_z_shift
from ..utils.geometry_torch import weighted_mean
from .losses import edge_loss


@dataclass
class GlobalAlignment:
    scale: torch.Tensor
    shift: torch.Tensor
    valid: torch.Tensor

    def apply(self, points: torch.Tensor) -> torch.Tensor:
        return self.scale[..., None, None, None] * points + self.shift[..., None, None, :]

    def detached(self) -> "GlobalAlignment":
        return GlobalAlignment(self.scale.detach(), self.shift.detach(), self.valid.detach())


def edge_angle_loss_v3(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Paper Eq. (13), normalized by the shorter image dimension."""
    return edge_loss(
        pred_points,
        gt_points,
        normalization_dimension="min",
    )


def solve_global_affine_alignment(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    align_resolution: int = 64,
    trunc: float = 1.0,
) -> GlobalAlignment:
    if pred_points.ndim == 3:
        pred_points = pred_points.unsqueeze(0)
        gt_points = gt_points.unsqueeze(0)
    mask = torch.isfinite(gt_points).all(dim=-1)
    safe_gt = torch.where(mask[..., None], gt_points, torch.ones_like(gt_points))
    pred_lr, gt_lr, mask_lr = utils3d.pt.masked_nearest_resize(
        pred_points, safe_gt, mask=mask, size=(align_resolution, align_resolution)
    )
    batch_has_valid = mask_lr.any(dim=(-2, -1))
    scale = pred_points.new_zeros(pred_points.shape[0])
    shift = pred_points.new_zeros((pred_points.shape[0], 3))
    if batch_has_valid.any():
        weight_lr = (
            mask_lr[batch_has_valid].flatten(-2, -1)
            / gt_lr[batch_has_valid][..., 2].flatten(-2, -1).clamp_min(1e-2)
        )
        solved_scale, solved_shift = align_points_scale_z_shift(
            pred_lr[batch_has_valid].flatten(-3, -2),
            gt_lr[batch_has_valid].flatten(-3, -2),
            weight_lr,
            trunc=trunc,
        )
        scale[batch_has_valid] = solved_scale
        shift[batch_has_valid] = solved_shift
    valid = (
        batch_has_valid
        & torch.isfinite(scale)
        & (scale > 0)
        & torch.isfinite(shift).all(dim=-1)
    )
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    shift = torch.where(valid[..., None], shift, torch.zeros_like(shift))
    return GlobalAlignment(scale=scale, shift=shift, valid=valid)


def affine_invariant_global_loss_v3(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    align_resolution: int = 64,
    trunc: float = 1.0,
) -> Tuple[torch.Tensor, GlobalAlignment]:
    squeeze = pred_points.ndim == 3
    if squeeze:
        pred_points = pred_points.unsqueeze(0)
        gt_points = gt_points.unsqueeze(0)
    mask = torch.isfinite(gt_points).all(dim=-1)
    safe_gt = torch.where(mask[..., None], gt_points, torch.ones_like(gt_points))
    alignment = solve_global_affine_alignment(
        pred_points, gt_points, align_resolution=align_resolution, trunc=trunc
    )
    # Alignment is the solved nuisance optimum for this prediction. Its
    # derivative is not needed for the minimized objective (envelope theorem),
    # and differentiating through the selected anchor ratio can explode when
    # two predicted anchor coordinates are nearly equal. Treat the solved
    # scale/shift as constants, matching the local-loss alignment route.
    aligned = alignment.detached().apply(pred_points)

    weight = (mask & alignment.valid[..., None, None]).float()
    weight = weight / safe_gt[..., 2].clamp_min(1e-5)
    mean_weight = weighted_mean(weight, mask, dim=(-2, -1), keepdim=True)
    weight = weight.clamp_max(10.0 * mean_weight)
    loss = ((aligned - safe_gt).abs() * weight[..., None]).mean(dim=(-3, -2, -1))
    if squeeze:
        loss = loss.squeeze(0)
    return loss, alignment


def grouped_weighted_median(
    values: torch.Tensor,
    weights: torch.Tensor,
    group_ids: torch.Tensor,
    num_groups: Optional[int] = None,
) -> torch.Tensor:
    """Vectorized component-wise weighted medians for integer groups."""
    if values.ndim == 1:
        values = values[:, None]
    if values.shape[0] == 0:
        groups = int(num_groups or 0)
        return values.new_zeros((groups, values.shape[1]))
    if (weights < 0).any():
        raise ValueError("Weighted median requires non-negative weights")
    if not torch.isfinite(weights).all():
        raise ValueError("Weighted median requires finite weights")
    if num_groups is None:
        num_groups = int(group_ids.max().item()) + 1

    medians = []
    for component in range(values.shape[1]):
        by_value = torch.argsort(values[:, component], stable=True)
        groups_after_value = group_ids[by_value]
        by_group = torch.argsort(groups_after_value, stable=True)
        order = by_value[by_group]
        sorted_groups = group_ids[order]
        sorted_weights = weights[order]
        accumulator_dtype = (
            torch.float64
            if sorted_weights.dtype
            in (torch.float16, torch.bfloat16, torch.float32)
            else sorted_weights.dtype
        )
        cumulative = sorted_weights.to(accumulator_dtype).cumsum(dim=0)

        counts = torch.bincount(sorted_groups, minlength=num_groups)
        if (counts == 0).any():
            raise ValueError("group_ids must densely cover [0, num_groups)")
        ends = counts.cumsum(dim=0) - 1
        starts = ends - counts + 1
        prefix = torch.where(
            starts > 0,
            cumulative[(starts - 1).clamp_min(0)],
            torch.zeros_like(cumulative[starts]),
        )
        totals = cumulative[ends] - prefix
        thresholds = prefix + 0.5 * totals
        median_positions = torch.searchsorted(cumulative, thresholds, right=False)
        # A global float32 prefix can dwarf the weight of later groups, making
        # prefix + group_total numerically equal to prefix. Accumulating in
        # float64 avoids most such cases; clamping also guarantees that a
        # round-off at a group boundary cannot select another group or N.
        median_positions = torch.maximum(
            starts,
            torch.minimum(median_positions.clamp_max(values.shape[0] - 1), ends),
        )
        medians.append(values[order[median_positions], component])
    return torch.stack(medians, dim=-1)


def _random_rotations(
    batch_size: int,
    device: torch.device,
    dtype: torch.dtype,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    matrices = torch.randn(
        batch_size, 3, 3, device=device, dtype=dtype, generator=generator
    )
    q, r = torch.linalg.qr(matrices)
    diagonal_sign = torch.sign(torch.diagonal(r, dim1=-2, dim2=-1))
    diagonal_sign = torch.where(diagonal_sign == 0, torch.ones_like(diagonal_sign), diagonal_sign)
    q = q * diagonal_sign[:, None, :]
    determinant = torch.linalg.det(q)
    handedness = torch.where(determinant < 0, -1.0, 1.0)
    return q * torch.stack(
        (torch.ones_like(handedness), torch.ones_like(handedness), handedness),
        dim=-1,
    )[:, None, :]


def radial_partition_local_loss(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
    alignment: GlobalAlignment,
    scales: Sequence[int] = (4, 16, 64),
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Paper Appendix A radial-partition loss with a shared global scale.

    Partition source is sampled per image from ground truth or the globally
    aligned prediction. Region translations are solved as weighted L1 medians.
    """
    squeeze = pred_points.ndim == 3
    if squeeze:
        pred_points = pred_points.unsqueeze(0)
        gt_points = gt_points.unsqueeze(0)
    batch_size, height, width, _ = pred_points.shape
    device, dtype = pred_points.device, pred_points.dtype
    mask = torch.isfinite(gt_points).all(dim=-1)
    safe_gt = torch.where(mask[..., None], gt_points, torch.ones_like(gt_points))
    detached_alignment = alignment.detached()
    globally_aligned = detached_alignment.apply(pred_points)

    choose_gt = torch.rand(
        batch_size, device=device, generator=generator
    ) < 0.5
    reference = torch.where(
        choose_gt[:, None, None, None], safe_gt, globally_aligned.detach()
    )
    rotations = _random_rotations(batch_size, device, dtype, generator)
    reference = torch.einsum("bij,bhwj->bhwi", rotations, reference)

    pixel_batch = torch.arange(batch_size, device=device)[:, None, None]
    pixel_batch = pixel_batch.expand(-1, height, width)
    residual_for_shift = (
        safe_gt - detached_alignment.scale[:, None, None, None] * pred_points
    )
    base_weight = mask.float() / safe_gt[..., 2].clamp_min(1e-5)
    total = pred_points.new_zeros(batch_size)

    for alpha in scales:
        log_scale_jitter = torch.rand(
            batch_size, device=device, dtype=dtype, generator=generator
        ) / float(alpha)
        perturbed = reference * log_scale_jitter.exp()[:, None, None, None]
        norm_inf = perturbed.abs().amax(dim=-1, keepdim=True).clamp_min(1e-6)
        norm_l2 = perturbed.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        partition_values = torch.cat((perturbed / norm_inf, norm_l2.log()), dim=-1)
        partition_keys = torch.floor(float(alpha) * partition_values).to(torch.long)

        valid_keys = torch.cat(
            (pixel_batch[mask][:, None], partition_keys[mask]), dim=-1
        )
        _, inverse = torch.unique(valid_keys, dim=0, sorted=True, return_inverse=True)
        valid_residual = residual_for_shift[mask]
        valid_weight = base_weight[mask]
        shifts = grouped_weighted_median(valid_residual, valid_weight, inverse)

        aligned_valid = (
            detached_alignment.scale[pixel_batch[mask], None] * pred_points[mask]
            + shifts[inverse]
        )
        point_loss = (aligned_valid - safe_gt[mask]).abs().mean(dim=-1) * valid_weight
        per_image = torch.zeros(batch_size, device=device, dtype=dtype)
        per_image.scatter_add_(0, pixel_batch[mask], point_loss)
        denominator = mask.sum(dim=(-2, -1)).clamp_min(1).to(dtype)
        total = total + per_image / denominator

    if squeeze:
        return total.squeeze(0)
    return total


def geometric_loss_sequence(
    points_sequence: Iterable[torch.Tensor],
    gt_points: torch.Tensor,
    global_weight: float = 1.0,
    local_weight: float = 1.0,
    edge_weight: float = 1.0,
    local_scales: Sequence[int] = (4, 16, 64),
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute L_global + L_local + L_edge for every k=0..K prediction."""
    total = gt_points.new_zeros(())
    records: Dict[str, torch.Tensor] = {}
    for index, points in enumerate(points_sequence):
        global_loss, alignment = affine_invariant_global_loss_v3(points, gt_points)
        local_loss = radial_partition_local_loss(
            points,
            gt_points,
            alignment,
            scales=local_scales,
            generator=generator,
        )
        edge, _ = edge_angle_loss_v3(points, gt_points)
        global_mean = global_loss.mean()
        local_mean = local_loss.mean()
        edge_mean = edge.mean()
        records[f"geometry/{index}/global"] = global_mean.detach()
        records[f"geometry/{index}/local"] = local_mean.detach()
        records[f"geometry/{index}/edge"] = edge_mean.detach()
        total = total + (
            global_weight * global_mean
            + local_weight * local_mean
            + edge_weight * edge_mean
        )
    records["geometry/total"] = total.detach()
    return total, records
