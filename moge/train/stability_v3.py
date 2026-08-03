from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch


def raw_residual_tail_loss(
    residuals: Sequence[torch.Tensor],
    *,
    threshold: float,
) -> torch.Tensor:
    """Penalize only raw SSR residual magnitudes above ``threshold``."""
    if threshold < 0:
        raise ValueError("Raw residual tail threshold must be non-negative")
    if not residuals:
        raise ValueError("Raw residual tail loss requires at least one tensor")
    penalties = [
        torch.relu(residual.abs() - float(threshold)).square().mean()
        for residual in residuals
    ]
    return torch.stack(penalties).mean()


@torch.no_grad()
def raw_residual_percentiles(
    residuals: Sequence[torch.Tensor],
    *,
    quantiles: Sequence[float] = (0.95, 0.99, 0.999),
) -> dict[str, float]:
    """Return absolute raw-residual percentiles over all refinement cycles."""
    if not residuals:
        raise ValueError("Residual percentiles require at least one tensor")
    if not quantiles or any(not 0.0 <= value <= 1.0 for value in quantiles):
        raise ValueError("Residual quantiles must lie in [0, 1]")
    values = torch.cat(
        [residual.detach().abs().reshape(-1).float() for residual in residuals]
    )
    levels = torch.tensor(quantiles, dtype=values.dtype, device=values.device)
    measured = torch.quantile(values, levels)
    names = {
        0.95: "ssr_raw_p95",
        0.99: "ssr_raw_p99",
        0.999: "ssr_raw_p999",
    }
    return {
        names.get(value, f"ssr_raw_q{str(value).replace('.', '_')}"): float(
            result.item()
        )
        for value, result in zip(quantiles, measured)
    }


@torch.no_grad()
def maximum_saturation_fraction(
    residuals: Sequence[torch.Tensor],
    *,
    bound: float,
    threshold_fraction: float = 0.95,
) -> float:
    if not residuals:
        raise ValueError("Saturation statistics require at least one tensor")
    if bound <= 0:
        return 0.0
    if not 0.0 < threshold_fraction <= 1.0:
        raise ValueError("Saturation threshold fraction must lie in (0, 1]")
    threshold = threshold_fraction * float(bound)
    return max(
        float((residual.detach().abs() >= threshold).float().mean().item())
        for residual in residuals
    )


def updated_threshold_streak(
    previous: int,
    *,
    value: float,
    threshold: float,
) -> int:
    if previous < 0:
        raise ValueError("Threshold streak cannot be negative")
    if threshold <= 0 or value <= threshold:
        return 0
    return previous + 1


def restored_threshold_streak(
    history: Iterable[Mapping[str, object]],
    *,
    value_key: str,
    threshold: float,
) -> int:
    streak = 0
    for record in reversed(list(history)):
        if float(record.get(value_key, 0.0)) <= threshold:
            break
        streak += 1
    return streak


def composite_point_rel_score(
    periodic: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    refinement_step: int,
    split: str,
) -> float:
    metrics = periodic[str(int(refinement_step))][split]
    if "structure_point_rel" not in metrics:
        raise KeyError("Composite selection requires structure_point_rel")
    return float(metrics["point_rel"]) + float(metrics["structure_point_rel"])


def selection_score(
    periodic: Mapping[str, Mapping[str, Mapping[str, float]]],
    *,
    refinement_step: int,
    split: str,
    scope: str,
) -> float:
    if scope == "composite":
        return composite_point_rel_score(
            periodic,
            refinement_step=refinement_step,
            split=split,
        )
    key = "point_rel" if scope == "full" else f"{scope}_point_rel"
    return float(periodic[str(int(refinement_step))][split][key])


def base_geometry_has_collapsed(
    *,
    current_point_rel: float,
    best_point_rel: float,
    ratio_threshold: float,
) -> bool:
    if ratio_threshold <= 0:
        return False
    return current_point_rel > ratio_threshold * max(best_point_rel, 1e-12)


def relative_window_improvement(
    earlier_score: float,
    later_score: float,
) -> float:
    """Positive values mean that the lower-is-better score improved."""
    return (earlier_score - later_score) / max(abs(earlier_score), 1e-12)


@dataclass(frozen=True)
class RecoveryDecision:
    recover: bool
    reason: str | None


def recovery_decision(
    *,
    raw_residual_maximum: float,
    raw_residual_limit: float,
    saturation_streak: int,
    maximum_saturation_streak: int,
    depth_span: float,
    maximum_depth_span: float,
) -> RecoveryDecision:
    if raw_residual_limit > 0 and raw_residual_maximum > raw_residual_limit:
        return RecoveryDecision(True, "raw_log_depth_residual_limit")
    if (
        maximum_saturation_streak > 0
        and saturation_streak >= maximum_saturation_streak
    ):
        return RecoveryDecision(True, "residual_saturation_streak")
    if maximum_depth_span > 0 and depth_span > maximum_depth_span:
        return RecoveryDecision(True, "voxel_depth_span_limit")
    return RecoveryDecision(False, None)
