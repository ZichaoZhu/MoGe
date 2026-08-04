import pytest
import torch

from moge.train.stability_v3 import (
    base_geometry_has_collapsed,
    maximum_saturation_fraction,
    raw_residual_abort_reason,
    raw_residual_peak_loss,
    raw_residual_percentiles,
    raw_residual_tail_loss,
    recovery_decision,
    relative_window_improvement,
    updated_threshold_streak,
)


def test_raw_residual_tail_loss_is_zero_inside_margin():
    raw = torch.tensor([-0.3, -0.1, 0.0, 0.3], requires_grad=True)
    loss = raw_residual_tail_loss([raw], threshold=0.3)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.equal(raw.grad, torch.zeros_like(raw))


def test_raw_residual_tail_loss_pushes_outliers_toward_margin():
    raw = torch.tensor([-0.5, 0.4], requires_grad=True)
    loss = raw_residual_tail_loss([raw], threshold=0.3)
    loss.backward()
    assert raw.grad is not None
    assert raw.grad[0] < 0
    assert raw.grad[1] > 0


def test_raw_residual_peak_loss_is_not_diluted_by_image_size():
    raw = torch.zeros(2, 32, 32, requires_grad=True)
    with torch.no_grad():
        raw[1, 4, 7] = 1.6
    loss = raw_residual_peak_loss([raw], threshold=0.3)
    assert loss.item() == pytest.approx((1.6 - 0.3) ** 2 / 2)
    loss.backward()
    assert raw.grad is not None
    assert raw.grad[1, 4, 7] > 0
    assert torch.count_nonzero(raw.grad) == 1


def test_raw_residual_abort_ignores_isolated_bounded_outlier():
    assert (
        raw_residual_abort_reason(
            maximum=1.634,
            emergency_limit=4.0,
            p999=0.211,
            p999_limit=0.75,
        )
        is None
    )
    assert (
        raw_residual_abort_reason(
            maximum=4.1,
            emergency_limit=4.0,
            p999=0.211,
            p999_limit=0.75,
        )
        == "raw_log_depth_residual_emergency_limit"
    )
    assert (
        raw_residual_abort_reason(
            maximum=1.2,
            emergency_limit=4.0,
            p999=0.8,
            p999_limit=0.75,
        )
        == "raw_log_depth_residual_p999_limit"
    )


def test_residual_percentiles_and_saturation_are_reported():
    raw = torch.arange(1000, dtype=torch.float32).reshape(1, 20, 50) / 1000
    percentiles = raw_residual_percentiles([raw])
    assert set(percentiles) == {"ssr_raw_p95", "ssr_raw_p99", "ssr_raw_p999"}
    assert percentiles["ssr_raw_p95"] == pytest.approx(0.94905, rel=1e-5)
    applied = 0.1 * torch.tanh(raw / 0.1)
    assert maximum_saturation_fraction([applied], bound=0.1) > 0.6


def test_saturation_streak_resets_after_safe_step():
    streak = updated_threshold_streak(0, value=0.06, threshold=0.05)
    assert streak == 1
    assert updated_threshold_streak(streak, value=0.04, threshold=0.05) == 0


def test_recovery_priorities_and_collapse_rules():
    decision = recovery_decision(
        raw_residual_maximum=1.6,
        raw_residual_limit=1.5,
        saturation_streak=0,
        maximum_saturation_streak=3,
        depth_span=200,
        maximum_depth_span=512,
    )
    assert decision.recover
    assert decision.reason == "raw_log_depth_residual_limit"
    assert base_geometry_has_collapsed(
        current_point_rel=0.21,
        best_point_rel=0.1,
        ratio_threshold=2.0,
    )
    assert relative_window_improvement(0.1, 0.09) == pytest.approx(0.1)
