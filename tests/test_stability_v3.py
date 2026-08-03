import pytest
import torch

from moge.train.stability_v3 import (
    base_geometry_has_collapsed,
    maximum_saturation_fraction,
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
