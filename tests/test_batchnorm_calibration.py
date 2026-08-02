from __future__ import annotations

import torch
from torch import nn

from tools.moge3.calibrate_ssr_batchnorm import (
    batch_slices,
    reset_for_cumulative_calibration,
    restore_momenta,
    verify_only_allowed_state_changed,
)


def test_batch_slices_include_short_final_batch() -> None:
    assert batch_slices(10, 4) == [(0, 4), (4, 8), (8, 10)]


def test_cumulative_calibration_reset_and_restore() -> None:
    module = nn.BatchNorm1d(3, momentum=0.1)
    module.running_mean.fill_(5)
    module.running_var.fill_(7)
    module.num_batches_tracked.fill_(9)
    momenta = reset_for_cumulative_calibration([module])
    assert module.momentum is None
    assert torch.equal(module.running_mean, torch.zeros(3))
    assert torch.equal(module.running_var, torch.ones(3))
    assert module.num_batches_tracked.item() == 0
    restore_momenta([module], momenta)
    assert module.momentum == 0.1


def test_state_audit_only_permits_named_buffers() -> None:
    reference = {
        "weight": torch.tensor([1.0]),
        "running_mean": torch.tensor([0.0]),
    }
    candidate = {
        "weight": torch.tensor([1.0]),
        "running_mean": torch.tensor([2.0]),
    }
    report = verify_only_allowed_state_changed(
        reference,
        candidate,
        {"running_mean"},
    )
    assert report["changed_allowed_buffer_count"] == 1
    candidate["weight"] = torch.tensor([3.0])
    try:
        verify_only_allowed_state_changed(
            reference,
            candidate,
            {"running_mean"},
        )
    except RuntimeError as error:
        assert "forbidden state" in str(error)
    else:
        raise AssertionError("Forbidden parameter modification was not detected")
