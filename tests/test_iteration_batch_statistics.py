from __future__ import annotations

import torch
from torch import nn

from moge.model.ssr import (
    capture_batch_norm_running_state,
    load_batch_norm_running_state,
)
from tools.moge3.calibrate_ssr_iteration_batchnorm import update_factorized


def test_running_state_round_trip() -> None:
    module = nn.Sequential(nn.BatchNorm1d(2), nn.Linear(2, 2))
    module[0].running_mean.copy_(torch.tensor([1.0, 2.0]))
    module[0].running_var.copy_(torch.tensor([3.0, 4.0]))
    module[0].num_batches_tracked.fill_(5)
    state = capture_batch_norm_running_state(module)
    module[0].running_mean.zero_()
    module[0].running_var.fill_(1)
    module[0].num_batches_tracked.zero_()
    load_batch_norm_running_state(module, state)
    assert torch.equal(module[0].running_mean, torch.tensor([1.0, 2.0]))
    assert torch.equal(module[0].running_var, torch.tensor([3.0, 4.0]))
    assert module[0].num_batches_tracked.item() == 5


def test_log_depth_update_preserves_projected_coordinates() -> None:
    factorized = torch.tensor([[[[0.2, -0.4, 1.0]]]])
    updated = update_factorized(factorized, torch.tensor([[[0.3]]]))
    assert torch.equal(updated[..., :2], factorized[..., :2])
    assert torch.allclose(updated[..., 2], torch.tensor([[[1.3]]]))
