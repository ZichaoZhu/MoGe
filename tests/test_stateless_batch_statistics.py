from __future__ import annotations

import torch
from torch import nn

from moge.model.ssr import stateless_batch_statistics


def test_stateless_batch_statistics_uses_batch_values_and_restores_buffers() -> None:
    batch_norm = nn.BatchNorm1d(2)
    batch_norm.running_mean.copy_(torch.tensor([10.0, -10.0]))
    batch_norm.running_var.copy_(torch.tensor([4.0, 9.0]))
    batch_norm.num_batches_tracked.fill_(17)
    batch_norm.eval()
    values = torch.tensor([[1.0, 3.0], [3.0, 7.0], [5.0, 11.0]])
    stored_output = batch_norm(values)
    mean_before = batch_norm.running_mean.clone()
    variance_before = batch_norm.running_var.clone()
    tracked_before = batch_norm.num_batches_tracked.clone()

    with stateless_batch_statistics(batch_norm):
        assert batch_norm.training is False
        assert batch_norm.running_mean is None
        assert batch_norm.running_var is None
        assert batch_norm.num_batches_tracked is None
        batch_output = batch_norm(values)
        assert torch.allclose(batch_output.mean(dim=0), torch.zeros(2), atol=1e-6)

    assert not torch.allclose(stored_output, batch_output)
    assert torch.equal(batch_norm.running_mean, mean_before)
    assert torch.equal(batch_norm.running_var, variance_before)
    assert torch.equal(batch_norm.num_batches_tracked, tracked_before)
    assert batch_norm.track_running_stats is True
    assert batch_norm.training is False
