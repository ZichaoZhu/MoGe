from __future__ import annotations

import pytest
import torch
from torch import nn

from moge.model.ssr import sparse_feature_normalization


@pytest.mark.parametrize("name", ["layer_norm", "group_norm"])
def test_batch_independent_normalization_has_no_running_state(name: str) -> None:
    module = sparse_feature_normalization(32, name)
    features = torch.randn(17, 32)
    module.train()
    training = module(features)
    module.eval()
    evaluation = module(features)
    assert torch.equal(training, evaluation)
    assert not any(
        key.endswith(("running_mean", "running_var", "num_batches_tracked"))
        for key in module.state_dict()
    )


def test_legacy_batch_normalization_remains_default_compatible() -> None:
    module = sparse_feature_normalization(32, "batch_norm")
    assert isinstance(module, nn.BatchNorm1d)
    assert "running_mean" in module.state_dict()


def test_unknown_sparse_normalization_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unsupported SSR normalization"):
        sparse_feature_normalization(32, "instance_norm")
