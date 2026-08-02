from __future__ import annotations

from unittest.mock import patch

from experiment.stage2_training_strategy.runs.exp11_hypersim_100_train_staged_joint_overfit.supervise_recovery import (
    process_alive,
)


def test_process_alive_rejects_zombie_process() -> None:
    zombie_stat = "123 (bash) Z 1 2 3"
    with patch("pathlib.Path.read_text", return_value=zombie_stat):
        assert process_alive(123) is False


def test_process_alive_accepts_sleeping_process() -> None:
    sleeping_stat = "123 (bash) S 1 2 3"
    with patch("pathlib.Path.read_text", return_value=sleeping_stat):
        assert process_alive(123) is True
