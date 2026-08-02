from __future__ import annotations

from unittest.mock import patch

from experiment.stage2_training_strategy.runs.exp12_hypersim_100_immediate_joint_finetuning.supervise_training import (
    process_alive,
    terminal_state,
)


def test_process_alive_rejects_zombie_process() -> None:
    zombie_stat = "123 (bash) Z 1 2 3"
    with patch("pathlib.Path.read_text", return_value=zombie_stat):
        assert process_alive(123) is False


def test_plateau_stop_is_a_distinct_terminal_state() -> None:
    assert terminal_state(
        report=None,
        result={"status": "failed", "exit_code": 1},
        decision={"action": "sent_sigint_after_evaluation_plateau"},
    ) == "stopped_on_plateau"


def test_report_takes_precedence_over_pipeline_exit_code() -> None:
    assert terminal_state(
        report={"status": "complete"},
        result={"status": "failed", "exit_code": 1},
        decision=None,
    ) == "complete"
