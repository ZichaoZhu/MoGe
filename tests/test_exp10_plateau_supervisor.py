from __future__ import annotations

from experiment.stage2_training_strategy.runs.exp10_hypersim_48_staged_joint_overfit.supervise_plateau import (
    plateau_state,
)


def _evaluations(*scores: float) -> list[dict[str, float | int]]:
    return [
        {"step": 3500 + index * 100, "score": score}
        for index, score in enumerate(scores)
    ]


def test_plateau_triggers_after_five_non_improving_evaluations() -> None:
    state = plateau_state(
        _evaluations(0.01845, 0.01858, 0.01860, 0.01870, 0.01890, 0.01900),
        start_step=3500,
        patience=5,
        minimum_relative_improvement=0.005,
    )

    assert state["triggered"] is True
    assert state["non_improving_window"] == [3600, 3700, 3800, 3900, 4000]
    assert state["best_step"] == 3500


def test_significant_improvement_resets_plateau_window() -> None:
    state = plateau_state(
        _evaluations(
            0.01845,
            0.01850,
            0.01830,
            0.01831,
            0.01832,
            0.01833,
            0.01834,
        ),
        start_step=3500,
        patience=5,
        minimum_relative_improvement=0.005,
    )

    assert state["triggered"] is False
    assert state["significant_reference_step"] == 3700
    assert state["consecutive_non_improving"] == 4


def test_small_best_improvement_does_not_reset_significance_patience() -> None:
    state = plateau_state(
        _evaluations(0.01845, 0.01844, 0.01843, 0.01842, 0.01841, 0.01840),
        start_step=3500,
        patience=5,
        minimum_relative_improvement=0.005,
    )

    assert state["triggered"] is True
    assert state["best_step"] == 4000
    assert state["best_score"] == 0.01840


def test_waits_for_initial_evaluation() -> None:
    state = plateau_state(
        [],
        start_step=3500,
        patience=5,
        minimum_relative_improvement=0.005,
    )

    assert state["triggered"] is False
    assert state["reason"] == "waiting_for_initial_evaluation"
