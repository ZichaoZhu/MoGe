import pytest

from experiment.stage3_stability_normalization.runs.exp17_ssr_residual_outlier_scan.summarize_results import summarize


def row(mode: str, sample_id: str, value: float, iteration: int = 3) -> dict:
    return {
        "mode": mode,
        "repeat": 0,
        "id": sample_id,
        "split": "train",
        "iteration": iteration,
        "max_abs_residual": value,
    }


def test_summary_separates_eval_mismatch_from_joint_amplification() -> None:
    rows = [
        row("eval", "a", 0.7),
        row("eval", "b", 0.2),
        row("train", "a", 0.4),
        row("train", "b", 0.3),
    ]
    summary = summarize(
        rows,
        threshold=0.5,
        failure_batch_ids={"a", "b"},
        exp16_failure_maximum=0.6,
    )
    assert summary["outliers"]["eval"] == ["a"]
    assert summary["outliers"]["train"] == []
    assert summary["decision"]["batchnorm_train_eval_mismatch_is_supported"]
    assert summary["decision"]["joint_update_amplification_is_supported"]
    assert (
        summary["exp16_failure_batch"][
            "relative_increase_over_baseline_failure_batch_train_maximum"
        ]
        == pytest.approx(0.5)
    )
