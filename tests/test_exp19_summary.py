from experiment.stage3_stability_normalization.runs.exp19_ssr_batchnorm_recalibration.summarize_results import (
    relative_change,
    residual_summary,
)


def test_relative_change_uses_lower_is_better() -> None:
    assert relative_change(2.0, 1.0) == 0.5
    assert relative_change(1.0, 2.0) == -1.0


def test_residual_summary_preserves_outlier_identity() -> None:
    report = {
        "summary": {
            "global_maximum": {
                "max_abs_residual": 0.7,
                "id": "sample",
                "iteration": 3,
            },
            "unique_sample_count_exceeding_threshold": 1,
            "unique_samples_exceeding_threshold": ["sample"],
        }
    }
    assert residual_summary(report) == {
        "maximum": 0.7,
        "maximum_id": "sample",
        "maximum_iteration": 3,
        "outlier_count": 1,
        "outlier_ids": ["sample"],
    }
