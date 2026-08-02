import pytest

from experiment.stage3_stability_normalization.runs.exp24_microbatch2_long_detached.summarize_results import (
    residual_core,
)
from experiment.stage3_stability_normalization.runs.exp24_microbatch2_long_detached.plot_results import (
    moving_average,
    relative_reduction,
)


def test_residual_core_keeps_safety_evidence() -> None:
    report = {
        "summary": {
            "threshold": 0.5,
            "unique_sample_count_exceeding_threshold": 1,
            "unique_samples_exceeding_threshold": ["sample"],
            "global_maximum": {"max_abs_residual": 0.7},
            "aggregates": [],
        }
    }
    assert residual_core(report)["unique_samples_exceeding_threshold"] == [
        "sample"
    ]


def test_plot_helpers_preserve_metric_direction() -> None:
    assert relative_reduction(0.10, 0.08) == pytest.approx(20.0)
    values = moving_average([1.0, 3.0, 5.0], 2)
    assert values[0] != values[0]
    assert values[1:].tolist() == [2.0, 4.0]
