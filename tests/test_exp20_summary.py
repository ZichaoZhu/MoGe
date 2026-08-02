from experiment.stage3_stability_normalization.runs.exp20_stateless_ssr_batch_statistics.summarize_results import (
    summarize,
)


def _residual(maximum: float, outliers: int) -> dict:
    return {
        "checkpoint_sha256": "same",
        "summary": {
            "threshold": 0.5,
            "global_maximum": {
                "max_abs_residual": maximum,
                "id": "sample",
                "iteration": 3,
            },
            "unique_sample_count_exceeding_threshold": outliers,
            "unique_samples_exceeding_threshold": (
                ["sample"] if outliers else []
            ),
        },
    }


def _metrics(value: float) -> dict:
    return {
        "checkpoint_sha256": "same",
        "metrics_by_k": {
            str(step): {
                split: {
                    "point_rel": value if step == 3 else 1.0,
                    "depth_rel": 1.0,
                    "boundary_f1": 0.0,
                    "structure_point_rel": value if step == 3 else 1.0,
                    "structure_depth_rel": 1.0,
                    "structure_boundary_f1": 0.0,
                }
                for split in ("train", "val", "test")
            }
            for step in (0, 1, 3, 5)
        },
    }


def test_summary_accepts_safe_universal_improvement() -> None:
    report = summarize(
        _residual(1.0, 1),
        _metrics(0.2),
        _residual(0.4, 0),
        _metrics(0.1),
    )
    assert report["checkpoint_unchanged"]
    assert report["decision"]["all_residual_outliers_removed"]
    assert report["decision"]["maximum_residual_below_threshold"]
    assert report["decision"]["k3_point_rel_improves_all_splits_and_scopes"]
