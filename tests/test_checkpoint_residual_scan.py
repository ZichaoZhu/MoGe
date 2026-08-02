from tools.moge3.scan_checkpoint_residuals import summarize_rows


def test_summarize_rows_tracks_modes_splits_and_highlights() -> None:
    rows = [
        {
            "mode": "eval",
            "repeat": 0,
            "id": "safe",
            "split": "train",
            "iteration": 1,
            "highlighted": False,
            "max_abs_residual": 0.2,
            "max_signed_residual": 0.2,
            "max_row": 1,
            "max_col": 2,
            "depth_span": 10,
        },
        {
            "mode": "train",
            "repeat": 0,
            "id": "outlier",
            "split": "train",
            "iteration": 1,
            "highlighted": True,
            "max_abs_residual": 0.6,
            "max_signed_residual": -0.6,
            "max_row": 3,
            "max_col": 4,
            "depth_span": 20,
        },
        {
            "mode": "train",
            "repeat": 1,
            "id": "outlier",
            "split": "train",
            "iteration": 1,
            "highlighted": True,
            "max_abs_residual": 0.7,
            "max_signed_residual": -0.7,
            "max_row": 5,
            "max_col": 6,
            "depth_span": 30,
        },
    ]
    summary = summarize_rows(rows, threshold=0.5)
    assert summary["row_count"] == 3
    assert summary["unique_sample_count"] == 2
    assert summary["unique_samples_exceeding_threshold"] == ["outlier"]
    assert summary["global_maximum"]["max_abs_residual"] == 0.7
    assert summary["highlighted_maximum"]["repeat"] == 1
    assert len(summary["aggregates"]) == 2


def test_summarize_rows_uses_strict_threshold() -> None:
    row = {
        "mode": "eval",
        "repeat": 0,
        "id": "at_limit",
        "split": "val",
        "iteration": 3,
        "highlighted": False,
        "max_abs_residual": 0.5,
        "max_signed_residual": 0.5,
        "max_row": 0,
        "max_col": 0,
        "depth_span": 1,
    }
    summary = summarize_rows([row], threshold=0.5)
    assert summary["unique_sample_count_exceeding_threshold"] == 0
    assert summary["aggregates"][0]["samples_exceeding_threshold"] == 0
