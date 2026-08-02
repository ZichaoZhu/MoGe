from experiment.stage3_stability_normalization.runs.exp23_true_microbatch2_batchnorm.summarize_results import (
    csv_step,
)


def test_csv_step_selects_requested_record(tmp_path) -> None:
    path = tmp_path / "history.csv"
    path.write_text(
        "step,train/k3_point_rel\n100,0.2\n200,0.1\n",
        encoding="utf-8",
    )
    assert csv_step(path, 200) == {"train/k3_point_rel": 0.1}
