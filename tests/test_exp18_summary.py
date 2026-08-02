from experiment.stage3_stability_normalization.runs.exp18_base_ssr_mode_factorial.summarize_results import summarize


def make_row(mode: str, sample_id: str, value: float, repeat: int = 0) -> dict:
    return {
        "mode": mode,
        "repeat": repeat,
        "id": sample_id,
        "iteration": 3,
        "max_abs_residual": value,
    }


def test_summary_localizes_outliers_to_ssr_eval() -> None:
    rows = [
        make_row("base_eval_ssr_eval", "a", 0.7),
        make_row("base_train_ssr_eval", "a", 0.7, 0),
        make_row("base_train_ssr_eval", "a", 0.7, 1),
        make_row("base_train_ssr_eval", "a", 0.7, 2),
        make_row("base_eval_ssr_train", "a", 0.3, 0),
        make_row("base_eval_ssr_train", "a", 0.3, 1),
        make_row("base_eval_ssr_train", "a", 0.3, 2),
        make_row("base_train_ssr_train", "a", 0.3, 0),
        make_row("base_train_ssr_train", "a", 0.3, 1),
        make_row("base_train_ssr_train", "a", 0.3, 2),
    ]
    summary = summarize(rows, threshold=0.5)
    assert summary["decision"]["outliers_only_when_ssr_eval"]
    assert summary["decision"]["ssr_eval_outlier_ids_are_base_mode_invariant"]
    differences = summary["paired_mode_differences"][
        "base_eval_vs_train_with_ssr_eval"
    ]
    assert all(item["maximum_absolute_difference"] == 0 for item in differences)
