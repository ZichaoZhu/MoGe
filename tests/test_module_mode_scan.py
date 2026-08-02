from tools.moge3.scan_module_modes import mode_decision


def make_rows(mode: str, values: dict[str, float]) -> list[dict]:
    return [
        {
            "mode": mode,
            "id": sample_id,
            "max_abs_residual": value,
        }
        for sample_id, value in values.items()
    ]


def test_mode_decision_identifies_ssr_eval_outliers() -> None:
    rows = []
    rows += make_rows("base_eval_ssr_eval", {"a": 0.7, "b": 0.2})
    rows += make_rows("base_eval_ssr_train", {"a": 0.3, "b": 0.2})
    rows += make_rows("base_train_ssr_eval", {"a": 0.6, "b": 0.2})
    rows += make_rows("base_train_ssr_train", {"a": 0.4, "b": 0.2})
    decision = mode_decision(rows, threshold=0.5)
    assert decision["outliers_only_when_ssr_eval"]
    assert decision["ssr_eval_always_worse_than_matching_ssr_train"]
    assert not decision["base_mode_changes_outlier_presence_with_fixed_ssr_mode"]


def test_mode_decision_does_not_hide_base_mode_effect() -> None:
    rows = []
    rows += make_rows("base_eval_ssr_eval", {"a": 0.7})
    rows += make_rows("base_eval_ssr_train", {"a": 0.3})
    rows += make_rows("base_train_ssr_eval", {"a": 0.4})
    rows += make_rows("base_train_ssr_train", {"a": 0.3})
    decision = mode_decision(rows, threshold=0.5)
    assert decision["base_mode_changes_outlier_presence_with_fixed_ssr_mode"]
