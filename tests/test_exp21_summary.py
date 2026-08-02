from experiment.stage3_stability_normalization.runs.exp21_iteration_specific_bn_statistics.summarize_results import (
    load_json,
)


def test_load_json(tmp_path) -> None:
    path = tmp_path / "report.json"
    path.write_text('{"status": "complete"}\n', encoding="utf-8")
    assert load_json(path) == {"status": "complete"}
