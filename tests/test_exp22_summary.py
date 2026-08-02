from experiment.stage3_stability_normalization.runs.exp22_batch_independent_normalization_screen.summarize_results import (
    point_rel_table,
)


def test_point_rel_table_preserves_full_and_structure_scopes() -> None:
    report = {
        "metrics_by_k": {
            "3": {
                "train": {
                    "point_rel": 0.1,
                    "structure_point_rel": 0.2,
                }
            }
        }
    }
    assert point_rel_table(report) == {
        "train": {"full": 0.1, "structure": 0.2}
    }
