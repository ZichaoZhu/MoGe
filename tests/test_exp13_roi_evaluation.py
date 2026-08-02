from __future__ import annotations

from moge.scripts.evaluate_hypersim_checkpoint_rois_v3 import (
    changes_from_k0,
    flatten_per_frame,
    fraction_improved,
    save_csv,
)


def test_roi_evaluation_flattens_steps_and_summarizes_changes():
    records = {
        0: [
            {
                "id": "a",
                "split": "train",
                "scene": "s",
                "frame": 0,
                "point_rel": 0.2,
                "structure_point_rel": 0.3,
            }
        ],
        3: [
            {
                "id": "a",
                "split": "train",
                "scene": "s",
                "frame": 0,
                "point_rel": 0.1,
                "structure_point_rel": 0.2,
            }
        ],
    }
    rows = flatten_per_frame(records)
    assert rows[0]["k3_structure_point_rel"] == 0.2
    metrics = {
        "0": {
            "train": {
                "point_rel": 0.2,
                "structure_point_rel": 0.3,
            }
        },
        "3": {
            "train": {
                "point_rel": 0.1,
                "structure_point_rel": 0.2,
            }
        },
    }
    changes = changes_from_k0(metrics, 3)
    assert changes["train"]["full_point_rel_relative_reduction"] == 0.5
    assert changes["train"]["structure_point_rel_relative_reduction"] > 0.33
    fractions = fraction_improved(rows, step=3)
    assert fractions["train"] == {
        "full_point_rel": 1.0,
        "structure_point_rel": 1.0,
    }


def test_per_frame_csv_accepts_rows_with_and_without_roi_columns(tmp_path):
    output = tmp_path / "metrics.csv"
    save_csv(
        output,
        [
            {"id": "plain", "k0_point_rel": 0.1},
            {
                "id": "roi",
                "k0_point_rel": 0.2,
                "k0_structure_point_rel": 0.3,
            },
        ],
    )
    text = output.read_text(encoding="utf-8")
    assert "k0_structure_point_rel" in text.splitlines()[0]
    assert len(text.splitlines()) == 3
