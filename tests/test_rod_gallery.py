import pytest

from tools.moge3.render_rod_gallery import (
    SELECTION_INPUTS,
    shard_entries,
    validate_gallery_selection,
)


def make_manifest():
    return {
        "samples": [
            {"id": "train-a", "split": "train"},
            {"id": "train-b", "split": "train"},
            {"id": "val-a", "split": "val"},
        ]
    }


def make_selection():
    return {
        "status": "locked-before-rendering",
        "selection_inputs": SELECTION_INPUTS,
        "shape": [4, 6],
        "expected_count": 2,
        "entries": [
            {
                "order": 1,
                "id": "train-a",
                "crop_xyxy": [0, 0, 3, 4],
                "display_mask": {"type": "near_quantile", "quantile": 0.35},
                "description": "first",
            },
            {
                "order": 2,
                "id": "train-b",
                "crop_xyxy": [3, 0, 6, 4],
                "display_mask": {"type": "near_quantile", "quantile": 0.35},
                "description": "second",
            },
        ],
    }


def test_validate_gallery_selection_accepts_locked_training_entries():
    entries = validate_gallery_selection(
        make_selection(),
        make_manifest(),
        height=4,
        width=6,
    )
    assert [entry["id"] for entry in entries] == ["train-a", "train-b"]


def test_validate_gallery_selection_rejects_validation_sample():
    selection = make_selection()
    selection["entries"][1]["id"] = "val-a"
    with pytest.raises(ValueError, match="not a training sample"):
        validate_gallery_selection(
            selection,
            make_manifest(),
            height=4,
            width=6,
        )


def test_validate_gallery_selection_rejects_prediction_based_provenance():
    selection = make_selection()
    selection["selection_inputs"] = "selected using K=3 error"
    with pytest.raises(ValueError, match="exclude predictions"):
        validate_gallery_selection(
            selection,
            make_manifest(),
            height=4,
            width=6,
        )


def test_shards_are_disjoint_and_cover_all_entries():
    entries = [{"order": index} for index in range(1, 25)]
    shards = [
        shard_entries(entries, shard_index=index, num_shards=4)
        for index in range(4)
    ]
    orders = [[entry["order"] for entry in shard] for shard in shards]
    assert orders[0] == [1, 5, 9, 13, 17, 21]
    assert orders[1] == [2, 6, 10, 14, 18, 22]
    assert orders[2] == [3, 7, 11, 15, 19, 23]
    assert orders[3] == [4, 8, 12, 16, 20, 24]
    assert sorted(sum(orders, [])) == list(range(1, 25))
