import pytest

from tools.moge3.export_exp12_pointclouds import crop_from_selection


def test_crop_from_locked_selection_preserves_xyxy_order():
    selection = {
        "crop_x0": 32,
        "crop_y0": 64,
        "crop_x1": 224,
        "crop_y1": 256,
    }
    assert crop_from_selection(selection) == [32, 64, 224, 256]


def test_crop_from_locked_selection_rejects_missing_coordinate():
    with pytest.raises(KeyError):
        crop_from_selection(
            {
                "crop_x0": 32,
                "crop_y0": 64,
                "crop_x1": 224,
            }
        )
