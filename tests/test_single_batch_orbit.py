import numpy as np

from tools.moge3.visualize_single_batch_orbit import (
    orbit_yaw_sequence,
    render_orbit_frame,
)


def test_orbit_yaw_sequence_ping_pongs_without_duplicate_turning_frames():
    values = orbit_yaw_sequence(-35.0, 35.0, 15)
    assert len(values) == 28
    assert values[0] == -35.0
    assert values[14] == 35.0
    assert values[13] == values[15] == 30.0
    assert values.count(35.0) == 1
    assert values[-1] == values[1]


def test_orbit_frame_has_three_labeled_point_maps():
    height, width = 6, 8
    y, x = np.mgrid[:height, :width].astype(np.float32)
    points = np.stack((x / width, y / height, 1.0 + x / width), axis=-1)
    colors = np.full((height, width, 3), 0.5, dtype=np.float32)
    valid = np.ones((height, width), dtype=bool)
    frame = render_orbit_frame(
        {
            "GT": points,
            "K=0 base": points,
            "K=3 refined": points,
        },
        colors,
        valid,
        np.median(points.reshape(-1, 3), axis=0),
        yaw=15.0,
        pitch=-8.0,
    )
    assert frame.mode == "RGB"
    assert frame.width > frame.height
