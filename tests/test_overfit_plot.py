import csv

import numpy as np

from moge.scripts.overfit_hypersim_v3 import moving_average, save_loss_curve


def test_moving_average_is_trailing_and_aligned():
    values = np.asarray([1.0, 2.0, 3.0, 4.0])
    result = moving_average(values, 3)
    assert np.isnan(result[:2]).all()
    assert np.allclose(result[2:], [2.0, 3.0])


def test_loss_curve_saves_axes_plot_and_numeric_source(tmp_path):
    losses = [0.05, 0.04, 0.035, 0.03, 0.025]
    save_loss_curve(tmp_path, losses, smoothing_window=3)

    for name in ("loss_curve.png", "loss_curve.pdf", "loss_curve.csv"):
        assert (tmp_path / name).stat().st_size > 0

    with (tmp_path / "loss_curve.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert [int(row["step"]) for row in rows] == [1, 2, 3, 4, 5]
    assert rows[0]["moving_mean_3"] == ""
    assert np.isclose(float(rows[-1]["moving_mean_3"]), 0.03)
