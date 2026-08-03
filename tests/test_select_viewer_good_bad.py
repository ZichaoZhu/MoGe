from tools.moge3.select_viewer_good_bad import select_good_bad


def test_selects_three_best_improvements_and_two_worst_degradations():
    rows = []
    for split in ("train", "val", "test"):
        for index, improvement in enumerate(
            (0.5, 0.4, 0.3, 0.2, -0.01, -0.1, -0.5),
            start=1,
        ):
            rows.append(
                {
                    "id": f"{split}-{index}",
                    "split": split,
                    "k0_point_rel": "1.0",
                    "k3_point_rel": str(1.0 - improvement),
                }
            )

    result = select_good_bad(rows, experiment="exp29")

    for split in ("train", "val", "test"):
        chosen = result["splits"][split]
        assert [row["id"] for row in chosen] == [
            f"{split}-1",
            f"{split}-2",
            f"{split}-3",
            f"{split}-7",
            f"{split}-6",
        ]
        assert [row["outcome"] for row in chosen] == [
            "improved",
            "improved",
            "improved",
            "degraded",
            "degraded",
        ]
