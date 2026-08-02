from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


ARMS = {
    "lr_2e6": ("SSR LR 2e-6", "#2563EB"),
    "lr_1e5": ("SSR LR 1e-5", "#F59E0B"),
    "lr_2e5": ("SSR LR 2e-5", "#DC2626"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the archived Exp13 metrics.")
    parser.add_argument("--experiment", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()
    experiment = args.experiment.resolve()
    source = experiment / "results" / "remote"
    output = experiment / "results"

    figure, axes = plt.subplots(2, 2, figsize=(13.2, 8.4), constrained_layout=True)
    panels = (
        ("train/k3_point_rel", "Train full Point Rel", axes[0, 0]),
        (
            "train/k3_structure_point_rel",
            "Train locked-structure Point Rel",
            axes[0, 1],
        ),
        ("val/k3_point_rel", "Validation full Point Rel", axes[1, 0]),
        (
            "val/k3_structure_point_rel",
            "Validation locked-structure Point Rel",
            axes[1, 1],
        ),
    )
    for arm, (label, color) in ARMS.items():
        rows = read_csv(source / arm / "evaluation_history.csv")
        steps = np.asarray([int(row["step"]) for row in rows])
        for key, _, axis in panels:
            values = 100 * np.asarray([float(row[key]) for row in rows])
            axis.plot(
                steps,
                values,
                color=color,
                marker="o",
                linewidth=2,
                markersize=4,
                label=label,
            )
    for _, title, axis in panels:
        axis.set_title(title)
        axis.set_xlabel("Training step")
        axis.set_ylabel("Point Rel (%) ↓")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Exp13 fixed-Base SSR learning-rate sweep", fontsize=15)
    figure.savefig(output / "metric_curves.png", dpi=180)
    figure.savefig(output / "metric_curves.pdf")
    plt.close(figure)

    splits = ("train", "val", "test")
    figure, axes = plt.subplots(1, 3, figsize=(14.4, 4.5), constrained_layout=True)
    x = np.arange(len(ARMS))
    width = 0.34
    for axis, split in zip(axes, splits):
        k3 = []
        k5 = []
        for arm in ARMS:
            report = json.loads(
                (source / f"{arm}_posthoc" / "report.json").read_text(
                    encoding="utf-8"
                )
            )
            k3.append(
                100
                * report["changes_from_k0"]["3"][split][
                    "structure_point_rel_relative_reduction"
                ]
            )
            k5.append(
                100
                * report["changes_from_k0"]["5"][split][
                    "structure_point_rel_relative_reduction"
                ]
            )
        axis.bar(x - width / 2, k3, width, label="K=3", color="#2563EB")
        axis.bar(x + width / 2, k5, width, label="K=5", color="#7C3AED")
        axis.axhline(0, color="#111827", linewidth=1)
        axis.set_xticks(x, [label for label, _ in ARMS.values()], rotation=18)
        axis.set_title(f"{split.title()} locked structure")
        axis.set_ylabel("Point Rel reduction from K=0 (%) ↑")
        axis.grid(True, axis="y", alpha=0.25)
        axis.legend(frameon=False)
    figure.suptitle("Exp13 iterative SSR gains and held-out stability", fontsize=15)
    figure.savefig(output / "k_sweep_structure.png", dpi=180)
    figure.savefig(output / "k_sweep_structure.pdf")
    plt.close(figure)


if __name__ == "__main__":
    main()
