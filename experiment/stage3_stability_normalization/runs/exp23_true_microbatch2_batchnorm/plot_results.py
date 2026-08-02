from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


COLORS = {"train": "#2563EB", "val": "#F59E0B", "test": "#DC2626"}


def history_step(path: Path, step: int) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["step"]) == step:
                return {
                    key: float(value)
                    for key, value in row.items()
                    if key != "step"
                }
    raise ValueError(f"Missing step {step}")


def reduction(base: float, refined: float) -> float:
    return 100 * (base - refined) / base


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--baseline-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    evaluation = report["evaluation"]
    baseline = history_step(args.baseline_history, 200)
    scopes = (
        ("train full", "train", "point_rel"),
        ("train structure", "train", "structure_point_rel"),
        ("val full", "val", "point_rel"),
        ("val structure", "val", "structure_point_rel"),
    )
    microbatch1 = []
    microbatch2 = []
    for _, split, metric in scopes:
        microbatch1.append(
            reduction(
                baseline[f"{split}/k0_{metric}"],
                baseline[f"{split}/k3_{metric}"],
            )
        )
        microbatch2.append(
            100
            * evaluation["changes_from_k0"]["3"][split][
                f"{'full' if metric == 'point_rel' else 'structure'}_point_rel_relative_reduction"
            ]
        )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x = np.arange(len(scopes))
    axes[0].bar(x - 0.18, microbatch1, 0.36, label="microbatch 1")
    axes[0].bar(x + 0.18, microbatch2, 0.36, label="microbatch 2")
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[0].set_xticks(x, [scope[0] for scope in scopes], rotation=20)
    axes[0].set_ylabel("K=0 → K=3 Point Rel reduction (%)")
    axes[0].set_title("Step 200 SSR contribution")
    axes[0].legend()

    steps = [0, 1, 3, 5]
    for split in ("train", "val", "test"):
        axes[1].plot(
            steps,
            [
                100
                * evaluation["metrics_by_k"][str(step)][split]["point_rel"]
                for step in steps
            ],
            marker="o",
            color=COLORS[split],
            label=split,
        )
        axes[2].plot(
            steps,
            [
                100
                * evaluation["metrics_by_k"][str(step)][split][
                    "structure_point_rel"
                ]
                for step in steps
            ],
            marker="o",
            color=COLORS[split],
            label=split,
        )
    for axis, title in (
        (axes[1], "Full-image K sweep"),
        (axes[2], "Locked-structure K sweep"),
    ):
        axis.set_xticks(steps)
        axis.set_xlabel("Refinement steps K")
        axis.set_ylabel("Point Rel (%)")
        axis.set_title(title)
        axis.legend()
    figure.suptitle(
        "Exp23 · True microbatch-2 BatchNorm",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=200)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
