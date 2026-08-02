from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


SPLITS = ("train", "val", "test")
SPLIT_LABELS = ("Train", "Validation", "Test")
OBJECTIVES = ("global", "local", "edge_paper")
OBJECTIVE_LABELS = ("Global", "Local", "Edge")
COLORS = ("#3274A1", "#E1812C", "#3A923A")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def values(
    rows: Sequence[Mapping[str, str]],
    *,
    split: str,
    key: str,
    value: str,
    metric: str,
) -> list[float]:
    return [
        float(row[metric])
        for row in rows
        if row["split"] == split and row[key] == value
    ]


def mean(values_: Sequence[float]) -> float:
    return float(np.mean(values_))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    objectives = read_csv(args.formal / "effective_batch_objectives.csv")
    scope_cosines = read_csv(args.formal / "effective_scope_cosines.csv")
    groups = read_csv(args.formal / "effective_group_gradients.csv")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(12.8, 8.7))
    x = np.arange(len(SPLITS))
    width = 0.23

    axis = axes[0, 0]
    for objective_index, (objective, label, color) in enumerate(
        zip(OBJECTIVES, OBJECTIVE_LABELS, COLORS, strict=True)
    ):
        means = [
            mean(
                values(
                    objectives,
                    split=split,
                    key="objective",
                    value=objective,
                    metric="gradient_l2",
                )
            )
            for split in SPLITS
        ]
        axis.bar(
            x + (objective_index - 1) * width,
            means,
            width,
            label=label,
            color=color,
        )
    axis.set_xticks(x, SPLIT_LABELS)
    axis.set_ylabel("SSR parameter gradient L2")
    axis.set_title("Gradient strength after SSR Jacobian")
    axis.set_yscale("log")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)

    axis = axes[0, 1]
    pairs = ("global_local", "global_edge", "local_edge")
    pair_labels = ("Global / Local", "Global / Edge", "Local / Edge")
    for pair_index, (pair, label, color) in enumerate(
        zip(pairs, pair_labels, COLORS, strict=True)
    ):
        pair_values = [
            mean([
                float(row["cosine"])
                for row in scope_cosines
                if row["split"] == split
                and row["gradient_pair"] == pair
                and row["parameter_scope"] == "non_output"
            ])
            for split in SPLITS
        ]
        axis.bar(
            x + (pair_index - 1) * width,
            pair_values,
            width,
            label=label,
            color=color,
        )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_xticks(x, SPLIT_LABELS)
    axis.set_ylabel("Cosine similarity")
    axis.set_title("Alignment inside 3D U-Net (output layer excluded)")
    axis.set_ylim(-1, 1)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)

    axis = axes[1, 0]
    group_names = (
        "input_fusion",
        "encoder",
        "bottleneck",
        "decoder",
        "output",
    )
    group_labels = ("Input", "Encoder", "Bottleneck", "Decoder", "Output")
    group_colors = ("#4C78A8", "#72B7B2", "#F2CF5B", "#F58518", "#E45756")
    train_totals = {
        (row["batch_id"], row["objective"]): float(row["gradient_l2"])
        for row in objectives
        if row["split"] == "train" and row["objective"] in OBJECTIVES
    }
    bottoms = np.zeros(len(OBJECTIVES))
    for group, label, color in zip(
        group_names,
        group_labels,
        group_colors,
        strict=True,
    ):
        energy = []
        for objective in OBJECTIVES:
            shares = [
                float(row["gradient_l2"]) ** 2
                / train_totals[(row["batch_id"], objective)] ** 2
                for row in groups
                if row["split"] == "train"
                and row["objective"] == objective
                and row["parameter_group"] == group
            ]
            energy.append(mean(shares))
        axis.bar(
            OBJECTIVE_LABELS,
            energy,
            bottom=bottoms,
            label=label,
            color=color,
        )
        bottoms += np.asarray(energy)
    axis.set_ylabel("Mean squared-gradient energy share")
    axis.set_title("Where train gradients enter SSR")
    axis.set_ylim(0, 1.02)
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False, ncol=3, fontsize=8)

    axis = axes[1, 1]
    for objective_index, (objective, label, color) in enumerate(
        zip(OBJECTIVES, OBJECTIVE_LABELS, COLORS, strict=True)
    ):
        projections = [
            mean(
                values(
                    objectives,
                    split=split,
                    key="objective",
                    value=objective,
                    metric="projection_fraction_to_paper_combined",
                )
            )
            for split in SPLITS
        ]
        axis.bar(
            x + (objective_index - 1) * width,
            projections,
            width,
            label=label,
            color=color,
        )
    axis.axhline(0, color="#222222", linewidth=0.8)
    axis.set_xticks(x, SPLIT_LABELS)
    axis.set_ylabel("Projection fraction")
    axis.set_title("Contribution along the paper-combined update")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)

    figure.suptitle(
        "Exp26 — SSR parameter-gradient conflict audit (K=1…3, microbatch=2)",
        fontsize=14,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(args.output, dpi=180)
    figure.savefig(args.output.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
