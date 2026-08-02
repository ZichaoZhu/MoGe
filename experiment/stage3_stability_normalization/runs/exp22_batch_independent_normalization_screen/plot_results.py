from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "layer_norm": "LayerNorm",
    "group_norm": "GroupNorm",
}
COLORS = {
    "train": "#2563EB",
    "val": "#F59E0B",
    "test": "#DC2626",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text(encoding="utf-8"))
    variants = list(LABELS)
    splits = ("train", "val", "test")

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    x = np.arange(len(variants))
    width = 0.24
    for split_index, split in enumerate(splits):
        offset = (split_index - 1) * width
        full = [
            100
            * report["variants"][name]["evaluation"][
                "k3_changes_from_k0"
            ][split]["full_point_rel_relative_reduction"]
            for name in variants
        ]
        structure = [
            100
            * report["variants"][name]["evaluation"][
                "k3_changes_from_k0"
            ][split]["structure_point_rel_relative_reduction"]
            for name in variants
        ]
        axes[0].bar(
            x + offset,
            full,
            width,
            color=COLORS[split],
            label=split,
        )
        axes[1].bar(
            x + offset,
            structure,
            width,
            color=COLORS[split],
            label=split,
        )

    maxima = [
        report["variants"][name]["residual"]["global_maximum"][
            "max_abs_residual"
        ]
        for name in variants
    ]
    outliers = [
        report["variants"][name]["residual"][
            "unique_sample_count_exceeding_threshold"
        ]
        for name in variants
    ]
    axes[2].bar(x, maxima, width=0.55, color=["#7C3AED", "#059669"])
    axes[2].axhline(0.5, color="#DC2626", linestyle="--", linewidth=1.5)
    for position, (maximum, count) in enumerate(zip(maxima, outliers)):
        axes[2].text(
            position,
            maximum + 0.015,
            f"{maximum:.3f}\n{count} outliers",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    for axis, title, ylabel in (
        (axes[0], "K=0 → K=3 full-image change", "Point Rel reduction (%)"),
        (axes[1], "K=0 → K=3 structure change", "Point Rel reduction (%)"),
        (axes[2], "Residual safety over 132 images", "max |Δ log Z|"),
    ):
        axis.set_xticks(x, [LABELS[name] for name in variants])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
    axes[0].axhline(0, color="#111827", linewidth=1)
    axes[1].axhline(0, color="#111827", linewidth=1)
    axes[0].legend(frameon=True)
    figure.suptitle(
        "Exp22 · Batch-independent SSR normalization screen",
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
