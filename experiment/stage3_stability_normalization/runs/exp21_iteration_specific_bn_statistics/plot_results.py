from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制Exp21逐轮统计结果。")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    policies = {
        "Original": report["baseline"],
        "Mixed B=8": report["prior_controls"]["mixed_batch8"],
        "Per-iteration B=8": report["iteration_specific"],
        "Per-image": report["prior_controls"]["per_image"],
    }
    colors = ["#d94b4b", "#2ba39f", "#e59c37", "#4b78d9"]

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.4))
    names = list(policies)
    residuals = [policies[name]["residual"]["maximum"] for name in names]
    outliers = [policies[name]["residual"]["outlier_count"] for name in names]
    bars = axes[0].bar(names, residuals, color=colors, alpha=0.9)
    axes[0].axhline(0.5, color="#222222", linestyle=":", linewidth=1.8)
    axes[0].set_ylabel(r"Maximum $|\Delta \log Z|$")
    axes[0].set_title("Residual safety")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].set_ylim(0, max(residuals) * 1.23)
    axes[0].tick_params(axis="x", rotation=15)
    for bar, value, count in zip(bars, residuals, outliers, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}\n({count})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    labels = [
        "Train full",
        "Train structure",
        "Val full",
        "Val structure",
        "Test full",
        "Test structure",
    ]
    positions = np.arange(len(labels))
    plotted = (
        ("Mixed B=8", report["prior_controls"]["mixed_batch8"], colors[1]),
        ("Per-iteration B=8", report["iteration_specific"], colors[2]),
        ("Per-image", report["prior_controls"]["per_image"], colors[3]),
    )
    width = 0.25
    for index, (label, policy, color) in enumerate(plotted):
        values = []
        changes = policy["k3_changes_from_original"]
        for split in ("train", "val", "test"):
            values.extend(
                [
                    100 * changes[split]["k3_point_rel_relative_reduction"],
                    100
                    * changes[split][
                        "k3_structure_point_rel_relative_reduction"
                    ],
                ]
            )
        axes[1].bar(
            positions + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
    axes[1].axhline(0, color="#222222", linewidth=1)
    axes[1].set_xticks(positions, labels, rotation=24, ha="right")
    axes[1].set_ylabel("K=3 Point Rel reduction vs. original (%)")
    axes[1].set_title("Accuracy–stability trade-off")
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle("Exp21 — SSR normalization-policy audit", fontsize=15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
