from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制Exp19重校准结果。")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(args.summary.read_text(encoding="utf-8"))
    names = ["Original", "Recal. B=1", "Recal. B=8"]
    residuals = [
        report["baseline"]["residual"]["maximum"],
        report["arms"]["b1"]["residual"]["maximum"],
        report["arms"]["b8"]["residual"]["maximum"],
    ]
    outliers = [
        report["baseline"]["residual"]["outlier_count"],
        report["arms"]["b1"]["residual"]["outlier_count"],
        report["arms"]["b8"]["residual"]["outlier_count"],
    ]
    colors = ["#d94b4b", "#4b78d9", "#27a6a1"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    bars = axes[0].bar(names, residuals, color=colors, alpha=0.9)
    axes[0].axhline(0.5, color="#222222", linestyle=":", linewidth=1.8)
    axes[0].set_ylabel(r"Maximum $|\Delta \log Z|$")
    axes[0].set_title("Eval-mode residual outliers")
    axes[0].grid(axis="y", alpha=0.25)
    for bar, value, count in zip(bars, residuals, outliers, strict=True):
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.025,
            f"{value:.3f}\n({count} outliers)",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    axes[0].set_ylim(0, max(residuals) * 1.22)

    labels = [
        "Train full",
        "Train structure",
        "Val full",
        "Val structure",
        "Test full",
        "Test structure",
    ]
    changes = {}
    for arm in ("b1", "b8"):
        values = []
        for split in ("train", "val", "test"):
            source = report["arms"][arm]["k3_changes_from_original"][split]
            values.extend(
                [
                    100.0 * source["k3_point_rel_relative_reduction"],
                    100.0 * source[
                        "k3_structure_point_rel_relative_reduction"
                    ],
                ]
            )
        changes[arm] = values
    positions = np.arange(len(labels))
    width = 0.36
    axes[1].bar(
        positions - width / 2,
        changes["b1"],
        width,
        label="Recal. B=1",
        color=colors[1],
    )
    axes[1].bar(
        positions + width / 2,
        changes["b8"],
        width,
        label="Recal. B=8",
        color=colors[2],
    )
    axes[1].axhline(0, color="#222222", linewidth=1)
    axes[1].set_xticks(positions, labels, rotation=24, ha="right")
    axes[1].set_ylabel("K=3 Point Rel reduction vs. original (%)")
    axes[1].set_title("Geometry change without weight updates")
    axes[1].legend(frameon=False)
    axes[1].grid(axis="y", alpha=0.25)

    fig.suptitle("Exp19 — SSR BatchNorm running-statistics recalibration", fontsize=15)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(args.output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
