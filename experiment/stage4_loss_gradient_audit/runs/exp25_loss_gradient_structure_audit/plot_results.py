from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


OBJECTIVES = ("global", "local", "edge_paper_min", "combined_paper")
COLORS = {
    "global": "#4C78A8",
    "local": "#F58518",
    "edge_paper_min": "#E45756",
    "combined_paper": "#54A24B",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def mean_metric(
    rows: Sequence[dict[str, Any]],
    *,
    split: str,
    step: int,
    objective: str,
    metric: str,
) -> float:
    selected = [
        float(row[metric])
        for row in rows
        if row["split"] == split
        and int(row["k"]) == step
        and row["objective"] == objective
    ]
    if not selected:
        raise ValueError(
            f"No rows for split={split}, K={step}, objective={objective}"
        )
    return float(np.mean(selected))


def paired_bars(
    axis: plt.Axes,
    rows: Sequence[dict[str, Any]],
    *,
    metric: str,
    title: str,
    ylabel: str,
    log_scale: bool = False,
) -> None:
    x = np.arange(len(OBJECTIVES))
    width = 0.34
    for offset, step, alpha in ((-width / 2, 0, 0.72), (width / 2, 3, 1.0)):
        values = [
            mean_metric(
                rows,
                split="train",
                step=step,
                objective=objective,
                metric=metric,
            )
            for objective in OBJECTIVES
        ]
        bars = axis.bar(
            x + offset,
            values,
            width,
            label=f"K={step}",
            color=[COLORS[objective] for objective in OBJECTIVES],
            alpha=alpha,
            edgecolor="white",
            linewidth=0.6,
        )
        axis.bar_label(
            bars,
            labels=[f"{value:.3g}" for value in values],
            fontsize=7,
            padding=2,
        )
    axis.set_xticks(x, ("Global", "Local", "Edge", "Combined"))
    axis.set_title(title)
    axis.set_ylabel(ylabel)
    if log_scale:
        axis.set_yscale("log")
    axis.legend(frameon=False)


def plot(input_csv: Path, output_prefix: Path) -> None:
    rows = load_rows(input_csv)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(15, 10))
    paired_bars(
        axes[0, 0],
        rows,
        metric="loss",
        title="Objective scalar values on locked train ROIs",
        ylabel="Loss (log scale)",
        log_scale=True,
    )
    paired_bars(
        axes[0, 1],
        rows,
        metric="gradient_l1_mass",
        title=r"Direct objective pressure on per-pixel $\zeta=\log Z$",
        ylabel="Gradient L1 mass",
    )
    paired_bars(
        axes[1, 0],
        rows,
        metric="structure_gradient_enrichment",
        title="Locked structure gradient enrichment",
        ylabel="Gradient share / pixel share",
    )
    paired_bars(
        axes[1, 1],
        rows,
        metric="gt_boundary_gradient_enrichment",
        title="GT depth-boundary gradient enrichment",
        ylabel="Gradient share / pixel share",
    )
    for axis in axes[1]:
        axis.axhline(
            1.0,
            color="#222222",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="Uniform per-pixel mass",
        )
    figure.suptitle(
        "Exp25 · Loss-gradient and fine-structure supervision audit",
        fontsize=16,
        fontweight="bold",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_prefix.with_suffix(".png"), dpi=220)
    figure.savefig(output_prefix.with_suffix(".pdf"))
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    plot(args.input, args.output_prefix)


if __name__ == "__main__":
    main()
