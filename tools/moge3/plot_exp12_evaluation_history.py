from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


COLORS = {"train": "#2563EB", "val": "#DC2626"}
METRICS = {
    "point_rel": ("Point Rel", "lower is better"),
    "depth_rel": ("Depth Rel", "lower is better"),
    "depth_delta_1.01": (r"Depth $\delta_{1.01}$", "higher is better"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the raw periodic evaluations from Exp12."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--best-step", type=int, default=3300)
    parser.add_argument("--stop-step", type=int, default=3600)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in csv.DictReader(file)
        ]
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}")
    return rows


def series(
    rows: list[dict[str, float]], column: str, *, percent: bool = True
) -> np.ndarray:
    scale = 100.0 if percent else 1.0
    return np.asarray([row[column] * scale for row in rows])


def decorate(axis, steps: np.ndarray, *, best_step: int, stop_step: int) -> None:
    axis.axvline(
        best_step,
        color="#059669",
        linestyle=":",
        linewidth=1.6,
        label="Best train K=3",
    )
    axis.axvspan(
        best_step,
        stop_step,
        color="#FEE2E2",
        alpha=0.36,
        linewidth=0,
        zorder=0,
    )
    axis.axvline(
        stop_step,
        color="#991B1B",
        linestyle="--",
        linewidth=1.4,
        label="Plateau stop",
    )
    margin = max(15.0, 0.04 * (float(steps[-1]) - float(steps[0])))
    axis.set_xlim(float(steps[0]) - margin, float(steps[-1]) + margin)
    axis.set_xlabel("Optimization step")
    axis.grid(True, which="major", alpha=0.24)
    axis.grid(True, which="minor", alpha=0.10)
    axis.minorticks_on()


def plot_metric(
    axis,
    rows: list[dict[str, float]],
    metric: str,
    *,
    best_step: int,
    stop_step: int,
) -> None:
    steps = series(rows, "step", percent=False)
    title, direction = METRICS[metric]
    for split in ("train", "val"):
        for refinement_step, linestyle, alpha in (
            (0, "--", 0.72),
            (3, "-", 1.0),
        ):
            axis.plot(
                steps,
                series(rows, f"{split}/k{refinement_step}_{metric}"),
                color=COLORS[split],
                linestyle=linestyle,
                linewidth=1.9 if refinement_step == 0 else 2.35,
                marker="o",
                markersize=4.0,
                alpha=alpha,
                label=f"{split.title()} K={refinement_step}",
            )
    decorate(axis, steps, best_step=best_step, stop_step=stop_step)
    axis.set_ylabel(f"{title} (%)")
    axis.set_title(f"{title} — {direction}", loc="left")


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(15.2, 10.0))
    for axis, metric in zip(
        axes.flat[:3], ("point_rel", "depth_rel", "depth_delta_1.01")
    ):
        plot_metric(
            axis,
            rows,
            metric,
            best_step=args.best_step,
            stop_step=args.stop_step,
        )

    steps = series(rows, "step", percent=False)
    contribution_axis = axes[1, 1]
    for split in ("train", "val"):
        k0 = series(rows, f"{split}/k0_point_rel", percent=False)
        k3 = series(rows, f"{split}/k3_point_rel", percent=False)
        reduction = 100.0 * (k0 - k3) / np.maximum(k0, 1e-12)
        contribution_axis.plot(
            steps,
            reduction,
            color=COLORS[split],
            linewidth=2.35,
            marker="o",
            markersize=4.0,
            label=split.title(),
        )
    contribution_axis.axhline(0.0, color="#111827", linewidth=1.0)
    decorate(
        contribution_axis,
        steps,
        best_step=args.best_step,
        stop_step=args.stop_step,
    )
    contribution_axis.set_ylabel("Relative Point Rel reduction (%)")
    contribution_axis.set_title(
        "SSR contribution: K=0 → K=3\npositive means SSR helps",
        loc="left",
    )

    handles, labels = axes[0, 0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    figure.legend(
        unique.values(),
        unique.keys(),
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        frameon=True,
        fontsize=9,
    )
    figure.suptitle(
        "Exp12: immediate joint fine-tuning from Exp11 step 3000",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.012,
        "Raw full-set evaluations every 100 steps; no smoothing. "
        "The process stopped after five evaluations without a 0.2% "
        "relative train K=3 improvement.",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.91))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for suffix in (".png", ".pdf"):
        figure.savefig(
            args.output_dir / f"evaluation_dashboard{suffix}",
            dpi=220 if suffix == ".png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
    plt.close(figure)


if __name__ == "__main__":
    main()
