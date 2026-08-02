from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable

import numpy as np


METRICS = {
    "point_rel": ("Point Rel", "lower"),
    "depth_rel": ("Depth Rel", "lower"),
    "depth_delta_1.01": (r"Depth $\delta_{1.01}$", "higher"),
    "boundary_f1": ("Depth-boundary F1", "higher"),
}

SPLIT_COLORS = {
    "train": "#2563EB",
    "val": "#DC2626",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot the original Exp11 periodic evaluation trajectory."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--best-step",
        type=int,
        default=3000,
    )
    parser.add_argument(
        "--stop-step",
        type=int,
        default=4478,
    )
    return parser.parse_args()


def load_history(path: Path) -> list[dict[str, float]]:
    with path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        rows = [
            {key: float(value) for key, value in row.items()}
            for row in reader
        ]
    if not rows:
        raise ValueError(f"No evaluation rows found in {path}")
    required = {"step"}
    for split in ("train", "val"):
        for refinement_step in (0, 3):
            required.update(
                f"{split}/k{refinement_step}_{metric}"
                for metric in METRICS
            )
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    return rows


def values(
    rows: Iterable[dict[str, float]],
    column: str,
    *,
    percent: bool = True,
) -> np.ndarray:
    scale = 100.0 if percent else 1.0
    return np.asarray([row[column] * scale for row in rows])


def decorate_axis(axis, *, best_step: int, stop_step: int) -> None:
    axis.axvspan(
        best_step,
        stop_step,
        color="#FEE2E2",
        alpha=0.34,
        linewidth=0,
        zorder=0,
    )
    axis.axvline(
        best_step,
        color="#059669",
        linestyle=":",
        linewidth=1.5,
        alpha=0.9,
    )
    axis.axvline(
        stop_step,
        color="#991B1B",
        linestyle="--",
        linewidth=1.3,
        alpha=0.85,
    )
    axis.set_xlim(0, max(stop_step + 100, axis.get_xlim()[1]))
    axis.set_xlabel("Optimization step")
    axis.grid(True, which="major", alpha=0.24)
    axis.grid(True, which="minor", alpha=0.1)
    axis.minorticks_on()


def plot_metric(
    axis,
    rows: list[dict[str, float]],
    metric: str,
    *,
    best_step: int,
    stop_step: int,
    show_legend: bool,
) -> None:
    steps = values(rows, "step", percent=False)
    title, direction = METRICS[metric]
    for split in ("train", "val"):
        for refinement_step, linestyle, alpha in (
            (0, "--", 0.7),
            (3, "-", 1.0),
        ):
            axis.plot(
                steps,
                values(rows, f"{split}/k{refinement_step}_{metric}"),
                color=SPLIT_COLORS[split],
                linestyle=linestyle,
                linewidth=1.8 if refinement_step == 0 else 2.35,
                marker="o",
                markersize=3.4,
                alpha=alpha,
                label=f"{split.title()} K={refinement_step}",
            )
    decorate_axis(axis, best_step=best_step, stop_step=stop_step)
    axis.set_ylabel(f"{title} (%)")
    axis.set_title(f"{title} — {direction} is better", loc="left")
    if show_legend:
        axis.legend(
            ncol=2,
            fontsize=8.5,
            loc="upper left",
            frameon=True,
        )


def save_figure(figure, output_prefix: Path) -> None:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_prefix.with_suffix(".png"),
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    figure.savefig(
        output_prefix.with_suffix(".pdf"),
        bbox_inches="tight",
        facecolor="white",
    )


def plot_point_rel_overview(
    rows: list[dict[str, float]],
    output_dir: Path,
    *,
    best_step: int,
    stop_step: int,
) -> None:
    import matplotlib.pyplot as plt

    steps = values(rows, "step", percent=False)
    train_k3 = values(rows, "train/k3_point_rel")
    best_index = int(np.argmin(train_k3))
    best_value = float(train_k3[best_index])
    best_eval_step = int(steps[best_index])

    figure, axis = plt.subplots(figsize=(12.4, 6.9))
    plot_metric(
        axis,
        rows,
        "point_rel",
        best_step=best_step,
        stop_step=stop_step,
        show_legend=True,
    )
    axis.scatter(
        [best_eval_step],
        [best_value],
        marker="*",
        s=180,
        color="#059669",
        edgecolor="white",
        linewidth=0.8,
        zorder=5,
    )
    axis.annotate(
        f"Best train K=3\n{best_value:.3f}% @ step {best_eval_step}",
        xy=(best_eval_step, best_value),
        xytext=(best_eval_step - 1120, best_value + 1.25),
        arrowprops={"arrowstyle": "->", "color": "#047857"},
        color="#065F46",
        fontsize=10,
        fontweight="bold",
    )
    axis.text(
        stop_step - 40,
        axis.get_ylim()[1] * 0.985,
        "Safety stop\nstep 4478",
        ha="right",
        va="top",
        color="#991B1B",
        fontsize=9,
    )
    axis.text(
        (best_step + stop_step) / 2,
        axis.get_ylim()[0] + 0.06 * np.ptp(axis.get_ylim()),
        "post-best drift",
        ha="center",
        color="#991B1B",
        fontsize=9,
    )
    figure.suptitle(
        "Exp11 original run: periodic Point Rel trajectory",
        fontsize=16,
        fontweight="bold",
        y=0.99,
    )
    figure.text(
        0.5,
        0.012,
        "Raw evaluations every 250 steps; no smoothing. "
        "All recorded points belong to Stage 1 (detached).",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.96))
    save_figure(figure, output_dir / "original_point_rel_curve")
    plt.close(figure)


def plot_dashboard(
    rows: list[dict[str, float]],
    output_dir: Path,
    *,
    best_step: int,
    stop_step: int,
) -> None:
    import matplotlib.pyplot as plt

    steps = values(rows, "step", percent=False)
    figure, axes = plt.subplots(2, 3, figsize=(17.0, 10.1))
    for axis, metric in zip(
        axes.flat[:4],
        ("point_rel", "depth_rel", "depth_delta_1.01", "boundary_f1"),
    ):
        plot_metric(
            axis,
            rows,
            metric,
            best_step=best_step,
            stop_step=stop_step,
            show_legend=metric == "point_rel",
        )

    ssr_axis = axes[1, 1]
    for split in ("train", "val"):
        k0 = values(rows, f"{split}/k0_point_rel", percent=False)
        k3 = values(rows, f"{split}/k3_point_rel", percent=False)
        improvement = 100.0 * (k0 - k3) / np.maximum(k0, 1e-12)
        ssr_axis.plot(
            steps,
            improvement,
            color=SPLIT_COLORS[split],
            linewidth=2.25,
            marker="o",
            markersize=3.4,
            label=split.title(),
        )
    ssr_axis.axhline(0.0, color="#111827", linewidth=1.0)
    decorate_axis(ssr_axis, best_step=best_step, stop_step=stop_step)
    ssr_axis.set_ylabel("Relative Point Rel reduction (%)")
    ssr_axis.set_title(
        "SSR contribution: K=0 → K=3\npositive means SSR helps",
        loc="left",
    )
    ssr_axis.legend(fontsize=8.5)

    gap_axis = axes[1, 2]
    for refinement_step, linestyle in ((0, "--"), (3, "-")):
        gap = (
            values(rows, f"val/k{refinement_step}_point_rel")
            - values(rows, f"train/k{refinement_step}_point_rel")
        )
        gap_axis.plot(
            steps,
            gap,
            color="#7C3AED",
            linestyle=linestyle,
            linewidth=2.15,
            marker="o",
            markersize=3.4,
            label=f"K={refinement_step}",
        )
    decorate_axis(gap_axis, best_step=best_step, stop_step=stop_step)
    gap_axis.set_ylabel("Validation − training Point Rel (pp)")
    gap_axis.set_title(
        "Generalization gap\nsmaller is better",
        loc="left",
    )
    gap_axis.legend(fontsize=8.5)

    figure.suptitle(
        "Exp11 original run: evaluation dashboard",
        fontsize=17,
        fontweight="bold",
        y=0.995,
    )
    figure.text(
        0.5,
        0.008,
        "Raw evaluations; no smoothing. Green dotted line: best train K=3 "
        "(step 3000). Red dashed line: safety stop (step 4478).",
        ha="center",
        fontsize=9,
        color="#4B5563",
    )
    figure.tight_layout(rect=(0, 0.026, 1, 0.97))
    save_figure(figure, output_dir / "original_metrics_dashboard")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    rows = load_history(args.input)

    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-whitegrid")
    plot_point_rel_overview(
        rows,
        args.output_dir,
        best_step=args.best_step,
        stop_step=args.stop_step,
    )
    plot_dashboard(
        rows,
        args.output_dir,
        best_step=args.best_step,
        stop_step=args.stop_step,
    )


if __name__ == "__main__":
    main()
