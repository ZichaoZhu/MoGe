from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


COLORS = {
    "microbatch1": "#64748B",
    "microbatch2": "#2563EB",
    "train": "#2563EB",
    "val": "#F59E0B",
    "test": "#DC2626",
}


def relative_reduction(base: float, refined: float) -> float:
    return 100.0 * (base - refined) / base


def moving_average(values: Iterable[float], window: int) -> np.ndarray:
    array = np.asarray(tuple(values), dtype=np.float64)
    if window <= 0:
        raise ValueError("window must be positive")
    if array.size < window:
        return np.full(array.shape, np.nan, dtype=np.float64)
    cumulative = np.cumsum(np.insert(array, 0, 0.0))
    averaged = (cumulative[window:] - cumulative[:-window]) / window
    return np.concatenate((np.full(window - 1, np.nan), averaged))


def load_history(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        key: np.asarray([float(row[key]) for row in rows], dtype=np.float64)
        for key in (
            "step",
            "loss",
            "ssr_max_abs_log_depth_residual",
        )
    }


def scope_reductions(metrics: dict) -> list[float]:
    output = []
    for split in ("train", "val", "test"):
        for scope in ("full", "structure"):
            key = f"{scope}_point_rel_relative_reduction"
            output.append(
                100.0 * metrics["changes_from_k0"]["3"][split][key]
            )
    return output


def residual_maximum(residual: dict) -> float:
    return float(residual["global_maximum"]["max_abs_residual"])


def plot_k_sweep(axis, metrics: dict, metric: str, title: str) -> None:
    steps = (0, 1, 3, 5)
    for split in ("train", "val", "test"):
        axis.plot(
            steps,
            [
                100.0 * metrics["metrics_by_k"][str(step)][split][metric]
                for step in steps
            ],
            marker="o",
            color=COLORS[split],
            label=split,
        )
    axis.set_xticks(steps)
    axis.set_xlabel("Refinement steps K")
    axis.set_ylabel("Point Rel (%)")
    axis.set_title(title)
    axis.legend()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--microbatch1-history", type=Path, required=True)
    parser.add_argument("--microbatch2-history", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    report = json.loads(args.report.read_text(encoding="utf-8"))
    histories = {
        "microbatch1": load_history(args.microbatch1_history),
        "microbatch2": load_history(args.microbatch2_history),
    }
    metrics1 = report["microbatch1"]["metrics"]
    metrics2 = report["microbatch2"]["metrics"]
    residual1 = report["microbatch1"]["residual"]
    residual2 = report["microbatch2"]["residual"]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 3, figsize=(19, 10))

    for variant, history in histories.items():
        axes[0, 0].plot(
            history["step"],
            moving_average(history["loss"], 25),
            color=COLORS[variant],
            label=variant,
        )
        axes[0, 1].plot(
            history["step"],
            moving_average(history["ssr_max_abs_log_depth_residual"], 25),
            color=COLORS[variant],
            label=variant,
        )
    axes[0, 0].set_title("Training loss · 25-step moving average")
    axes[0, 0].set_xlabel("Optimization step")
    axes[0, 0].set_ylabel("Geometry loss")
    axes[0, 0].legend()
    axes[0, 1].axhline(0.5, color="#DC2626", linestyle="--", label="safety limit")
    axes[0, 1].set_title("Train-mode residual · 25-step moving average")
    axes[0, 1].set_xlabel("Optimization step")
    axes[0, 1].set_ylabel("max |Δ log Z|")
    axes[0, 1].legend()

    labels = (
        "train\nfull",
        "train\nstructure",
        "val\nfull",
        "val\nstructure",
        "test\nfull",
        "test\nstructure",
    )
    x = np.arange(len(labels))
    axes[0, 2].bar(
        x - 0.18,
        scope_reductions(metrics1),
        0.36,
        color=COLORS["microbatch1"],
        label="microbatch 1",
    )
    axes[0, 2].bar(
        x + 0.18,
        scope_reductions(metrics2),
        0.36,
        color=COLORS["microbatch2"],
        label="microbatch 2",
    )
    axes[0, 2].axhline(0, color="#111827", linewidth=1)
    axes[0, 2].set_xticks(x, labels)
    axes[0, 2].set_ylabel("K=0 → K=3 Point Rel reduction (%)")
    axes[0, 2].set_title("SSR independent contribution at step 800")
    axes[0, 2].legend()

    plot_k_sweep(axes[1, 0], metrics2, "point_rel", "Microbatch 2 · full image")
    plot_k_sweep(
        axes[1, 1],
        metrics2,
        "structure_point_rel",
        "Microbatch 2 · locked structure",
    )

    maxima = (residual_maximum(residual1), residual_maximum(residual2))
    outliers = (
        residual1["unique_sample_count_exceeding_threshold"],
        residual2["unique_sample_count_exceeding_threshold"],
    )
    safety_x = np.arange(2)
    safety_labels = ("microbatch 1", "microbatch 2")
    axes[1, 2].bar(
        safety_x,
        maxima,
        color=(COLORS["microbatch1"], COLORS["microbatch2"]),
    )
    axes[1, 2].axhline(0.5, color="#DC2626", linestyle="--")
    axes[1, 2].set_ylim(0, max(maxima) * 1.20)
    for index, (maximum, count) in enumerate(zip(maxima, outliers)):
        axes[1, 2].text(
            index,
            maximum + max(maxima) * 0.035,
            f"max={maximum:.3f}\noutliers={count}",
            ha="center",
            va="bottom",
        )
    axes[1, 2].set_xticks(safety_x, safety_labels)
    axes[1, 2].set_ylabel("Eval-mode max |Δ log Z|")
    axes[1, 2].set_title("132-image recursive residual audit")

    figure.suptitle(
        "Exp24 · Long detached microbatch-2 BatchNorm audit",
        fontsize=15,
        fontweight="bold",
    )
    figure.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".png"), dpi=200)
    figure.savefig(args.output.with_suffix(".pdf"))
    plt.close(figure)


if __name__ == "__main__":
    main()
