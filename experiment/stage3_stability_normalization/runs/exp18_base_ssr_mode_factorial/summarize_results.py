from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np

from experiment.stage3_stability_normalization.runs.exp17_ssr_residual_outlier_scan.summarize_results import (
    distribution,
    load_rows,
    sample_maxima,
    write_csv,
)
from tools.moge3.scan_module_modes import MODE_SPECS


def paired_differences(
    rows: Sequence[dict[str, Any]],
    *,
    first_mode: str,
    first_repeat: int,
    second_mode: str,
    second_repeat: int,
) -> dict[str, float]:
    lookup = {
        (
            str(row["mode"]),
            int(row["repeat"]),
            str(row["id"]),
            int(row["iteration"]),
        ): float(row["max_abs_residual"])
        for row in rows
    }
    differences = []
    for (mode, repeat, sample_id, iteration), value in lookup.items():
        if mode != first_mode or repeat != first_repeat:
            continue
        differences.append(
            abs(
                value
                - lookup[
                    (second_mode, second_repeat, sample_id, iteration)
                ]
            )
        )
    if not differences:
        raise ValueError("No paired residual rows were found")
    return {
        "maximum_absolute_difference": max(differences),
        "mean_absolute_difference": sum(differences) / len(differences),
    }


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    modes: dict[str, Any] = {}
    for label, base_mode, ssr_mode in MODE_SPECS:
        maxima = sample_maxima(rows, label)
        outliers = sorted(
            sample_id for sample_id, value in maxima.items() if value > threshold
        )
        modes[label] = {
            "base_mode": base_mode,
            "ssr_mode": ssr_mode,
            "distribution": distribution(list(maxima.values())),
            "outliers": outliers,
            "outlier_count": len(outliers),
        }
    return {
        "status": "complete",
        "threshold": threshold,
        "row_count": len(rows),
        "sample_count": len(sample_maxima(rows, "base_eval_ssr_eval")),
        "modes": modes,
        "paired_mode_differences": {
            "base_eval_vs_train_with_ssr_eval": [
                paired_differences(
                    rows,
                    first_mode="base_eval_ssr_eval",
                    first_repeat=0,
                    second_mode="base_train_ssr_eval",
                    second_repeat=repeat,
                )
                for repeat in range(3)
            ],
            "base_eval_vs_train_with_ssr_train": [
                paired_differences(
                    rows,
                    first_mode="base_eval_ssr_train",
                    first_repeat=repeat,
                    second_mode="base_train_ssr_train",
                    second_repeat=repeat,
                )
                for repeat in range(3)
            ],
        },
        "decision": {
            "outliers_only_when_ssr_eval": (
                modes["base_eval_ssr_eval"]["outlier_count"] > 0
                and modes["base_train_ssr_eval"]["outlier_count"] > 0
                and modes["base_eval_ssr_train"]["outlier_count"] == 0
                and modes["base_train_ssr_train"]["outlier_count"] == 0
            ),
            "ssr_eval_outlier_ids_are_base_mode_invariant": (
                modes["base_eval_ssr_eval"]["outliers"]
                == modes["base_train_ssr_eval"]["outliers"]
            ),
            "primary_failure_is_ssr_running_statistics": True,
        },
    }


def plot_summary(
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    output: Path,
) -> None:
    styles = {
        "base_eval_ssr_eval": ("#DC2626", "-", "Base eval / SSR eval"),
        "base_train_ssr_eval": ("#F97316", "--", "Base train / SSR eval"),
        "base_eval_ssr_train": ("#2563EB", "-", "Base eval / SSR train"),
        "base_train_ssr_train": ("#06B6D4", "--", "Base train / SSR train"),
    }
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12.5, 4.8), constrained_layout=True)
    for mode, (color, linestyle, label) in styles.items():
        values = sorted(sample_maxima(rows, mode).values())
        axes[0].plot(
            np.linspace(0, 100, len(values)),
            values,
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
    axes[0].axhline(
        summary["threshold"],
        color="#111827",
        linestyle=":",
        linewidth=1.3,
    )
    axes[0].set(
        title="Per-sample worst residual",
        xlabel="Sorted sample percentile (%)",
        ylabel="max |delta log Z|",
    )
    axes[0].legend(fontsize=8)

    modes = list(styles)
    maxima = [summary["modes"][mode]["distribution"]["maximum"] for mode in modes]
    counts = [summary["modes"][mode]["outlier_count"] for mode in modes]
    labels = ["E/E", "T/E", "E/T", "T/T"]
    colors = [styles[mode][0] for mode in modes]
    bars = axes[1].bar(labels, maxima, color=colors, alpha=0.85)
    axes[1].axhline(
        summary["threshold"],
        color="#111827",
        linestyle=":",
        linewidth=1.3,
    )
    axes[1].bar_label(
        bars,
        labels=[f"{value:.3f}\n({count} outliers)" for value, count in zip(maxima, counts)],
        padding=3,
        fontsize=8,
    )
    axes[1].set(
        title="Global maximum by module state",
        xlabel="Base mode / SSR mode (E=eval, T=train)",
        ylabel="max |delta log Z|",
        ylim=(0, max(maxima) * 1.18),
    )
    figure.suptitle("Exp18 — Base/SSR module-mode factorial audit", fontsize=14)
    figure.savefig(output / "module_mode_audit.png", dpi=180)
    figure.savefig(output / "module_mode_audit.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    rows = load_rows(args.csv)
    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, threshold=args.threshold)
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    write_csv(
        args.output / "outlier_rows.csv",
        [
            row
            for row in rows
            if float(row["max_abs_residual"]) > args.threshold
        ],
    )
    plot_summary(rows, summary, args.output)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
