from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np


NUMERIC_FLOATS = {
    "max_abs_residual",
    "max_signed_residual",
    "p99_abs_residual",
    "p999_abs_residual",
    "mean_abs_residual",
    "cumulative_max_abs_residual",
    "depth_span",
}
NUMERIC_INTS = {
    "repeat",
    "iteration",
    "pixels_exceeding_threshold",
    "active_voxels",
    "max_row",
    "max_col",
}


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for raw in csv.DictReader(handle):
            row: dict[str, Any] = dict(raw)
            for key in NUMERIC_FLOATS:
                row[key] = float(row[key])
            for key in NUMERIC_INTS:
                row[key] = int(row[key])
            row["highlighted"] = row["highlighted"] == "True"
            rows.append(row)
    if not rows:
        raise ValueError("Residual CSV is empty")
    return rows


def distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "median": float(np.quantile(array, 0.5)),
        "p90": float(np.quantile(array, 0.9)),
        "p95": float(np.quantile(array, 0.95)),
        "p99": float(np.quantile(array, 0.99)),
        "maximum": float(array.max()),
    }


def sample_maxima(
    rows: Sequence[dict[str, Any]],
    mode: str,
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row["mode"] == mode:
            grouped[str(row["id"])].append(float(row["max_abs_residual"]))
    return {sample_id: max(values) for sample_id, values in grouped.items()}


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
    failure_batch_ids: set[str],
    exp16_failure_maximum: float,
) -> dict[str, Any]:
    eval_maxima = sample_maxima(rows, "eval")
    train_maxima = sample_maxima(rows, "train")
    failure_train = {
        key: value for key, value in train_maxima.items() if key in failure_batch_ids
    }
    failure_eval = {
        key: value for key, value in eval_maxima.items() if key in failure_batch_ids
    }
    if set(failure_train) != failure_batch_ids or set(failure_eval) != failure_batch_ids:
        raise ValueError("Not every Exp16 failure-batch sample was scanned")

    by_mode_iteration: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_mode_iteration[(str(row["mode"]), int(row["iteration"]))].append(row)
    iteration_summary = []
    for (mode, iteration), selected in sorted(by_mode_iteration.items()):
        worst = max(selected, key=lambda row: float(row["max_abs_residual"]))
        iteration_summary.append(
            {
                "mode": mode,
                "iteration": iteration,
                "maximum": float(worst["max_abs_residual"]),
                "sample_id": str(worst["id"]),
                "unique_samples_exceeding_threshold": len(
                    {
                        str(row["id"])
                        for row in selected
                        if float(row["max_abs_residual"]) > threshold
                    }
                ),
            }
        )

    eval_outliers = sorted(
        sample_id for sample_id, value in eval_maxima.items() if value > threshold
    )
    train_outliers = sorted(
        sample_id for sample_id, value in train_maxima.items() if value > threshold
    )
    failure_train_maximum = max(failure_train.values())
    return {
        "status": "complete",
        "threshold": threshold,
        "row_count": len(rows),
        "sample_count": len(eval_maxima),
        "mode_distributions": {
            "eval": distribution(list(eval_maxima.values())),
            "train": distribution(list(train_maxima.values())),
        },
        "outliers": {
            "eval": eval_outliers,
            "eval_count": len(eval_outliers),
            "train": train_outliers,
            "train_count": len(train_outliers),
        },
        "iteration_summary": iteration_summary,
        "exp16_failure_batch": {
            "ids": sorted(failure_batch_ids),
            "baseline_eval_maximum": max(failure_eval.values()),
            "baseline_eval_maximum_id": max(failure_eval, key=failure_eval.get),
            "baseline_train_maximum": failure_train_maximum,
            "baseline_train_maximum_id": max(failure_train, key=failure_train.get),
            "exp16_step820_train_maximum": exp16_failure_maximum,
            "relative_increase_over_baseline_failure_batch_train_maximum": (
                exp16_failure_maximum / failure_train_maximum - 1.0
            ),
        },
        "decision": {
            "baseline_train_mode_exceeds_threshold": bool(train_outliers),
            "baseline_eval_mode_exceeds_threshold": bool(eval_outliers),
            "joint_update_amplification_is_supported": (
                not train_outliers
                and exp16_failure_maximum > failure_train_maximum
            ),
            "batchnorm_train_eval_mismatch_is_supported": (
                bool(eval_outliers) and not train_outliers
            ),
        },
    }


def plot_summary(
    rows: Sequence[dict[str, Any]],
    summary: dict[str, Any],
    output: Path,
) -> None:
    eval_values = sorted(sample_maxima(rows, "eval").values())
    train_values = sorted(sample_maxima(rows, "train").values())
    threshold = float(summary["threshold"])

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axes[0].plot(
        np.linspace(0, 100, len(eval_values)),
        eval_values,
        linewidth=2.0,
        label="eval mode",
        color="#DC2626",
    )
    axes[0].plot(
        np.linspace(0, 100, len(train_values)),
        train_values,
        linewidth=2.0,
        label="train mode (worst of 3)",
        color="#2563EB",
    )
    axes[0].axhline(
        threshold,
        color="#111827",
        linestyle="--",
        linewidth=1.2,
        label="safety threshold",
    )
    axes[0].set(
        title="Per-sample worst SSR residual",
        xlabel="Sorted sample percentile (%)",
        ylabel="max |delta log Z|",
    )
    axes[0].legend(frameon=True)

    labels = ["K=1", "K=2", "K=3"]
    eval_iteration = [
        entry["maximum"]
        for entry in summary["iteration_summary"]
        if entry["mode"] == "eval"
    ]
    train_iteration = [
        entry["maximum"]
        for entry in summary["iteration_summary"]
        if entry["mode"] == "train"
    ]
    positions = np.arange(3)
    width = 0.36
    axes[1].bar(
        positions - width / 2,
        eval_iteration,
        width,
        label="eval mode",
        color="#F87171",
    )
    axes[1].bar(
        positions + width / 2,
        train_iteration,
        width,
        label="train mode",
        color="#60A5FA",
    )
    axes[1].axhline(threshold, color="#111827", linestyle="--", linewidth=1.2)
    axes[1].set_xticks(positions, labels)
    axes[1].set(
        title="Worst residual by recurrence",
        xlabel="SSR iteration",
        ylabel="max |delta log Z|",
    )
    axes[1].legend(frameon=True)
    figure.suptitle("Exp17 — Exp15 checkpoint residual audit", fontsize=14)
    figure.savefig(output / "residual_audit.png", dpi=180)
    figure.savefig(output / "residual_audit.pdf")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--exp16-failure-maximum", type=float, default=0.5331000089645386)
    args = parser.parse_args()

    rows = load_rows(args.csv)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    args.output.mkdir(parents=True, exist_ok=True)
    summary = summarize(
        rows,
        threshold=args.threshold,
        failure_batch_ids=set(config["exp16_step820_batch"]),
        exp16_failure_maximum=args.exp16_failure_maximum,
    )
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
