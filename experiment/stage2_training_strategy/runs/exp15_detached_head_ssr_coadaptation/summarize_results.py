from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SCOPES = {"full": "", "crop": "crop_", "structure": "structure_"}
SPLITS = ("train", "val", "test")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(
    report: dict[str, Any],
    *,
    k: str,
    split: str,
    scope: str,
    name: str,
) -> float:
    return float(report["metrics_by_k"][k][split][f"{SCOPES[scope]}{name}"])


def reduction(before: float, after: float) -> float:
    return (before - after) / before


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 Exp15 detached 共适应结果。")
    parser.add_argument("--baseline-report", type=pathlib.Path, required=True)
    parser.add_argument("--final-report", type=pathlib.Path, required=True)
    parser.add_argument("--evaluation-history", type=pathlib.Path, required=True)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline = read_json(args.baseline_report)
    final = read_json(args.final_report)
    args.output.mkdir(parents=True, exist_ok=True)

    comparisons: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for split in SPLITS:
        comparisons[split] = {}
        for scope in SCOPES:
            initial = metric(
                baseline, k="0", split=split, scope=scope, name="point_rel"
            )
            values = {
                k: metric(final, k=k, split=split, scope=scope, name="point_rel")
                for k in ("0", "1", "3", "5")
            }
            comparisons[split][scope] = {
                "initial_k0_point_rel": initial,
                **{f"final_k{k}_point_rel": value for k, value in values.items()},
                "base_head_relative_reduction": reduction(initial, values["0"]),
                "final_ssr_k3_relative_reduction": reduction(
                    values["0"], values["3"]
                ),
                "total_relative_reduction": reduction(initial, values["3"]),
                "k5_vs_k3_relative_reduction": reduction(
                    values["3"], values["5"]
                ),
            }
            for stage, k, report in (
                ("initial", "0", baseline),
                *((f"final", k, final) for k in ("0", "1", "3", "5")),
            ):
                rows.append(
                    {
                        "stage": stage,
                        "k": int(k),
                        "split": split,
                        "scope": scope,
                        "point_rel": metric(
                            report,
                            k=k,
                            split=split,
                            scope=scope,
                            name="point_rel",
                        ),
                        "depth_rel": metric(
                            report,
                            k=k,
                            split=split,
                            scope=scope,
                            name="depth_rel",
                        ),
                        "depth_delta_1.01": metric(
                            report,
                            k=k,
                            split=split,
                            scope=scope,
                            name="depth_delta_1.01",
                        ),
                        "boundary_f1": metric(
                            report,
                            k=k,
                            split=split,
                            scope=scope,
                            name="boundary_f1",
                        ),
                    }
                )

    summary = {
        "status": "complete",
        "baseline_checkpoint_sha256": baseline["checkpoint_sha256"],
        "final_checkpoint_sha256": final["checkpoint_sha256"],
        "final_checkpoint_step": final["checkpoint_step"],
        "comparisons": comparisons,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output / "metrics_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with args.evaluation_history.open(newline="", encoding="utf-8") as handle:
        history = list(csv.DictReader(handle))
    steps = np.asarray([int(row["step"]) for row in history])
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    for axis, prefix, title in (
        (axes[0], "", "Full image"),
        (axes[1], "structure_", "Locked thin-structure ROIs"),
    ):
        for split, color in (("train", "#2563eb"), ("val", "#dc2626")):
            for k, style in ((0, "-"), (3, "--")):
                values = [
                    100 * float(row[f"{split}/k{k}_{prefix}point_rel"])
                    for row in history
                ]
                axis.plot(
                    steps,
                    values,
                    style,
                    color=color,
                    marker="o",
                    linewidth=2,
                    label=f"{split} K={k}",
                )
        axis.set_title(title)
        axis.set_xlabel("Optimization step")
        axis.set_ylabel("Point Rel (%)")
        axis.grid(alpha=0.25)
        axis.legend(frameon=False, fontsize=9)
    for suffix in ("png", "pdf"):
        figure.savefig(args.output / f"training_metrics.{suffix}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    x = np.arange(3, dtype=float)
    width = 0.22
    for axis, scope, title in (
        (axes[0], "full", "Full image"),
        (axes[1], "structure", "Locked thin-structure ROIs"),
    ):
        for offset, (k, color) in enumerate(
            zip(("0", "3", "5"), ("#94a3b8", "#2563eb", "#f59e0b"))
        ):
            values = [
                100
                * metric(
                    final,
                    k=k,
                    split=split,
                    scope=scope,
                    name="point_rel",
                )
                for split in SPLITS
            ]
            axis.bar(
                x + (offset - 1) * width,
                values,
                width,
                color=color,
                label=f"K={k}",
            )
        axis.set_title(title)
        axis.set_xticks(x, SPLITS)
        axis.set_ylabel("Point Rel (%)")
        axis.grid(axis="y", alpha=0.25)
        axis.legend(frameon=False)
    for suffix in ("png", "pdf"):
        figure.savefig(args.output / f"k_sweep.{suffix}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
