from __future__ import annotations

import argparse
import csv
import json
import pathlib
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


SCOPES = {
    "full": "",
    "crop": "crop_",
    "structure": "structure_",
}
SPLITS = ("train", "val", "test")
KS = ("0", "1", "3", "5")


def read_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(
    report: dict[str, Any],
    *,
    k: str,
    split: str,
    scope: str,
    name: str,
) -> float | None:
    key = f"{SCOPES[scope]}{name}"
    value = report["metrics_by_k"][k][split].get(key)
    return float(value) if value is not None else None


def relative_reduction(before: float | None, after: float | None) -> float | None:
    if before is None or after is None or before == 0:
        return None
    return (before - after) / before


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 Exp14 与 Exp13 warm start 的变化。")
    parser.add_argument("--warm-report", type=pathlib.Path, required=True)
    parser.add_argument("--final-report", type=pathlib.Path, required=True)
    parser.add_argument("--evaluation-history", type=pathlib.Path)
    parser.add_argument("--output", type=pathlib.Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    warm = read_json(args.warm_report)
    final = read_json(args.final_report)
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for stage, report in (("warm_start", warm), ("final", final)):
        for k in KS:
            for split in SPLITS:
                for scope in SCOPES:
                    point_rel = metric(
                        report,
                        k=k,
                        split=split,
                        scope=scope,
                        name="point_rel",
                    )
                    if point_rel is None:
                        continue
                    rows.append(
                        {
                            "stage": stage,
                            "k": int(k),
                            "split": split,
                            "scope": scope,
                            "point_rel": point_rel,
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
                            "depth_delta_1.25": metric(
                                report,
                                k=k,
                                split=split,
                                scope=scope,
                                name="depth_delta_1.25",
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

    with (args.output / "metrics_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    comparisons: dict[str, Any] = {}
    for split in SPLITS:
        comparisons[split] = {}
        for scope in SCOPES:
            warm_k0 = metric(
                warm, k="0", split=split, scope=scope, name="point_rel"
            )
            warm_k3 = metric(
                warm, k="3", split=split, scope=scope, name="point_rel"
            )
            final_k0 = metric(
                final, k="0", split=split, scope=scope, name="point_rel"
            )
            final_k3 = metric(
                final, k="3", split=split, scope=scope, name="point_rel"
            )
            final_k5 = metric(
                final, k="5", split=split, scope=scope, name="point_rel"
            )
            comparisons[split][scope] = {
                "warm_k0_point_rel": warm_k0,
                "warm_k3_point_rel": warm_k3,
                "final_k0_point_rel": final_k0,
                "final_k3_point_rel": final_k3,
                "final_k5_point_rel": final_k5,
                "base_head_relative_reduction": relative_reduction(
                    warm_k0,
                    final_k0,
                ),
                "final_ssr_k3_relative_reduction": relative_reduction(
                    final_k0,
                    final_k3,
                ),
                "joint_k3_relative_reduction": relative_reduction(
                    warm_k3,
                    final_k3,
                ),
                "total_relative_reduction": relative_reduction(
                    warm_k0,
                    final_k3,
                ),
                "k5_vs_k3_relative_reduction": relative_reduction(
                    final_k3,
                    final_k5,
                ),
                "boundary_f1_change_at_k3": (
                    metric(
                        final,
                        k="3",
                        split=split,
                        scope=scope,
                        name="boundary_f1",
                    )
                    - metric(
                        warm,
                        k="3",
                        split=split,
                        scope=scope,
                        name="boundary_f1",
                    )
                ),
            }

    summary = {
        "status": "complete",
        "warm_checkpoint_sha256": warm["checkpoint_sha256"],
        "warm_checkpoint_step": warm["checkpoint_step"],
        "final_checkpoint_sha256": final["checkpoint_sha256"],
        "final_checkpoint_step": final["checkpoint_step"],
        "comparisons": comparisons,
    }
    (args.output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    if args.evaluation_history is not None:
        with args.evaluation_history.open(newline="", encoding="utf-8") as handle:
            history = list(csv.DictReader(handle))
        steps = np.asarray([int(row["step"]) for row in history])
        figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
        for axis, scope, title in (
            (axes[0], "", "Full image"),
            (axes[1], "structure_", "Locked thin-structure ROIs"),
        ):
            for split, color in (("train", "#2563eb"), ("val", "#dc2626")):
                for k, style in ((0, "-"), (3, "--")):
                    values = 100.0 * np.asarray(
                        [
                            float(row[f"{split}/k{k}_{scope}point_rel"])
                            for row in history
                        ]
                    )
                    axis.plot(
                        steps,
                        values,
                        style,
                        color=color,
                        marker="o",
                        linewidth=2,
                        label=f"{split} K={k}",
                    )
            axis.axvline(700, color="#64748b", linestyle=":", linewidth=1.5)
            axis.set_title(title)
            axis.set_xlabel("Optimization step")
            axis.set_ylabel("Point Rel (%)")
            axis.grid(alpha=0.25)
            axis.legend(frameon=False, fontsize=9)
        for suffix in ("png", "pdf"):
            figure.savefig(args.output / f"training_metrics.{suffix}", dpi=180)
        plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.0, 4.4), constrained_layout=True)
    x = np.arange(len(SPLITS), dtype=float)
    width = 0.22
    colors = ("#94a3b8", "#2563eb", "#f59e0b")
    for axis, scope, title in (
        (axes[0], "full", "Full image"),
        (axes[1], "structure", "Locked thin-structure ROIs"),
    ):
        for offset, (k, color) in enumerate(zip(("0", "3", "5"), colors)):
            values = [
                100.0
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
                width=width,
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
