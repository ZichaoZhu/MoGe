from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize a two-stage joint fine-tuning experiment"
    )
    parser.add_argument(
        "--experiment",
        default="exp6_two_stage_joint_finetuning",
    )
    parser.add_argument("--baseline-report", type=Path, required=True)
    parser.add_argument("--training-report", type=Path, required=True)
    parser.add_argument("--best-report", type=Path, required=True)
    parser.add_argument("--latest-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--maximum-boundary-drop", type=float, default=0.005)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def relative_reduction(after: float, before: float) -> float:
    return 1.0 - after / before


def build_rows(
    baseline: Dict[str, object],
    reports: Dict[str, Dict[str, object]],
) -> List[Dict[str, object]]:
    rows = []
    baseline_metrics = baseline["metrics_by_k"]["0"]
    for checkpoint_name, report in reports.items():
        for split in ("train", "val", "test"):
            initial = baseline_metrics[split]
            current_k0 = report["metrics_by_k"]["0"][split]
            current_k3 = report["metrics_by_k"]["3"][split]
            rows.append(
                {
                    "checkpoint": checkpoint_name,
                    "step": int(report["checkpoint_step"]),
                    "split": split,
                    "initial_point_rel": float(initial["point_rel"]),
                    "k0_point_rel": float(current_k0["point_rel"]),
                    "k3_point_rel": float(current_k3["point_rel"]),
                    "k3_point_reduction_vs_initial": relative_reduction(
                        float(current_k3["point_rel"]),
                        float(initial["point_rel"]),
                    ),
                    "ssr_point_reduction_k3_vs_k0": relative_reduction(
                        float(current_k3["point_rel"]),
                        float(current_k0["point_rel"]),
                    ),
                    "initial_boundary_f1": float(initial["boundary_f1"]),
                    "k0_boundary_f1": float(current_k0["boundary_f1"]),
                    "k3_boundary_f1": float(current_k3["boundary_f1"]),
                    "k3_boundary_change_vs_initial": (
                        float(current_k3["boundary_f1"])
                        - float(initial["boundary_f1"])
                    ),
                    "ssr_boundary_change_k3_vs_k0": (
                        float(current_k3["boundary_f1"])
                        - float(current_k0["boundary_f1"])
                    ),
                }
            )
    return rows


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_plot(
    output: Path,
    rows: List[Dict[str, object]],
    experiment: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = []
    values = []
    for checkpoint in ("best", "latest"):
        for split in ("train", "val", "test"):
            row = next(
                value
                for value in rows
                if value["checkpoint"] == checkpoint and value["split"] == split
            )
            labels.append(f"{checkpoint}\n{split}")
            values.append(row)
    x = np.arange(len(values))
    width = 0.25
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(12, 9))
    axes[0].bar(
        x - width,
        [row["initial_point_rel"] for row in values],
        width,
        label="initial MoGe-2 K=0",
    )
    axes[0].bar(
        x,
        [row["k0_point_rel"] for row in values],
        width,
        label="joint model K=0",
    )
    axes[0].bar(
        x + width,
        [row["k3_point_rel"] for row in values],
        width,
        label="joint model K=3",
    )
    axes[0].set_ylabel("Aligned point Rel (lower is better)")
    axes[0].set_xticks(x, labels)
    axes[0].legend()

    axes[1].bar(
        x - width,
        [row["initial_boundary_f1"] for row in values],
        width,
        label="initial MoGe-2 K=0",
    )
    axes[1].bar(
        x,
        [row["k0_boundary_f1"] for row in values],
        width,
        label="joint model K=0",
    )
    axes[1].bar(
        x + width,
        [row["k3_boundary_f1"] for row in values],
        width,
        label="joint model K=3",
    )
    axes[1].set_ylabel("Depth-boundary F1 (higher is better)")
    axes[1].set_xticks(x, labels)
    axes[1].legend()
    figure.suptitle(experiment.replace("_", " "))
    figure.tight_layout()
    figure.savefig(output / "joint_finetuning_summary.png", dpi=180)
    figure.savefig(output / "joint_finetuning_summary.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    paths = {
        name: assert_safe_path(
            path,
            safe_root=args.safe_root,
            must_exist=True,
        )
        for name, path in {
            "baseline": args.baseline_report,
            "training": args.training_report,
            "best": args.best_report,
            "latest": args.latest_report,
        }.items()
    }
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    baseline = read_json(paths["baseline"])
    training = read_json(paths["training"])
    reports = {
        "best": read_json(paths["best"]),
        "latest": read_json(paths["latest"]),
    }
    rows = build_rows(baseline, reports)
    best_val = next(
        row
        for row in rows
        if row["checkpoint"] == "best" and row["split"] == "val"
    )
    acceptance = {
        "validation_k3_point_better_than_initial": (
            best_val["k3_point_reduction_vs_initial"] > 0
        ),
        "validation_boundary_drop_within_limit": (
            best_val["k3_boundary_change_vs_initial"]
            >= -args.maximum_boundary_drop
        ),
        "ssr_improves_validation_k3_over_same_checkpoint_k0": (
            best_val["ssr_point_reduction_k3_vs_k0"] > 0
        ),
    }
    acceptance["passed"] = all(acceptance.values())
    save_csv(output / "joint_finetuning_summary.csv", rows)
    save_plot(output, rows, args.experiment)
    report = {
        "status": "complete",
        "experiment": args.experiment,
        "training_best_step": training["best_step"],
        "acceptance": acceptance,
        "maximum_boundary_f1_drop": args.maximum_boundary_drop,
        "test_not_used_for_selection": True,
        "rows": rows,
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
