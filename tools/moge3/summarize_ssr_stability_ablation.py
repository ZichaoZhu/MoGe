from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize the MoGe-3 SSR K/edge-weight stability ablation"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    return parser.parse_args()


def read_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_csv(path: Path, records: Iterable[Dict[str, object]]) -> None:
    rows = list(records)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def relative_reduction(after: float, before: float) -> float:
    return 1.0 - after / before


def build_row(
    arm: Dict[str, object],
    best_report: Dict[str, object],
    latest_report: Dict[str, object],
) -> Dict[str, object]:
    arm_id = str(arm["id"])
    training_k = int(arm["training_refinement_steps"])
    edge_weight = float(arm["edge_weight"])
    reported_k = int(
        best_report["paper_alignment"]["training_refinement_steps"]
    )
    reported_edge = float(best_report["loss_weights"]["edge"])
    if reported_k != training_k or reported_edge != edge_weight:
        raise ValueError(f"Run configuration mismatch for {arm_id}")
    if int(latest_report["checkpoint_step"]) != int(arm["steps"]):
        raise ValueError(f"Latest checkpoint step mismatch for {arm_id}")

    row: Dict[str, object] = {
        "arm": arm_id,
        "training_k": training_k,
        "edge_weight": edge_weight,
        "best_step": int(best_report["best_checkpoint_step"]),
        "latest_step": int(latest_report["checkpoint_step"]),
        "training_seconds": float(
            best_report["training_seconds_including_periodic_eval"]
        ),
        "peak_memory_bytes": int(best_report["peak_memory_bytes"]),
    }
    for checkpoint_name, report in (
        ("best", best_report),
        ("latest", latest_report),
    ):
        metrics_by_k = report["metrics_by_k"]
        for split in SPLITS:
            base = metrics_by_k["0"][split]
            refined = metrics_by_k[str(training_k)][split]
            row[f"{checkpoint_name}_{split}_point_rel"] = float(
                refined["point_rel"]
            )
            row[f"{checkpoint_name}_{split}_depth_rel"] = float(
                refined["depth_rel"]
            )
            row[f"{checkpoint_name}_{split}_boundary_f1"] = float(
                refined["boundary_f1"]
            )
            row[f"{checkpoint_name}_{split}_point_rel_reduction"] = (
                relative_reduction(
                    float(refined["point_rel"]),
                    float(base["point_rel"]),
                )
            )
            row[f"{checkpoint_name}_{split}_boundary_f1_change"] = float(
                refined["boundary_f1"]
            ) - float(base["boundary_f1"])
            row[f"{checkpoint_name}_{split}_point_rel_k0"] = float(
                base["point_rel"]
            )
            row[f"{checkpoint_name}_{split}_boundary_f1_k0"] = float(
                base["boundary_f1"]
            )
    return row


def verify_common_baseline(rows: List[Dict[str, object]]) -> None:
    for checkpoint_name in ("best", "latest"):
        for split in SPLITS:
            for metric in ("point_rel_k0", "boundary_f1_k0"):
                values = np.asarray(
                    [
                        float(row[f"{checkpoint_name}_{split}_{metric}"])
                        for row in rows
                    ]
                )
                if not np.allclose(values, values[0], rtol=0, atol=1e-10):
                    raise ValueError(
                        f"Base-model metric differs across arms: "
                        f"{checkpoint_name}/{split}/{metric}"
                    )


def save_plot(output: Path, rows: List[Dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    labels = [
        f"{row['arm']}\nK={row['training_k']}, edge={row['edge_weight']:g}"
        for row in rows
    ]
    x = np.arange(len(rows))
    width = 0.26
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(14, 9))

    for axis, metric, title, direction in (
        (axes[0, 0], "point_rel", "Validation aligned point Rel", "lower"),
        (axes[0, 1], "boundary_f1", "Validation depth-boundary F1", "higher"),
    ):
        baseline = np.asarray(
            [float(row[f"latest_val_{metric}_k0"]) for row in rows]
        )
        best = np.asarray(
            [float(row[f"best_val_{metric}"]) for row in rows]
        )
        latest = np.asarray(
            [float(row[f"latest_val_{metric}"]) for row in rows]
        )
        axis.bar(x - width, baseline, width, label="K=0 base", color="#9CA3AF")
        axis.bar(x, best, width, label="selected best", color="#2563EB")
        axis.bar(x + width, latest, width, label="step 2000", color="#F59E0B")
        axis.set_title(f"{title} ({direction} is better)")
        axis.set_xticks(x, labels)
        axis.legend(fontsize=8)

    for axis, split, title in (
        (axes[1, 0], "train", "Latest checkpoint: training split"),
        (axes[1, 1], "val", "Latest checkpoint: validation split"),
    ):
        point = 100 * np.asarray(
            [float(row[f"latest_{split}_point_rel_reduction"]) for row in rows]
        )
        boundary = np.asarray(
            [float(row[f"latest_{split}_boundary_f1_change"]) for row in rows]
        )
        axis.bar(
            x - width / 2,
            point,
            width,
            label="Point Rel reduction (%)",
            color="#10B981",
        )
        boundary_axis = axis.twinx()
        boundary_axis.bar(
            x + width / 2,
            boundary,
            width,
            label="Boundary F1 change",
            color="#DC2626",
            alpha=0.8,
        )
        axis.axhline(0, color="#374151", linewidth=0.8)
        boundary_axis.axhline(0, color="#374151", linewidth=0.8)
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.set_ylabel("Point Rel reduction (%)")
        boundary_axis.set_ylabel("Boundary F1 change")
        handles_1, labels_1 = axis.get_legend_handles_labels()
        handles_2, labels_2 = boundary_axis.get_legend_handles_labels()
        axis.legend(handles_1 + handles_2, labels_1 + labels_2, fontsize=8)

    figure.suptitle("MoGe-3 SSR refinement-step and edge-weight ablation")
    figure.tight_layout()
    figure.savefig(output / "ablation_summary.png", dpi=180)
    figure.savefig(output / "ablation_summary.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    config_path = assert_safe_path(
        args.config,
        safe_root=args.safe_root,
        must_exist=True,
    )
    runs = assert_safe_path(args.runs, safe_root=args.safe_root, must_exist=True)
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    config = read_json(config_path)

    rows: List[Dict[str, object]] = []
    for arm in config["arms"]:
        arm_dir = assert_safe_path(
            runs / str(arm["id"]),
            safe_root=args.safe_root,
            must_exist=True,
        )
        best_report = read_json(arm_dir / "report.json")
        latest_report = read_json(arm_dir / "latest_metrics" / "report.json")
        rows.append(build_row(arm, best_report, latest_report))
    verify_common_baseline(rows)

    maximum_boundary_drop = float(
        config["acceptance"]["maximum_validation_boundary_f1_drop"]
    )
    eligible = [
        row
        for row in rows
        if float(row["best_val_point_rel_reduction"]) > 0
        and float(row["best_val_boundary_f1_change"]) >= -maximum_boundary_drop
    ]
    winner = (
        min(eligible, key=lambda row: float(row["best_val_point_rel"]))["arm"]
        if eligible
        else None
    )
    save_csv(output / "ablation_summary.csv", rows)
    save_plot(output, rows)
    report = {
        "status": "complete",
        "experiment": config["experiment_id"],
        "hypothesis": config["hypothesis"],
        "selection": {
            "primary": "best full-validation point Rel at the arm's training K",
            "constraint": (
                "best full-validation boundary F1 drop must not exceed "
                f"{maximum_boundary_drop}"
            ),
            "winner": winner,
            "eligible_arms": [row["arm"] for row in eligible],
            "test_metrics_not_used_for_selection": True,
        },
        "rows": rows,
        "artifacts": {
            "table": "ablation_summary.csv",
            "plot": "ablation_summary.png",
            "vector_plot": "ablation_summary.pdf",
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
