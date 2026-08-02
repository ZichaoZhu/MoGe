from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总Exp19重校准结果。")
    parser.add_argument("--baseline-residual", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def residual_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "maximum": float(summary["global_maximum"]["max_abs_residual"]),
        "maximum_id": str(summary["global_maximum"]["id"]),
        "maximum_iteration": int(summary["global_maximum"]["iteration"]),
        "outlier_count": int(summary["unique_sample_count_exceeding_threshold"]),
        "outlier_ids": list(summary["unique_samples_exceeding_threshold"]),
    }


def metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for step in ("0", "1", "3", "5"):
        result[step] = {}
        for split in ("train", "val", "test"):
            metrics = report["metrics_by_k"][step][split]
            result[step][split] = {
                key: float(metrics[key])
                for key in (
                    "point_rel",
                    "depth_rel",
                    "boundary_f1",
                    "structure_point_rel",
                    "structure_depth_rel",
                    "structure_boundary_f1",
                )
                if key in metrics
            }
    return result


def relative_change(before: float, after: float) -> float:
    return (before - after) / max(abs(before), 1e-12)


def compare_arm(
    baseline_residual: dict[str, Any],
    baseline_metrics: dict[str, Any],
    arm_residual: dict[str, Any],
    arm_metrics: dict[str, Any],
) -> dict[str, Any]:
    residual_before = residual_summary(baseline_residual)
    residual_after = residual_summary(arm_residual)
    metrics_before = metric_summary(baseline_metrics)
    metrics_after = metric_summary(arm_metrics)
    changes = {}
    for split in ("train", "val", "test"):
        changes[split] = {}
        for scope in ("point_rel", "structure_point_rel"):
            before = metrics_before["3"][split].get(scope)
            after = metrics_after["3"][split].get(scope)
            if before is not None and after is not None:
                changes[split][f"k3_{scope}_relative_reduction"] = relative_change(
                    before,
                    after,
                )
    k0_max_abs_difference = max(
        abs(
            metrics_before["0"][split]["point_rel"]
            - metrics_after["0"][split]["point_rel"]
        )
        for split in ("train", "val", "test")
    )
    return {
        "residual": residual_after,
        "metrics": metrics_after,
        "k3_changes_from_original": changes,
        "k0_point_rel_maximum_absolute_difference": k0_max_abs_difference,
        "outliers_removed": (
            residual_before["outlier_count"] - residual_after["outlier_count"]
        ),
        "maximum_residual_relative_reduction": relative_change(
            residual_before["maximum"],
            residual_after["maximum"],
        ),
    }


def summarize(
    *,
    baseline_residual: dict[str, Any],
    baseline_metrics: dict[str, Any],
    calibration: dict[str, Any],
    arm_reports: dict[str, tuple[dict[str, Any], dict[str, Any]]],
) -> dict[str, Any]:
    baseline = {
        "residual": residual_summary(baseline_residual),
        "metrics": metric_summary(baseline_metrics),
    }
    arms = {
        name: {
            "calibration": calibration["arms"][f"batch_{name[1:]}"],
            **compare_arm(
                baseline_residual,
                baseline_metrics,
                residual,
                metrics,
            ),
        }
        for name, (residual, metrics) in arm_reports.items()
    }
    eligible = [
        name
        for name, arm in arms.items()
        if arm["residual"]["outlier_count"] == 0
    ]
    best = min(
        eligible or list(arms),
        key=lambda name: (
            arms[name]["residual"]["outlier_count"],
            arms[name]["residual"]["maximum"],
            arms[name]["metrics"]["3"]["val"]["point_rel"],
        ),
    )
    return {
        "status": "complete",
        "hypothesis": "SSR BatchNorm运行统计失配导致推理态递归残差离群。",
        "baseline": baseline,
        "arms": arms,
        "selected_arm": best,
        "decision": {
            "any_arm_removes_all_outliers": bool(eligible),
            "batch_size_changes_result": (
                arms["b1"]["residual"] != arms["b8"]["residual"]
            ),
            "learned_parameters_unchanged": all(
                arm["calibration"]["state_audit"][
                    "bitwise_unchanged_state_count"
                ]
                > 0
                for arm in arms.values()
            ),
        },
    }


def main() -> None:
    args = parse_args()
    report = summarize(
        baseline_residual=load_json(args.baseline_residual),
        baseline_metrics=load_json(args.baseline_metrics),
        calibration=load_json(args.calibration),
        arm_reports={
            f"b{batch_size}": (
                load_json(args.artifacts / f"residual_b{batch_size}" / "report.json"),
                load_json(args.artifacts / f"metrics_b{batch_size}" / "report.json"),
            )
            for batch_size in (1, 8)
        },
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, args.output)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
