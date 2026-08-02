from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from experiment.stage3_stability_normalization.runs.exp19_ssr_batchnorm_recalibration.summarize_results import (
    compare_arm,
    metric_summary,
    residual_summary,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总Exp21逐轮统计结果。")
    parser.add_argument("--baseline-residual", type=Path, required=True)
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--residual", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--exp19-summary", type=Path, required=True)
    parser.add_argument("--exp20-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(
    baseline_residual: dict[str, Any],
    baseline_metrics: dict[str, Any],
    calibration: dict[str, Any],
    residual: dict[str, Any],
    metrics: dict[str, Any],
    exp19: dict[str, Any],
    exp20: dict[str, Any],
) -> dict[str, Any]:
    arm = compare_arm(
        baseline_residual,
        baseline_metrics,
        residual,
        metrics,
    )
    statistics_hash = calibration["iteration_statistics_sha256"]
    return {
        "status": "complete",
        "normalization_policy": "iteration_specific_running_statistics",
        "checkpoint_unchanged": (
            baseline_residual["checkpoint_sha256"]
            == residual["checkpoint_sha256"]
            == metrics["checkpoint_sha256"]
            == calibration["source_checkpoint_sha256"]
        ),
        "statistics_hash_consistent": (
            statistics_hash
            == residual["ssr_iteration_statistics_sha256"]
            == metrics["ssr_iteration_statistics_sha256"]
        ),
        "calibration": calibration,
        "baseline": {
            "residual": residual_summary(baseline_residual),
            "metrics": metric_summary(baseline_metrics),
        },
        "iteration_specific": arm,
        "prior_controls": {
            "mixed_batch8": exp19["arms"]["b8"],
            "per_image": exp20["batch_statistics"],
        },
        "decision": {
            "all_residual_outliers_removed": (
                arm["residual"]["outlier_count"] == 0
            ),
            "maximum_residual_below_threshold": (
                arm["residual"]["maximum"]
                <= float(residual["summary"]["threshold"])
            ),
            "k3_point_rel_improves_all_splits_and_scopes": all(
                value > 0
                for split in arm["k3_changes_from_original"].values()
                for value in split.values()
            ),
        },
    }


def main() -> None:
    args = parse_args()
    report = summarize(
        load_json(args.baseline_residual),
        load_json(args.baseline_metrics),
        load_json(args.calibration),
        load_json(args.residual),
        load_json(args.metrics),
        load_json(args.exp19_summary),
        load_json(args.exp20_summary),
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
