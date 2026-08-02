from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def core_metrics(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint_step": report["checkpoint_step"],
        "metrics_by_k": report["metrics_by_k"],
        "changes_from_k0": report["changes_from_k0"],
        "fraction_improved": report["fraction_improved"],
    }


def residual_core(report: dict[str, Any]) -> dict[str, Any]:
    summary = report["summary"]
    return {
        "threshold": summary["threshold"],
        "unique_sample_count_exceeding_threshold": summary[
            "unique_sample_count_exceeding_threshold"
        ],
        "unique_samples_exceeding_threshold": summary[
            "unique_samples_exceeding_threshold"
        ],
        "global_maximum": summary["global_maximum"],
        "aggregates": summary["aggregates"],
    }


def summarize(
    *,
    root: Path,
    mode: str,
    baseline_metrics: Path | None,
    baseline_residual: Path | None,
) -> dict[str, Any]:
    if mode == "smoke":
        report = load_json(root / "resume_audit.json")
        payload = {"status": "complete", "mode": mode, "resume_audit": report}
    else:
        training = load_json(root / "training" / "report.json")
        evaluation = load_json(root / "evaluation" / "report.json")
        residual = load_json(root / "residual" / "report.json")
        payload = {
            "status": training["status"],
            "mode": mode,
            "training": {
                "start_step": 200,
                "end_step": training["optimization_steps"],
                "best_step": training["best_step"],
                "best_k3_point_rel": training["best_k3_point_rel"],
                "peak_memory_bytes": training["peak_memory_bytes"],
                "total_seconds": training["total_seconds"],
                "latest_periodic": training["latest_periodic"],
            },
            "microbatch2": {
                "metrics": core_metrics(evaluation),
                "residual": residual_core(residual),
            },
            "microbatch1": {
                "metrics": core_metrics(load_json(baseline_metrics)),
                "residual": residual_core(load_json(baseline_residual)),
            },
        }
    destination = root / "report.json"
    temporary = destination.with_suffix(".json.incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "formal"), required=True)
    parser.add_argument("--baseline-metrics", type=Path)
    parser.add_argument("--baseline-residual", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            summarize(
                root=args.root,
                mode=args.mode,
                baseline_metrics=args.baseline_metrics,
                baseline_residual=args.baseline_residual,
            ),
            ensure_ascii=False,
        )
    )
