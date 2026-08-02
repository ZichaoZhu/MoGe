from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def csv_step(path: Path, step: int) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if int(row["step"]) == step:
                return {
                    key: float(value)
                    for key, value in row.items()
                    if key != "step"
                }
    raise ValueError(f"Step {step} not found in {path}")


def summarize(
    *,
    root: Path,
    mode: str,
    baseline_history: Path | None,
) -> dict[str, Any]:
    training = load_json(root / "training" / "report.json")
    payload: dict[str, Any] = {
        "status": training["status"],
        "mode": mode,
        "training": {
            "steps": training["optimization_steps"],
            "best_step": training["best_step"],
            "best_k3_point_rel": training["best_k3_point_rel"],
            "peak_memory_bytes": training["peak_memory_bytes"],
            "total_seconds": training["total_seconds"],
            "initial_periodic": training["initial_periodic"],
            "latest_periodic": training["latest_periodic"],
        },
    }
    if baseline_history is not None:
        payload["exp15_microbatch1_step200"] = csv_step(
            baseline_history,
            200,
        )
    evaluation_path = root / "evaluation" / "report.json"
    residual_path = root / "residual" / "report.json"
    if evaluation_path.is_file():
        evaluation = load_json(evaluation_path)
        payload["evaluation"] = {
            "metrics_by_k": evaluation["metrics_by_k"],
            "changes_from_k0": evaluation["changes_from_k0"],
            "fraction_improved": evaluation["fraction_improved"],
        }
    if residual_path.is_file():
        payload["residual"] = load_json(residual_path)["summary"]
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
    parser.add_argument("--baseline-history", type=Path)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(
        json.dumps(
            summarize(
                root=args.root,
                mode=args.mode,
                baseline_history=args.baseline_history,
            ),
            ensure_ascii=False,
        )
    )
