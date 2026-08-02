from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


VARIANTS = ("layer_norm", "group_norm")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def point_rel_table(report: dict[str, Any], step: int = 3) -> dict[str, Any]:
    metrics = report["metrics_by_k"][str(step)]
    return {
        split: {
            "full": values["point_rel"],
            "structure": values.get("structure_point_rel"),
        }
        for split, values in metrics.items()
    }


def summarize(root: Path, mode: str) -> dict[str, Any]:
    variants: dict[str, Any] = {}
    for name in VARIANTS:
        variant_root = root / name
        training = load_json(variant_root / "training" / "report.json")
        result: dict[str, Any] = {
            "normalization": name,
            "training": {
                "status": training["status"],
                "steps": training["optimization_steps"],
                "best_step": training["best_step"],
                "peak_memory_bytes": training["peak_memory_bytes"],
                "initial_periodic": training["initial_periodic"],
                "latest_periodic": training["latest_periodic"],
            },
        }
        evaluation_path = variant_root / "evaluation" / "report.json"
        residual_path = variant_root / "residual" / "report.json"
        if evaluation_path.is_file():
            evaluation = load_json(evaluation_path)
            result["evaluation"] = {
                "checkpoint_step": evaluation["checkpoint_step"],
                "k3_point_rel": point_rel_table(evaluation, 3),
                "k5_point_rel": point_rel_table(evaluation, 5),
                "k3_changes_from_k0": evaluation["changes_from_k0"]["3"],
                "k3_fraction_improved": evaluation["fraction_improved"]["3"],
            }
        if residual_path.is_file():
            residual = load_json(residual_path)
            result["residual"] = residual["summary"]
        variants[name] = result

    complete = all(
        value["training"]["status"] == "complete"
        for value in variants.values()
    )
    payload = {
        "status": "complete" if complete else "failed",
        "mode": mode,
        "variants": variants,
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
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(summarize(args.root, args.mode), ensure_ascii=False))
