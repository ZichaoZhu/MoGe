from __future__ import annotations

import csv
import json
import os
import pathlib
from typing import Any


SAFE_ROOT = pathlib.Path("/mnt/data/home/zhuzichao")
ARMS = ("lr_2e6", "lr_1e5", "lr_2e5")


def safe_path(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve(strict=False)
    if resolved == SAFE_ROOT or SAFE_ROOT not in resolved.parents:
        raise ValueError(f"Path escapes safe root: {resolved}")
    return resolved


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def relative_reduction(base: float, refined: float) -> float:
    return (base - refined) / max(base, 1e-12)


def training_summary(
    arm: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    if not rows:
        return {"arm": arm, "status": "missing"}
    initial = rows[0]
    best = min(
        rows,
        key=lambda row: float(row["train/k3_structure_point_rel"]),
    )
    best_validation = min(
        rows,
        key=lambda row: float(row["val/k3_structure_point_rel"]),
    )
    last = rows[-1]

    def snapshot(row: dict[str, str]) -> dict[str, Any]:
        values: dict[str, Any] = {"step": int(row["step"])}
        for split in ("train", "val"):
            for scope in ("full", "structure"):
                key = (
                    f"{split}/k3_point_rel"
                    if scope == "full"
                    else f"{split}/k3_structure_point_rel"
                )
                base_key = (
                    f"{split}/k0_point_rel"
                    if scope == "full"
                    else f"{split}/k0_structure_point_rel"
                )
                if key in row and base_key in row:
                    base = float(row[base_key])
                    refined = float(row[key])
                    values[f"{split}_{scope}_k0_point_rel"] = base
                    values[f"{split}_{scope}_k3_point_rel"] = refined
                    values[f"{split}_{scope}_relative_reduction"] = (
                        relative_reduction(base, refined)
                    )
        return values

    return {
        "arm": arm,
        "status": "complete" if int(last["step"]) >= 800 else "running",
        "initial": snapshot(initial),
        "best_train_structure": snapshot(best),
        "best_validation_structure": snapshot(best_validation),
        "last": snapshot(last),
    }


def posthoc_summary(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text(encoding="utf-8"))
    return {
        "checkpoint_step": report["checkpoint_step"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "metrics_by_k": report["metrics_by_k"],
        "changes_from_k0": report["changes_from_k0"],
        "fraction_improved": report["fraction_improved"],
        "roi_counts": report["roi_counts"],
    }


def main() -> None:
    experiment = safe_path(pathlib.Path(__file__).parent)
    results = safe_path(experiment / "results")
    results.mkdir(parents=True, exist_ok=True)
    summaries = []
    for arm in ARMS:
        training = training_summary(
            arm,
            read_csv(experiment / "artifacts" / arm / "evaluation_history.csv"),
        )
        training["posthoc"] = posthoc_summary(
            experiment / "artifacts" / f"{arm}_posthoc" / "report.json"
        )
        summaries.append(training)

    completed = [
        summary
        for summary in summaries
        if summary["status"] == "complete" and summary["posthoc"] is not None
    ]
    training_fit_recommendation = None
    stable_generalization_recommendation = None
    if completed:
        training_fit_recommendation = max(
            completed,
            key=lambda summary: (
                summary["posthoc"]["changes_from_k0"]["3"]["train"].get(
                    "structure_point_rel_relative_reduction",
                    float("-inf"),
                ),
                summary["posthoc"]["changes_from_k0"]["3"]["val"].get(
                    "structure_point_rel_relative_reduction",
                    float("-inf"),
                ),
            ),
        )["arm"]
        stable_generalization_recommendation = max(
            completed,
            key=lambda summary: (
                summary["posthoc"]["changes_from_k0"]["5"]["test"].get(
                    "structure_point_rel_relative_reduction",
                    float("-inf"),
                ),
                summary["posthoc"]["changes_from_k0"]["3"]["test"].get(
                    "structure_point_rel_relative_reduction",
                    float("-inf"),
                ),
            ),
        )["arm"]
    payload = {
        "status": "complete" if len(completed) == len(ARMS) else "partial",
        "experiment": "exp13_fixed_base_ssr_diagnostic",
        "arms": summaries,
        "recommendations": {
            "training_fit": training_fit_recommendation,
            "k5_test_structure_stability": (
                stable_generalization_recommendation
            ),
            "single_unqualified_winner": None,
            "reason": (
                "The learning-rate sweep exposes a training-fit versus "
                "held-out stability trade-off; report both choices instead "
                "of collapsing them into one score."
            ),
        },
    }
    temporary = results / "summary.json.incomplete"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, results / "summary.json")

    rows = []
    for summary in summaries:
        best = summary.get("best_train_structure", {})
        best_validation = summary.get("best_validation_structure", {})
        rows.append(
            {
                "arm": summary["arm"],
                "status": summary["status"],
                "best_step": best.get("step"),
                "train_full_k3_point_rel": best.get(
                    "train_full_k3_point_rel"
                ),
                "train_structure_k3_point_rel": best.get(
                    "train_structure_k3_point_rel"
                ),
                "train_structure_relative_reduction": best.get(
                    "train_structure_relative_reduction"
                ),
                "val_full_k3_point_rel": best.get("val_full_k3_point_rel"),
                "val_structure_k3_point_rel": best.get(
                    "val_structure_k3_point_rel"
                ),
                "val_structure_relative_reduction": best.get(
                    "val_structure_relative_reduction"
                ),
                "best_validation_step": best_validation.get("step"),
                "best_validation_structure_k3_point_rel": best_validation.get(
                    "val_structure_k3_point_rel"
                ),
                "best_validation_structure_relative_reduction": (
                    best_validation.get("val_structure_relative_reduction")
                ),
                "best_validation_full_k3_point_rel": best_validation.get(
                    "val_full_k3_point_rel"
                ),
                "posthoc_complete": summary["posthoc"] is not None,
            }
        )
    with (results / "metrics_summary.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(payload, ensure_ascii=False))


if __name__ == "__main__":
    main()
