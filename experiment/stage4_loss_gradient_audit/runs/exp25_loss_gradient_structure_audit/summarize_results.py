from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any, Sequence


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def selected(
    rows: Sequence[dict[str, str]],
    *,
    split: str,
    step: int,
    objective: str,
) -> list[dict[str, str]]:
    result = [
        row
        for row in rows
        if row["split"] == split
        and int(row["k"]) == step
        and row["objective"] == objective
    ]
    if not result:
        raise ValueError(
            f"No rows for split={split}, K={step}, objective={objective}"
        )
    return result


def average(rows: Sequence[dict[str, str]], key: str) -> float:
    return mean(float(row[key]) for row in rows)


def relative_reduction(before: float, after: float) -> float:
    return (before - after) / max(before, 1e-30)


def summarize(
    rows: Sequence[dict[str, str]],
    report: dict[str, Any],
) -> dict[str, Any]:
    split_objective = {}
    for split in ("train", "val", "test"):
        before_rows = selected(
            rows,
            split=split,
            step=0,
            objective="combined_paper",
        )
        after_rows = selected(
            rows,
            split=split,
            step=3,
            objective="combined_paper",
        )
        before = average(before_rows, "loss")
        after = average(after_rows, "loss")
        split_objective[split] = {
            "sample_count": len(before_rows),
            "k0_combined_loss": before,
            "k3_combined_loss": after,
            "k3_relative_reduction": relative_reduction(before, after),
        }

    train_k3 = {}
    for objective in ("global", "local", "edge_paper_min", "combined_paper"):
        objective_rows = selected(
            rows,
            split="train",
            step=3,
            objective=objective,
        )
        train_k3[objective] = {
            key: average(objective_rows, key)
            for key in (
                "loss",
                "gradient_l1_mass",
                "structure_gradient_enrichment",
                "gt_boundary_gradient_enrichment",
            )
        }
    combined_rows = selected(
        rows,
        split="train",
        step=3,
        objective="combined_paper",
    )
    return {
        "status": "complete",
        "checkpoint_step": report["checkpoint_step"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "shape": report["shape"],
        "selection": report["selection"],
        "edge_formula": report["edge_formula"],
        "peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
        "objective_change_k0_to_k3": split_objective,
        "train_k3_objectives": train_k3,
        "train_k3_gradient_cosines": {
            key: average(combined_rows, key)
            for key in (
                "cosine_global_local",
                "cosine_global_edge",
                "cosine_local_edge",
            )
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--per-sample", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = summarize(
        load_csv(args.per_sample),
        json.loads(args.report.read_text(encoding="utf-8")),
    )
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
