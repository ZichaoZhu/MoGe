from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return sum(finite) / len(finite) if finite else math.nan


def mean_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    group_keys: Sequence[str],
    metrics: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, ...], list[Mapping[str, str]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(row[key]) for key in group_keys)].append(row)
    output = []
    for group, selected in sorted(groups.items()):
        record: dict[str, Any] = dict(zip(group_keys, group, strict=True))
        record["count"] = len(selected)
        for metric in metrics:
            record[metric] = finite_mean(
                [float(row[metric]) for row in selected]
            )
        output.append(record)
    return output


def group_energy_shares(
    objective_rows: Sequence[Mapping[str, str]],
    group_rows: Sequence[Mapping[str, str]],
) -> list[dict[str, Any]]:
    totals = {
        (row["batch_id"], row["objective"]): float(row["gradient_l2"])
        for row in objective_rows
    }
    shares = []
    for row in group_rows:
        total = totals[(row["batch_id"], row["objective"])]
        group_norm = float(row["gradient_l2"])
        shares.append(
            {
                "split": row["split"],
                "objective": row["objective"],
                "parameter_group": row["parameter_group"],
                "energy_share": (
                    group_norm * group_norm / (total * total)
                    if total > 0
                    else math.nan
                ),
            }
        )
    return mean_rows(
        shares,
        group_keys=("split", "objective", "parameter_group"),
        metrics=("energy_share",),
    )


def ratio_by_batch(
    objective_rows: Sequence[Mapping[str, str]],
    numerator: str,
    denominator: str,
    metric: str,
) -> dict[str, float]:
    indexed = {
        (row["batch_id"], row["objective"]): row
        for row in objective_rows
    }
    split_values: dict[str, list[float]] = defaultdict(list)
    for (batch_id, objective), row in indexed.items():
        if objective != numerator:
            continue
        left = float(row[metric])
        right = float(indexed[(batch_id, denominator)][metric])
        split_values[row["split"]].append(left / right)
    return {
        split: finite_mean(values)
        for split, values in sorted(split_values.items())
    }


def summarize(
    objective_rows: Sequence[Mapping[str, str]],
    cosine_rows: Sequence[Mapping[str, str]],
    group_rows: Sequence[Mapping[str, str]],
    report: Mapping[str, Any],
    effective_objective_rows: Sequence[Mapping[str, str]] | None = None,
    effective_cosine_rows: Sequence[Mapping[str, str]] | None = None,
    effective_group_rows: Sequence[Mapping[str, str]] | None = None,
    cross_split_rows: Sequence[Mapping[str, str]] | None = None,
    effective_scope_rows: Sequence[Mapping[str, str]] | None = None,
) -> dict[str, Any]:
    objective_means = mean_rows(
        objective_rows,
        group_keys=("split", "objective"),
        metrics=(
            "loss",
            "gradient_l2",
            "gradient_rms",
            "loss_normalized_gradient_rms",
            "cosine_to_paper_combined",
            "projection_fraction_to_paper_combined",
            "cosine_to_checkpoint_combined",
            "projection_fraction_to_checkpoint_combined",
            "max_abs_log_depth_residual",
        ),
    )
    cosine_means = mean_rows(
        cosine_rows,
        group_keys=("split", "gradient_pair"),
        metrics=("cosine",),
    )
    energy_shares = group_energy_shares(objective_rows, group_rows)
    paper_checkpoint_cosine = {
        row["split"]: row["cosine_to_checkpoint_combined"]
        for row in objective_means
        if row["objective"] == "combined_paper"
    }
    edge_output_energy = {
        row["split"]: row["energy_share"]
        for row in energy_shares
        if row["objective"] == "edge_paper"
        and row["parameter_group"] == "output"
    }
    payload = {
        "status": "complete",
        "checkpoint_step": report["checkpoint_step"],
        "checkpoint_sha256": report["checkpoint_sha256"],
        "shape": report["shape"],
        "microbatch_size": report["microbatch_size"],
        "pairs_per_split": report["pairs_per_split"],
        "parameter_count": report["parameter_count"],
        "selection": report["selection"],
        "elapsed_seconds": report["elapsed_seconds"],
        "peak_cuda_memory_bytes": report["peak_cuda_memory_bytes"],
        "objective_means": objective_means,
        "cosine_means": cosine_means,
        "group_energy_shares": energy_shares,
        "edge_to_global_gradient_l2_ratio": ratio_by_batch(
            objective_rows,
            "edge_paper",
            "global",
            "gradient_l2",
        ),
        "edge_to_local_gradient_l2_ratio": ratio_by_batch(
            objective_rows,
            "edge_paper",
            "local",
            "gradient_l2",
        ),
        "edge_output_layer_gradient_energy_share": edge_output_energy,
        "paper_to_checkpoint_combined_gradient_cosine": (
            paper_checkpoint_cosine
        ),
    }
    if (
        effective_objective_rows is not None
        and effective_cosine_rows is not None
        and effective_group_rows is not None
    ):
        payload["effective_batch_size"] = report["effective_batch_size"]
        payload["effective_objectives"] = mean_rows(
            effective_objective_rows,
            group_keys=("split", "objective"),
            metrics=(
                "loss",
                "gradient_l2",
                "gradient_rms",
                "loss_normalized_gradient_rms",
                "cosine_to_paper_combined",
                "projection_fraction_to_paper_combined",
                "cosine_to_checkpoint_combined",
                "projection_fraction_to_checkpoint_combined",
                "max_abs_log_depth_residual",
            ),
        )
        payload["effective_cosines"] = mean_rows(
            effective_cosine_rows,
            group_keys=("split", "gradient_pair"),
            metrics=("cosine",),
        )
        payload["effective_group_energy_shares"] = group_energy_shares(
            effective_objective_rows,
            effective_group_rows,
        )
        payload["effective_edge_to_global_gradient_l2_ratio"] = (
            ratio_by_batch(
                effective_objective_rows,
                "edge_paper",
                "global",
                "gradient_l2",
            )
        )
        payload["effective_edge_to_local_gradient_l2_ratio"] = ratio_by_batch(
            effective_objective_rows,
            "edge_paper",
            "local",
            "gradient_l2",
        )
    if cross_split_rows is not None:
        payload["cross_split_cosines"] = [
            {
                "split_pair": row["split_pair"],
                "left_split": row["left_split"],
                "right_split": row["right_split"],
                "objective": row["objective"],
                "parameter_scope": row["parameter_scope"],
                "cosine": float(row["cosine"]),
            }
            for row in cross_split_rows
        ]
    if effective_scope_rows is not None:
        payload["effective_scope_cosines"] = [
            {
                "split": row["split"],
                "gradient_pair": row["gradient_pair"],
                "parameter_scope": row["parameter_scope"],
                "cosine": float(row["cosine"]),
            }
            for row in effective_scope_rows
        ]
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = json.loads(
        (args.formal / "report.json").read_text(encoding="utf-8")
    )
    payload = summarize(
        read_csv(args.formal / "per_batch_objectives.csv"),
        read_csv(args.formal / "per_batch_cosines.csv"),
        read_csv(args.formal / "per_group_gradients.csv"),
        report,
        read_csv(args.formal / "effective_batch_objectives.csv"),
        read_csv(args.formal / "effective_batch_cosines.csv"),
        read_csv(args.formal / "effective_group_gradients.csv"),
        (
            read_csv(args.formal / "cross_split_cosines.csv")
            if (args.formal / "cross_split_cosines.csv").is_file()
            else None
        ),
        (
            read_csv(args.formal / "effective_scope_cosines.csv")
            if (args.formal / "effective_scope_cosines.csv").is_file()
            else None
        ),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
