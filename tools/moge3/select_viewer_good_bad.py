from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select the three strongest K=0→K=3 improvements and the two "
            "strongest degradations from each data split."
        )
    )
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    return parser.parse_args()


def relative_improvement(row: Mapping[str, str]) -> float:
    base = float(row["k0_point_rel"])
    refined = float(row["k3_point_rel"])
    return (base - refined) / max(base, 1e-12)


def select_good_bad(
    rows: Iterable[Mapping[str, str]],
    *,
    experiment: str,
    splits: Sequence[str] = SPLITS,
) -> Dict[str, Any]:
    rows = list(rows)
    selected: Dict[str, list[Dict[str, Any]]] = {}
    policies: Dict[str, str] = {}
    for split in splits:
        split_rows = [row for row in rows if str(row["split"]) == split]
        ranked = sorted(
            (
                {
                    "id": str(row["id"]),
                    "relativeImprovement": relative_improvement(row),
                }
                for row in split_rows
            ),
            key=lambda row: (-float(row["relativeImprovement"]), str(row["id"])),
        )
        if len(ranked) < 5:
            raise ValueError(f"{split} contains fewer than five samples")
        positive = [row for row in ranked if row["relativeImprovement"] > 0]
        negative = [row for row in ranked if row["relativeImprovement"] < 0]
        if len(positive) < 3 or len(negative) < 2:
            raise ValueError(
                f"{split} needs at least three improvements and two degradations"
            )
        ranks = {str(row["id"]): index + 1 for index, row in enumerate(ranked)}
        chosen = positive[:3] + list(reversed(negative[-2:]))
        selected[split] = [
            {
                **row,
                "picture": picture,
                "rank": ranks[str(row["id"])],
                "total": len(ranked),
                "outcome": (
                    "improved"
                    if float(row["relativeImprovement"]) > 0
                    else "degraded"
                ),
            }
            for picture, row in enumerate(chosen, start=1)
        ]
        policies[split] = (
            "K=0到K=3 Point Rel改善最大的三张，加退化最严重的两张"
        )
    return {
        "version": 1,
        "experiment": experiment,
        "metric": "full_point_rel_relative_reduction_k0_to_k3",
        "policy": policies,
        "splits": selected,
    }


def main() -> None:
    args = parse_args()
    metrics = assert_safe_path(
        args.metrics,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(
        args.output,
        safe_root=args.safe_root,
        writable=True,
    )
    with metrics.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    payload = select_good_bad(rows, experiment=args.experiment)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
