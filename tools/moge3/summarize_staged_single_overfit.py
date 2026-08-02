from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Dict, Iterable, Mapping


SNAPSHOTS = ("initial", "stage1", "final")
REFINEMENT_STEPS = (0, 1, 3, 5)
SCOPES = ("full", "crop", "structure")
METRICS = (
    "point_rel",
    "depth_rel",
    "depth_delta_1.01",
    "depth_delta_1.25",
    "boundary_f1",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize all completed exp9 single-image runs"
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument(
        "--hash-checkpoints",
        action="store_true",
        help="Compute SHA-256 for server-only checkpoints",
    )
    return parser.parse_args()


def sample_directories(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.iterdir()
        if path.is_dir() and (path / "config.json").is_file()
    )


def read_snapshot(sample_dir: Path, snapshot: str) -> Mapping[str, object]:
    path = sample_dir / "metrics" / f"{snapshot}.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"Incomplete metric snapshot: {path}")
    return payload


def metric_rows(sample_dir: Path) -> Iterable[Dict[str, object]]:
    config = json.loads((sample_dir / "config.json").read_text(encoding="utf-8"))
    sample_id = str(config["sample_id"])
    for snapshot in SNAPSHOTS:
        report = read_snapshot(sample_dir, snapshot)
        metrics = report["metrics"]
        for k in REFINEMENT_STEPS:
            for scope in SCOPES:
                values = metrics[f"k{k}"][scope]
                yield {
                    "sample_directory": sample_dir.name,
                    "sample_id": sample_id,
                    "snapshot": snapshot,
                    "step": int(report["step"]),
                    "training_stage": report["stage"],
                    "k": k,
                    "scope": scope,
                    **{metric: values[metric] for metric in METRICS},
                }


def relative_gain(before: float, after: float) -> float:
    return (before - after) / max(abs(before), 1e-12)


def contribution_report(sample_dir: Path) -> Dict[str, object]:
    snapshots = {
        name: read_snapshot(sample_dir, name)["metrics"]
        for name in SNAPSHOTS
    }
    initial_k0 = snapshots["initial"]["k0"]
    stage1_k0 = snapshots["stage1"]["k0"]
    stage1_k3 = snapshots["stage1"]["k3"]
    final_k0 = snapshots["final"]["k0"]
    final_k3 = snapshots["final"]["k3"]

    def compare(
        before: Mapping[str, Mapping[str, float]],
        after: Mapping[str, Mapping[str, float]],
    ) -> Dict[str, object]:
        return {
            scope: {
                metric: {
                    "before": before[scope][metric],
                    "after": after[scope][metric],
                    "relative_gain": relative_gain(
                        float(before[scope][metric]),
                        float(after[scope][metric]),
                    ),
                }
                for metric in ("point_rel", "depth_rel")
            }
            for scope in SCOPES
        }

    return {
        "sample_directory": sample_dir.name,
        "sample_id": json.loads(
            (sample_dir / "config.json").read_text(encoding="utf-8")
        )["sample_id"],
        "base_stage1": compare(initial_k0, stage1_k0),
        "ssr_stage1": compare(stage1_k0, stage1_k3),
        "joint_stage": compare(stage1_k3, final_k3),
        "total_pipeline": compare(initial_k0, final_k3),
        "final_k3_better_than_k0": {
            scope: (
                final_k3[scope]["point_rel"]
                < final_k0[scope]["point_rel"]
            )
            for scope in SCOPES
        },
        "targets": {
            "full_depth_rel_below_1_percent": (
                final_k3["full"]["depth_rel"] < 0.01
            ),
            "full_delta_1.01_above_95_percent": (
                final_k3["full"]["depth_delta_1.01"] > 0.95
            ),
            "structure_point_rel_below_2_percent": (
                final_k3["structure"]["point_rel"] < 0.02
            ),
        },
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_manifest(
    root: Path,
    samples: Iterable[Path],
    *,
    compute_hashes: bool,
) -> list[Dict[str, object]]:
    records: list[Dict[str, object]] = []
    for sample_dir in samples:
        checkpoint_dir = sample_dir / "checkpoints"
        if not checkpoint_dir.is_dir():
            continue
        for checkpoint in sorted(checkpoint_dir.glob("*.pt")):
            records.append(
                {
                    "sample_directory": sample_dir.name,
                    "path": str(checkpoint.relative_to(root)),
                    "size_bytes": checkpoint.stat().st_size,
                    "sha256": sha256(checkpoint) if compute_hashes else None,
                }
            )
    return records


def write_csv(path: Path, rows: list[Mapping[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def run(args: argparse.Namespace) -> None:
    root = args.experiment_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)
    samples = sample_directories(root)
    if len(samples) != 5:
        raise ValueError(f"Expected five exp9 sample directories, found {len(samples)}")

    rows = [row for sample_dir in samples for row in metric_rows(sample_dir)]
    contributions = [contribution_report(sample_dir) for sample_dir in samples]
    checkpoints = checkpoint_manifest(
        root,
        samples,
        compute_hashes=args.hash_checkpoints,
    )
    payload = {
        "status": "complete",
        "experiment": root.name,
        "sample_count": len(samples),
        "snapshots": list(SNAPSHOTS),
        "refinement_steps": list(REFINEMENT_STEPS),
        "scopes": list(SCOPES),
        "metric_rows": rows,
        "contributions": contributions,
        "acceptance": {
            "final_k3_better_than_k0_full_count": sum(
                item["final_k3_better_than_k0"]["full"]
                for item in contributions
            ),
            "final_k3_better_than_k0_structure_count": sum(
                item["final_k3_better_than_k0"]["structure"]
                for item in contributions
            ),
            "full_depth_rel_below_1_percent_count": sum(
                item["targets"]["full_depth_rel_below_1_percent"]
                for item in contributions
            ),
            "full_delta_1.01_above_95_percent_count": sum(
                item["targets"]["full_delta_1.01_above_95_percent"]
                for item in contributions
            ),
            "structure_point_rel_below_2_percent_count": sum(
                item["targets"]["structure_point_rel_below_2_percent"]
                for item in contributions
            ),
        },
        "checkpoints": checkpoints,
    }
    write_csv(root / "metrics_summary.csv", rows)
    (root / "metrics_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "checkpoint_manifest.json").write_text(
        json.dumps(checkpoints, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["acceptance"], ensure_ascii=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
