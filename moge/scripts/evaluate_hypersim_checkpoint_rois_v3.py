from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import torch

from moge.model.v3 import MoGeModel
from moge.model.ssr import stateless_batch_statistics
from moge.scripts.train_hypersim_joint_v3 import (
    RawSample,
    aggregate_evaluation,
    evaluate_model,
    load_fine_structure_rois,
    load_raw_samples,
)
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a MoGe-3 checkpoint on full Hypersim splits and locked "
            "fine-structure ROIs."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fine-structure-rois", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        nargs="+",
        default=[0, 1, 3, 5],
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    parser.add_argument(
        "--ssr-batch-statistics",
        action="store_true",
        help=(
            "Use current SSR sparse-feature batch statistics in eval mode "
            "without reading or updating BatchNorm running buffers."
        ),
    )
    parser.add_argument(
        "--ssr-iteration-statistics",
        type=Path,
        help=(
            "Torch file containing per-iteration SSR BatchNorm states under "
            "the key iteration_states."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def flatten_per_frame(
    records: Dict[int, List[Dict[str, object]]],
) -> List[Dict[str, object]]:
    by_step = {
        step: {str(record["id"]): record for record in step_records}
        for step, step_records in records.items()
    }
    first_step = min(records)
    rows = []
    for first in records[first_step]:
        sample_id = str(first["id"])
        row: Dict[str, object] = {
            "id": sample_id,
            "split": first["split"],
            "scene": first["scene"],
            "frame": first["frame"],
        }
        for step in sorted(records):
            source = by_step[step][sample_id]
            for key, value in source.items():
                if key not in {"id", "split", "scene", "frame"}:
                    row[f"k{step}_{key}"] = value
        rows.append(row)
    return rows


def relative_reduction(base: float, refined: float) -> float:
    return (base - refined) / max(base, 1e-12)


def changes_from_k0(
    metrics: Dict[str, Dict[str, Dict[str, float]]],
    step: int,
) -> Dict[str, Dict[str, float]]:
    changes: Dict[str, Dict[str, float]] = {}
    for split, refined in metrics[str(step)].items():
        base = metrics["0"][split]
        split_changes = {}
        for scope in ("full", "crop", "structure"):
            key = "point_rel" if scope == "full" else f"{scope}_point_rel"
            if key in base and key in refined:
                split_changes[f"{scope}_point_rel_relative_reduction"] = (
                    relative_reduction(base[key], refined[key])
                )
        changes[split] = split_changes
    return changes


def fraction_improved(
    rows: List[Dict[str, object]],
    *,
    step: int,
) -> Dict[str, Dict[str, float]]:
    result = {}
    for split in ("train", "val", "test"):
        selected = [row for row in rows if row["split"] == split]
        split_result = {}
        for scope in ("full", "crop", "structure"):
            suffix = "point_rel" if scope == "full" else f"{scope}_point_rel"
            base_key = f"k0_{suffix}"
            refined_key = f"k{step}_{suffix}"
            scoped = [
                row
                for row in selected
                if base_key in row and refined_key in row
            ]
            if scoped:
                split_result[f"{scope}_point_rel"] = sum(
                    float(row[refined_key]) < float(row[base_key])
                    for row in scoped
                ) / len(scoped)
        result[split] = split_result
    return result


def run(args: argparse.Namespace) -> Dict[str, object]:
    steps = sorted(set(int(step) for step in args.evaluation_steps))
    if steps != args.evaluation_steps or steps[0] != 0:
        raise ValueError("Evaluation steps must be sorted, unique, and start at K=0")
    if max(steps) > 7:
        raise ValueError("Evaluation steps cannot exceed K=7")
    data = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    checkpoint_path = assert_safe_path(
        args.checkpoint,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(
        args.output,
        safe_root=args.safe_root,
        writable=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    rois = load_fine_structure_rois(
        args.fine_structure_rois,
        safe_root=args.safe_root,
        height=args.height,
        width=args.width,
    )
    samples: List[RawSample] = load_raw_samples(
        data,
        manifest,
        height=args.height,
        width=args.width,
        safe_root=args.safe_root,
        splits=("train", "val", "test"),
        fine_structure_rois=rois,
    )
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    pretrained = args.pretrained or checkpoint.get("pretrained")
    if not pretrained:
        raise ValueError("Checkpoint does not identify its pretrained base model")
    checkpoint_args = checkpoint.get("args", {})
    ssr_normalization = checkpoint_args.get(
        "ssr_normalization",
        "batch_norm",
    )
    device = torch.device(args.device)
    start = time.perf_counter()
    model = MoGeModel.from_pretrained(
        str(pretrained),
        model_kwargs={"ssr": {"normalization": ssr_normalization}},
    ).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if args.ssr_batch_statistics and args.ssr_iteration_statistics is not None:
        raise ValueError(
            "Stateless and per-iteration SSR statistics are mutually exclusive"
        )
    iteration_states = None
    iteration_statistics_sha256 = None
    if args.ssr_iteration_statistics is not None:
        statistics_path = assert_safe_path(
            args.ssr_iteration_statistics,
            safe_root=args.safe_root,
            must_exist=True,
        )
        statistics_payload = torch.load(
            statistics_path,
            map_location="cpu",
            weights_only=False,
        )
        iteration_states = statistics_payload["iteration_states"]
        if len(iteration_states) < steps[-1]:
            raise ValueError(
                "Per-iteration statistics do not cover all evaluation steps"
            )
        iteration_statistics_sha256 = sha256_file(statistics_path)
    statistics_context = (
        stateless_batch_statistics(model.ssr)
        if args.ssr_batch_statistics
        else nullcontext()
    )
    with statistics_context:
        records = evaluate_model(
            model,
            samples,
            device=device,
            num_tokens=args.num_tokens,
            refinement_steps=steps,
            batch_size=args.batch_size,
            boundary_threshold=args.boundary_threshold,
            ssr_batch_norm_states=iteration_states,
        )
    metrics = aggregate_evaluation(records)
    per_frame = flatten_per_frame(records)
    save_csv(output / "per_frame_metrics.csv", per_frame)
    report: Dict[str, object] = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "ssr_normalization": ssr_normalization,
        "checkpoint_selection": checkpoint.get("selection"),
        "pretrained": pretrained,
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "evaluation_refinement_steps": steps,
        "ssr_normalization_policy": (
            f"batch_independent_{ssr_normalization}"
            if ssr_normalization != "batch_norm"
            else (
                "current_batch_statistics_without_running_buffers"
                if args.ssr_batch_statistics
                else (
                    "iteration_specific_running_statistics"
                    if iteration_states is not None
                    else "stored_running_statistics"
                )
            )
        ),
        "ssr_iteration_statistics_sha256": iteration_statistics_sha256,
        "counts": manifest["counts"],
        "roi_counts": {
            split: sum(
                sample.split == split and sample.crop_xyxy is not None
                for sample in samples
            )
            for split in ("train", "val", "test")
        },
        "metrics_by_k": metrics,
        "changes_from_k0": {
            str(step): changes_from_k0(metrics, step)
            for step in steps
            if step
        },
        "fraction_improved": {
            str(step): fraction_improved(per_frame, step=step)
            for step in steps
            if step
        },
        "elapsed_seconds": time.perf_counter() - start,
    }
    temporary = output / "report.json.incomplete"
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output / "report.json")
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
