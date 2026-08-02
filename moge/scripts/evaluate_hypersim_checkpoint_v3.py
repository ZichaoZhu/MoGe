from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from moge.model.v3 import MoGeModel
from moge.scripts.train_hypersim_generalization_v3 import (
    METRIC_KEYS,
    SPLITS,
    aggregate_by_split,
    changes_from_k0,
    save_csv,
)
from moge.scripts.train_hypersim_smallset_v3 import (
    cache_base_predictions,
    evaluate,
)
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one MoGe-3 SSR checkpoint on all Hypersim splits"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
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
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if sorted(set(args.evaluation_steps)) != args.evaluation_steps:
        raise ValueError("Evaluation steps must be sorted and unique")
    if not args.evaluation_steps or args.evaluation_steps[0] != 0:
        raise ValueError("Evaluation steps must start at K=0")
    if 3 not in args.evaluation_steps:
        raise ValueError("Evaluation steps must include K=3 for comparison")

    data = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    checkpoint_path = assert_safe_path(
        args.checkpoint,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = assert_safe_path(
        data / "manifest.json",
        safe_root=args.safe_root,
        must_exist=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    pretrained = args.pretrained or checkpoint.get("pretrained")
    if not pretrained:
        raise ValueError("Checkpoint does not identify the pretrained base model")

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    model = MoGeModel.from_pretrained(pretrained).to(device).eval()
    checkpoint_type = "full_model" if "model" in checkpoint else "ssr_only"
    if checkpoint_type == "full_model":
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.ssr.load_state_dict(checkpoint["ssr"], strict=True)
    samples = cache_base_predictions(
        data,
        manifest,
        model,
        height=args.height,
        width=args.width,
        num_tokens=args.num_tokens,
        device=device,
        safe_root=args.safe_root,
    )
    refiner = model.ssr
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    records_by_k: Dict[int, List[Dict[str, object]]] = {}
    metrics_by_k: Dict[str, Dict[str, Dict[str, float]]] = {}
    for refinement_step in args.evaluation_steps:
        records, predictions = evaluate(
            refiner,
            samples,
            device=device,
            refinement_steps=max(1, refinement_step),
            boundary_threshold=args.boundary_threshold,
            use_refiner=refinement_step > 0,
            batch_size=args.batch_size,
        )
        records_by_k[refinement_step] = records
        metrics_by_k[str(refinement_step)] = aggregate_by_split(records)
        del predictions

    records_by_id = {
        step: {record["id"]: record for record in records}
        for step, records in records_by_k.items()
    }
    per_frame: List[Dict[str, object]] = []
    fractions: Dict[str, Dict[str, float]] = {}
    for split in SPLITS:
        split_samples = [sample for sample in samples if sample.split == split]
        split_rows: List[Dict[str, object]] = []
        for sample in split_samples:
            row: Dict[str, object] = {
                "id": sample.sample_id,
                "split": split,
                "scene": sample.scene,
                "frame": sample.frame,
            }
            for refinement_step in args.evaluation_steps:
                source = records_by_id[refinement_step][sample.sample_id]
                for key in METRIC_KEYS:
                    row[f"k{refinement_step}_{key}"] = source[key]
            split_rows.append(row)
            per_frame.append(row)
        fractions[split] = {
            "point_rel_improved_k3_vs_k0": float(
                np.mean(
                    [
                        row["k3_point_rel"] < row["k0_point_rel"]
                        for row in split_rows
                    ]
                )
            ),
            "depth_rel_improved_k3_vs_k0": float(
                np.mean(
                    [
                        row["k3_depth_rel"] < row["k0_depth_rel"]
                        for row in split_rows
                    ]
                )
            ),
            "boundary_f1_improved_k3_vs_k0": float(
                np.mean(
                    [
                        row["k3_boundary_f1"] > row["k0_boundary_f1"]
                        for row in split_rows
                    ]
                )
            ),
        }

    save_csv(output / "per_frame_metrics.csv", per_frame)
    report = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_type": checkpoint_type,
        "pretrained": pretrained,
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "evaluation_refinement_steps": args.evaluation_steps,
        "counts": manifest["counts"],
        "metrics_by_k": metrics_by_k,
        "k3_changes_from_k0": changes_from_k0(metrics_by_k, 3),
        "fraction_improved": fractions,
        "elapsed_seconds": time.perf_counter() - start,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "normal_prediction": False,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
