from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.scan_checkpoint_residuals import (
    residual_rows_for_output,
    sha256_file,
    summarize_rows,
    write_csv_atomic,
    write_json_atomic,
)


MODE_SPECS = (
    ("base_eval_ssr_eval", "eval", "eval"),
    ("base_eval_ssr_train", "eval", "train"),
    ("base_train_ssr_eval", "train", "eval"),
    ("base_train_ssr_train", "train", "train"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Factorial audit of Base and SSR train/eval module states."
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
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--stochastic-repeats", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=181)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    parser.add_argument("--max-samples-per-split", type=int)
    return parser.parse_args()


def configure_modes(model: Any, *, base_mode: str, ssr_mode: str) -> None:
    if base_mode not in {"eval", "train"} or ssr_mode not in {"eval", "train"}:
        raise ValueError("Base and SSR modes must be eval or train")
    model.eval() if base_mode == "eval" else model.train()
    model.ssr.eval() if ssr_mode == "eval" else model.ssr.train()


def _seed_forward(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def scan_combination(
    model: Any,
    samples: Iterable[Any],
    *,
    label: str,
    base_mode: str,
    ssr_mode: str,
    repeats: int,
    device: torch.device,
    num_tokens: int,
    refinement_steps: int,
    threshold: float,
    seed: int,
) -> list[dict[str, Any]]:
    configure_modes(model, base_mode=base_mode, ssr_mode=ssr_mode)
    sample_list = list(samples)
    rows: list[dict[str, Any]] = []
    for repeat in range(repeats):
        for sample_index, sample in enumerate(sample_list):
            _seed_forward(seed + 1_000_003 * repeat + sample_index)
            image = sample.image.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(
                    image,
                    num_tokens=num_tokens,
                    num_refinement_steps=refinement_steps,
                    return_intermediates=True,
                    detach_base_from_refiner=False,
                )
            sample_rows = residual_rows_for_output(
                output,
                sample_id=sample.sample_id,
                split=sample.split,
                scene=sample.scene,
                frame=sample.frame,
                mode=label,
                repeat=repeat,
                threshold=threshold,
                highlighted=False,
            )
            rows.extend(sample_rows)
            print(
                json.dumps(
                    {
                        "mode": label,
                        "repeat": repeat,
                        "sample_index": sample_index + 1,
                        "sample_count": len(sample_list),
                        "id": sample.sample_id,
                        "maximum": max(
                            float(row["max_abs_residual"]) for row in sample_rows
                        ),
                        "exceeds_threshold": any(
                            float(row["max_abs_residual"]) > threshold
                            for row in sample_rows
                        ),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del image, output
    return rows


def mode_decision(
    rows: list[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    maxima: dict[str, dict[str, float]] = {}
    for label, _, _ in MODE_SPECS:
        selected = [row for row in rows if row["mode"] == label]
        by_sample: dict[str, float] = {}
        for row in selected:
            sample_id = str(row["id"])
            by_sample[sample_id] = max(
                by_sample.get(sample_id, 0.0),
                float(row["max_abs_residual"]),
            )
        maxima[label] = by_sample
    outlier_counts = {
        label: sum(value > threshold for value in by_sample.values())
        for label, by_sample in maxima.items()
    }
    global_maxima = {
        label: max(by_sample.values())
        for label, by_sample in maxima.items()
    }
    ssr_eval_labels = ("base_eval_ssr_eval", "base_train_ssr_eval")
    ssr_train_labels = ("base_eval_ssr_train", "base_train_ssr_train")
    return {
        "outlier_sample_counts": outlier_counts,
        "global_maxima": global_maxima,
        "ssr_eval_always_worse_than_matching_ssr_train": (
            global_maxima["base_eval_ssr_eval"]
            > global_maxima["base_eval_ssr_train"]
            and global_maxima["base_train_ssr_eval"]
            > global_maxima["base_train_ssr_train"]
        ),
        "outliers_only_when_ssr_eval": (
            any(outlier_counts[label] for label in ssr_eval_labels)
            and not any(outlier_counts[label] for label in ssr_train_labels)
        ),
        "base_mode_changes_outlier_presence_with_fixed_ssr_mode": (
            (outlier_counts["base_eval_ssr_eval"] > 0)
            != (outlier_counts["base_train_ssr_eval"] > 0)
            or (outlier_counts["base_eval_ssr_train"] > 0)
            != (outlier_counts["base_train_ssr_train"] > 0)
        ),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from moge.model.v3 import MoGeModel
    from moge.scripts.train_hypersim_joint_v3 import load_raw_samples

    if not 1 <= args.refinement_steps <= 7:
        raise ValueError("--refinement-steps must be in [1, 7]")
    if args.stochastic_repeats <= 0:
        raise ValueError("--stochastic-repeats must be positive")
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive")

    data = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    checkpoint_path = assert_safe_path(
        args.checkpoint,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    samples = load_raw_samples(
        data,
        manifest,
        height=args.height,
        width=args.width,
        safe_root=args.safe_root,
        splits=tuple(args.splits),
    )
    if args.max_samples_per_split is not None:
        if args.max_samples_per_split <= 0:
            raise ValueError("--max-samples-per-split must be positive")
        samples = [
            sample
            for split in args.splits
            for sample in [entry for entry in samples if entry.split == split][
                : args.max_samples_per_split
            ]
        ]

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained = args.pretrained or checkpoint.get("pretrained")
    if not pretrained:
        raise ValueError("Checkpoint does not identify its pretrained base model")
    device = torch.device(args.device)
    model = MoGeModel.from_pretrained(str(pretrained)).to(device)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    mode_repeats: dict[str, int] = {}
    for mode_index, (label, base_mode, ssr_mode) in enumerate(MODE_SPECS):
        model.load_state_dict(checkpoint["model"], strict=True)
        repeats = (
            1
            if base_mode == "eval" and ssr_mode == "eval"
            else args.stochastic_repeats
        )
        mode_repeats[label] = repeats
        rows.extend(
            scan_combination(
                model,
                samples,
                label=label,
                base_mode=base_mode,
                ssr_mode=ssr_mode,
                repeats=repeats,
                device=device,
                num_tokens=args.num_tokens,
                refinement_steps=args.refinement_steps,
                threshold=args.threshold,
                seed=args.seed + 10_000_019 * mode_index,
            )
        )

    write_csv_atomic(output / "per_frame_module_modes.csv", rows)
    report: dict[str, Any] = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "pretrained": str(pretrained),
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "refinement_steps": args.refinement_steps,
        "mode_repeats": mode_repeats,
        "splits": list(args.splits),
        "sample_count": len(samples),
        "summary": summarize_rows(rows, threshold=args.threshold),
        "mode_decision": mode_decision(rows, threshold=args.threshold),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output / "report.json", report)
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
