from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from collections import defaultdict
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Scan every per-iteration SSR log-depth residual in a MoGe-3 "
            "checkpoint under deterministic eval and single-image train modes."
        )
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
    parser.add_argument("--train-mode-repeats", type=int, default=3)
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("eval", "train"),
        default=["eval", "train"],
        help="Module modes to scan. Defaults to both eval and train.",
    )
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=171)
    parser.add_argument(
        "--ssr-batch-statistics",
        action="store_true",
        help=(
            "Use current SSR sparse-feature statistics without reading or "
            "updating BatchNorm running buffers. Requires --modes eval."
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
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--highlight-ids", nargs="*", default=[])
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv_atomic(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("Cannot write an empty residual scan")
    temporary = path.with_suffix(path.suffix + ".incomplete")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    threshold: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Residual summary requires at least one row")
    grouped: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), str(row["split"]), int(row["iteration"]))].append(
            row
        )

    aggregates = []
    for (mode, split, iteration), selected in sorted(grouped.items()):
        worst = max(selected, key=lambda row: float(row["max_abs_residual"]))
        sample_ids = {str(row["id"]) for row in selected}
        exceeding_ids = {
            str(row["id"])
            for row in selected
            if float(row["max_abs_residual"]) > threshold
        }
        aggregates.append(
            {
                "mode": mode,
                "split": split,
                "iteration": iteration,
                "row_count": len(selected),
                "sample_count": len(sample_ids),
                "samples_exceeding_threshold": len(exceeding_ids),
                "maximum": float(worst["max_abs_residual"]),
                "maximum_sample_id": str(worst["id"]),
                "maximum_repeat": int(worst["repeat"]),
                "maximum_row": int(worst["max_row"]),
                "maximum_col": int(worst["max_col"]),
            }
        )

    worst = max(rows, key=lambda row: float(row["max_abs_residual"]))
    highlighted = [row for row in rows if bool(row["highlighted"])]
    worst_highlighted = (
        max(highlighted, key=lambda row: float(row["max_abs_residual"]))
        if highlighted
        else None
    )
    unique_exceeding = sorted(
        {
            str(row["id"])
            for row in rows
            if float(row["max_abs_residual"]) > threshold
        }
    )
    return {
        "threshold": threshold,
        "row_count": len(rows),
        "unique_sample_count": len({str(row["id"]) for row in rows}),
        "unique_samples_exceeding_threshold": unique_exceeding,
        "unique_sample_count_exceeding_threshold": len(unique_exceeding),
        "global_maximum": {
            key: worst[key]
            for key in (
                "mode",
                "repeat",
                "id",
                "split",
                "iteration",
                "max_abs_residual",
                "max_signed_residual",
                "max_row",
                "max_col",
                "depth_span",
            )
        },
        "highlighted_maximum": (
            {
                key: worst_highlighted[key]
                for key in (
                    "mode",
                    "repeat",
                    "id",
                    "split",
                    "iteration",
                    "max_abs_residual",
                    "max_signed_residual",
                    "max_row",
                    "max_col",
                    "depth_span",
                )
            }
            if worst_highlighted is not None
            else None
        ),
        "aggregates": aggregates,
    }


def _seed_forward(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def residual_rows_for_output(
    output: dict[str, Any],
    *,
    sample_id: str,
    split: str,
    scene: str,
    frame: int,
    mode: str,
    repeat: int,
    threshold: float,
    highlighted: bool,
) -> list[dict[str, Any]]:
    residuals = output["log_depth_residuals"]
    voxel_stats = output["voxel_stats"]
    if len(residuals) != len(voxel_stats):
        raise RuntimeError("Residual and voxel-stat iteration counts disagree")

    rows = []
    cumulative = None
    for iteration, (residual_batch, stats) in enumerate(
        zip(residuals, voxel_stats, strict=True),
        start=1,
    ):
        residual = residual_batch[0].detach().float()
        if not torch.isfinite(residual).all():
            raise RuntimeError(f"Non-finite residual in {sample_id}, iteration {iteration}")
        absolute = residual.abs()
        flat_index = int(absolute.argmax().item())
        width = residual.shape[-1]
        row, col = divmod(flat_index, width)
        cumulative = residual if cumulative is None else cumulative + residual
        cumulative_absolute = cumulative.abs()
        rows.append(
            {
                "mode": mode,
                "repeat": repeat,
                "id": sample_id,
                "split": split,
                "scene": scene,
                "frame": frame,
                "iteration": iteration,
                "highlighted": highlighted,
                "max_abs_residual": float(absolute[row, col].item()),
                "max_signed_residual": float(residual[row, col].item()),
                "max_row": row,
                "max_col": col,
                "p99_abs_residual": float(torch.quantile(absolute, 0.99).item()),
                "p999_abs_residual": float(torch.quantile(absolute, 0.999).item()),
                "mean_abs_residual": float(absolute.mean().item()),
                "pixels_exceeding_threshold": int((absolute > threshold).sum().item()),
                "cumulative_max_abs_residual": float(cumulative_absolute.max().item()),
                "depth_span": float(stats["depth_span"][0].detach().item()),
                "active_voxels": int(stats["active_voxels"].detach().item()),
            }
        )
    return rows


def scan_mode(
    model: Any,
    samples: Iterable[Any],
    *,
    mode: str,
    repeats: int,
    device: torch.device,
    num_tokens: int,
    refinement_steps: int,
    threshold: float,
    seed: int,
    highlight_ids: set[str],
    ssr_batch_norm_states: list[
        dict[str, dict[str, torch.Tensor]]
    ]
    | None = None,
) -> list[dict[str, Any]]:
    if mode not in {"eval", "train"}:
        raise ValueError(f"Unsupported scan mode: {mode}")
    model.eval() if mode == "eval" else model.train()
    rows: list[dict[str, Any]] = []
    sample_list = list(samples)
    for repeat in range(repeats):
        for sample_index, sample in enumerate(sample_list):
            forward_seed = seed + 1_000_003 * repeat + sample_index
            _seed_forward(forward_seed)
            image = sample.image.unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(
                    image,
                    num_tokens=num_tokens,
                    num_refinement_steps=refinement_steps,
                    return_intermediates=True,
                    detach_base_from_refiner=False,
                    ssr_batch_norm_states=ssr_batch_norm_states,
                )
            rows.extend(
                residual_rows_for_output(
                    output,
                    sample_id=sample.sample_id,
                    split=sample.split,
                    scene=sample.scene,
                    frame=sample.frame,
                    mode=mode,
                    repeat=repeat,
                    threshold=threshold,
                    highlighted=sample.sample_id in highlight_ids,
                )
            )
            sample_rows = rows[-len(output["log_depth_residuals"]) :]
            print(
                json.dumps(
                    {
                        "mode": mode,
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    from moge.model.v3 import MoGeModel
    from moge.scripts.train_hypersim_joint_v3 import load_raw_samples

    if not 1 <= args.refinement_steps <= 7:
        raise ValueError("--refinement-steps must be in [1, 7]")
    if args.train_mode_repeats <= 0:
        raise ValueError("--train-mode-repeats must be positive")
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive")
    if args.ssr_batch_statistics and args.modes != ["eval"]:
        raise ValueError("--ssr-batch-statistics requires exactly --modes eval")
    if args.ssr_iteration_statistics is not None and args.modes != ["eval"]:
        raise ValueError(
            "--ssr-iteration-statistics requires exactly --modes eval"
        )
    if args.ssr_batch_statistics and args.ssr_iteration_statistics is not None:
        raise ValueError(
            "Stateless and per-iteration SSR statistics are mutually exclusive"
        )

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
    checkpoint_args = checkpoint.get("args", {})
    ssr_normalization = checkpoint_args.get(
        "ssr_normalization",
        "batch_norm",
    )

    device = torch.device(args.device)
    started = time.perf_counter()
    model = MoGeModel.from_pretrained(
        str(pretrained),
        model_kwargs={"ssr": {"normalization": ssr_normalization}},
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
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
        if len(iteration_states) < args.refinement_steps:
            raise ValueError(
                "Per-iteration statistics do not cover the residual scan"
            )
        iteration_statistics_sha256 = sha256_file(statistics_path)
    checkpoint_state = {
        key: value.detach().cpu().clone()
        for key, value in checkpoint["model"].items()
    }

    rows: list[dict[str, Any]] = []
    statistics_context = nullcontext()
    if args.ssr_batch_statistics:
        from moge.model.ssr import stateless_batch_statistics

        statistics_context = stateless_batch_statistics(model.ssr)
    with statistics_context:
        if "eval" in args.modes:
            rows.extend(
                scan_mode(
                    model,
                    samples,
                    mode="eval",
                    repeats=1,
                    device=device,
                    num_tokens=args.num_tokens,
                    refinement_steps=args.refinement_steps,
                    threshold=args.threshold,
                    seed=args.seed,
                    highlight_ids=set(args.highlight_ids),
                    ssr_batch_norm_states=iteration_states,
                )
            )
    if "train" in args.modes:
        model.load_state_dict(checkpoint_state, strict=True)
        rows.extend(
            scan_mode(
                model,
                samples,
                mode="train",
                repeats=args.train_mode_repeats,
                device=device,
                num_tokens=args.num_tokens,
                refinement_steps=args.refinement_steps,
                threshold=args.threshold,
                seed=args.seed + 10_000_019,
                highlight_ids=set(args.highlight_ids),
            )
        )

    write_csv_atomic(output / "per_frame_residuals.csv", rows)
    report: dict[str, Any] = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "ssr_normalization": ssr_normalization,
        "pretrained": str(pretrained),
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "refinement_steps": args.refinement_steps,
        "modes": list(args.modes),
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
        "train_mode_repeats": args.train_mode_repeats,
        "splits": list(args.splits),
        "max_samples_per_split": args.max_samples_per_split,
        "sample_count": len(samples),
        "highlight_ids": list(args.highlight_ids),
        "summary": summarize_rows(rows, threshold=args.threshold),
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output / "report.json", report)
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
