from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from moge.model.ssr import factorize_points, unfactorize_points
from moge.model.v3 import MoGeModel
from moge.scripts.overfit_hypersim_staged_v3 import build_structure_mask
from moge.scripts.train_hypersim_joint_v3 import (
    RawSample,
    load_fine_structure_rois,
    load_raw_samples,
)
from moge.scripts.train_hypersim_smallset_v3 import depth_edge_map
from moge.train.losses import edge_loss
from moge.train.losses_v3 import (
    affine_invariant_global_loss_v3,
    edge_angle_loss_v3,
    radial_partition_local_loss,
)
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit MoGe-3 global/local/edge loss gradients with respect to "
            "per-pixel log depth on locked fine-structure samples."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--fine-structure-rois", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument("--refinement-steps", type=int, nargs="+", default=[0, 3])
    parser.add_argument(
        "--train-roi-samples",
        type=int,
        default=12,
        help=(
            "Use the first N ROI-bearing train samples in the pre-prediction "
            "data manifest; all locked validation/test ROIs are retained."
        ),
    )
    parser.add_argument("--local-scales", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=251)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError("Gradient audit cannot write an empty CSV")
    temporary = path.with_suffix(path.suffix + ".incomplete")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def select_locked_samples(
    samples: Sequence[RawSample],
    *,
    train_limit: int,
) -> list[RawSample]:
    if train_limit < 0:
        raise ValueError("train_limit cannot be negative")
    train = [
        sample
        for sample in samples
        if sample.split == "train" and sample.crop_xyxy is not None
    ][:train_limit]
    leaveout = [
        sample
        for sample in samples
        if sample.split in {"val", "test"} and sample.crop_xyxy is not None
    ]
    selected = train + leaveout
    if not selected:
        raise ValueError("No locked fine-structure samples were selected")
    return selected


def full_resolution_masks(
    sample: RawSample,
    *,
    boundary_threshold: float,
) -> dict[str, torch.Tensor]:
    if sample.crop_xyxy is None:
        raise ValueError(f"Sample {sample.sample_id} has no locked crop")
    gt = sample.gt_points
    valid = torch.isfinite(gt).all(dim=-1) & (gt[..., 2] > 0)
    crop = torch.zeros_like(valid)
    structure = torch.zeros_like(valid)
    x0, y0, x1, y1 = sample.crop_xyxy
    crop[y0:y1, x0:x1] = valid[y0:y1, x0:x1]
    structure_crop = build_structure_mask(
        gt,
        sample.crop_xyxy,
        sample.structure_mask_config
        or {"type": "near_quantile", "quantile": 0.5},
    )
    structure[y0:y1, x0:x1] = structure_crop
    boundary = depth_edge_map(gt[..., 2], valid, boundary_threshold)
    return {
        "valid": valid,
        "crop": crop,
        "structure": structure,
        "gt_boundary": boundary,
    }


def gradient_statistics(
    gradient: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
) -> dict[str, float | int]:
    absolute = gradient.detach().abs()
    valid = masks["valid"].to(device=gradient.device)
    valid_values = absolute[valid]
    if not valid_values.numel():
        raise ValueError("Gradient audit sample has no valid pixels")
    total_mass = valid_values.sum().clamp_min(1e-30)
    valid_pixels = int(valid.sum().item())
    result: dict[str, float | int] = {
        "valid_pixels": valid_pixels,
        "gradient_abs_mean": float(valid_values.mean().item()),
        "gradient_rms": float(valid_values.square().mean().sqrt().item()),
        "gradient_abs_max": float(valid_values.max().item()),
        "gradient_l1_mass": float(total_mass.item()),
        "gradient_nonzero_fraction": float((valid_values > 0).float().mean().item()),
    }
    for name in ("crop", "structure", "gt_boundary"):
        mask = masks[name].to(device=gradient.device) & valid
        pixels = int(mask.sum().item())
        pixel_share = pixels / valid_pixels
        mass_share = float((absolute[mask].sum() / total_mass).item())
        result[f"{name}_pixels"] = pixels
        result[f"{name}_pixel_share"] = pixel_share
        result[f"{name}_gradient_mass_share"] = mass_share
        result[f"{name}_gradient_enrichment"] = (
            mass_share / pixel_share if pixel_share > 0 else math.nan
        )
    return result


def gradient_cosine(left: torch.Tensor, right: torch.Tensor, valid: torch.Tensor) -> float:
    left_values = left[valid]
    right_values = right[valid]
    denominator = left_values.norm() * right_values.norm()
    if float(denominator) == 0:
        return math.nan
    return float(torch.dot(left_values, right_values).div(denominator).item())


def objective_gradients(
    prediction: torch.Tensor,
    gt_points: torch.Tensor,
    masks: Mapping[str, torch.Tensor],
    *,
    local_scales: Sequence[int],
    seed: int,
) -> list[dict[str, float | int | str]]:
    factorized = factorize_points(prediction.float())
    uv = factorized[..., :2].detach()
    initial_zeta = factorized[..., 2].detach()
    gt_batch = gt_points.float().unsqueeze(0)
    gradients: dict[str, torch.Tensor] = {}
    losses: dict[str, float] = {}

    def differentiate(name: str, loss_builder: Any) -> None:
        zeta = initial_zeta.clone().requires_grad_(True)
        points = unfactorize_points(
            torch.cat((uv, zeta[..., None]), dim=-1)
        ).unsqueeze(0)
        loss = loss_builder(points)
        gradient = torch.autograd.grad(loss, zeta)[0]
        if not torch.isfinite(gradient).all():
            raise RuntimeError(f"Non-finite {name} gradient")
        losses[name] = float(loss.detach().item())
        gradients[name] = gradient.detach()

    differentiate(
        "global",
        lambda points: affine_invariant_global_loss_v3(
            points,
            gt_batch,
        )[0].mean(),
    )

    def local_builder(points: torch.Tensor) -> torch.Tensor:
        _, alignment = affine_invariant_global_loss_v3(points, gt_batch)
        generator = torch.Generator(device=points.device).manual_seed(seed)
        return radial_partition_local_loss(
            points,
            gt_batch,
            alignment,
            scales=local_scales,
            generator=generator,
        ).mean()

    differentiate("local", local_builder)
    differentiate(
        "edge_legacy_max",
        lambda points: edge_loss(
            points,
            gt_batch,
            normalization_dimension="max",
        )[0].mean(),
    )
    differentiate(
        "edge_paper_min",
        lambda points: edge_angle_loss_v3(points, gt_batch)[0].mean(),
    )
    losses["combined_paper"] = (
        losses["global"] + losses["local"] + losses["edge_paper_min"]
    )
    gradients["combined_paper"] = (
        gradients["global"] + gradients["local"] + gradients["edge_paper_min"]
    )

    valid = masks["valid"].to(device=prediction.device)
    cosines = {
        "cosine_global_local": gradient_cosine(
            gradients["global"], gradients["local"], valid
        ),
        "cosine_global_edge": gradient_cosine(
            gradients["global"], gradients["edge_paper_min"], valid
        ),
        "cosine_local_edge": gradient_cosine(
            gradients["local"], gradients["edge_paper_min"], valid
        ),
    }
    rows = []
    for name in (
        "global",
        "local",
        "edge_legacy_max",
        "edge_paper_min",
        "combined_paper",
    ):
        rows.append(
            {
                "objective": name,
                "loss": losses[name],
                **gradient_statistics(gradients[name], masks),
                **cosines,
            }
        )
    return rows


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["split"]), int(row["k"]), str(row["objective"]))].append(
            row
        )
    ignored = {"sample_id", "scene", "frame", "split", "k", "objective"}
    aggregates = []
    for (split, step, objective), selected in sorted(groups.items()):
        record: dict[str, Any] = {
            "split": split,
            "k": step,
            "objective": objective,
            "sample_count": len(selected),
        }
        for key in selected[0]:
            if key in ignored:
                continue
            values = np.asarray(
                [float(row[key]) for row in selected],
                dtype=np.float64,
            )
            finite = values[np.isfinite(values)]
            if finite.size:
                record[f"{key}_mean"] = float(finite.mean())
                record[f"{key}_median"] = float(np.median(finite))
        aggregates.append(record)
    return aggregates


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.refinement_steps != sorted(set(args.refinement_steps)):
        raise ValueError("refinement_steps must be sorted and unique")
    if args.refinement_steps[0] < 0 or args.refinement_steps[-1] > 7:
        raise ValueError("refinement_steps must be in [0, 7]")
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
    samples = load_raw_samples(
        data,
        manifest,
        height=args.height,
        width=args.width,
        safe_root=args.safe_root,
        splits=("train", "val", "test"),
        fine_structure_rois=rois,
    )
    selected = select_locked_samples(
        samples,
        train_limit=args.train_roi_samples,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained = checkpoint.get("pretrained")
    if not pretrained:
        raise ValueError("Checkpoint does not identify its pretrained base model")
    normalization = checkpoint.get("args", {}).get(
        "ssr_normalization",
        "batch_norm",
    )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    model = MoGeModel.from_pretrained(
        str(pretrained),
        model_kwargs={"ssr": {"normalization": normalization}},
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    for sample_index, sample in enumerate(selected):
        masks = full_resolution_masks(
            sample,
            boundary_threshold=args.boundary_threshold,
        )
        with torch.inference_mode():
            output_dict = model(
                sample.image.unsqueeze(0).to(device),
                num_tokens=args.num_tokens,
                num_refinement_steps=args.refinement_steps[-1],
                return_intermediates=True,
            )
        sequence = output_dict["points_sequence"]
        gt = sample.gt_points.to(device)
        for step in args.refinement_steps:
            objective_rows = objective_gradients(
                sequence[step][0].detach(),
                gt,
                masks,
                local_scales=tuple(args.local_scales),
                seed=args.seed + 1009 * sample_index + step,
            )
            for row in objective_rows:
                rows.append(
                    {
                        "sample_id": sample.sample_id,
                        "scene": sample.scene,
                        "frame": sample.frame,
                        "split": sample.split,
                        "k": step,
                        **row,
                    }
                )
        del output_dict, sequence, gt

    atomic_write_csv(output / "per_sample_gradients.csv", rows)
    aggregates = aggregate_rows(rows)
    atomic_write_csv(output / "aggregate_gradients.csv", aggregates)
    legacy = [
        row
        for row in rows
        if row["objective"] == "edge_legacy_max"
    ]
    paper = [
        row
        for row in rows
        if row["objective"] == "edge_paper_min"
    ]
    edge_ratios = [
        {
            "sample_id": left["sample_id"],
            "k": left["k"],
            "loss_ratio": float(right["loss"]) / max(float(left["loss"]), 1e-30),
            "gradient_l1_ratio": float(right["gradient_l1_mass"])
            / max(float(left["gradient_l1_mass"]), 1e-30),
        }
        for left, right in zip(legacy, paper, strict=True)
    ]
    report: dict[str, Any] = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "ssr_normalization": normalization,
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "refinement_steps": args.refinement_steps,
        "local_scales": args.local_scales,
        "selection": {
            "policy": (
                "first N ROI-bearing train samples in the pre-prediction data "
                "manifest, plus every locked validation/test ROI"
            ),
            "train_roi_limit": args.train_roi_samples,
            "sample_ids": [sample.sample_id for sample in selected],
            "counts": {
                split: sum(sample.split == split for sample in selected)
                for split in ("train", "val", "test")
            },
        },
        "gradient_target": (
            "Per-pixel log depth zeta with fixed u,v; this measures objective "
            "pressure before the SSR network Jacobian."
        ),
        "edge_formula": {
            "legacy_normalizer": "max(H,W)",
            "paper_normalizer": "min(H,W)",
            "expected_paper_to_legacy_ratio": max(args.height, args.width)
            / min(args.height, args.width),
            "observed_loss_ratio_mean": float(
                np.mean([row["loss_ratio"] for row in edge_ratios])
            ),
            "observed_gradient_l1_ratio_mean": float(
                np.mean([row["gradient_l1_ratio"] for row in edge_ratios])
            ),
        },
        "peak_cuda_memory_bytes": (
            {
                "allocated": int(torch.cuda.max_memory_allocated(device)),
                "reserved": int(torch.cuda.max_memory_reserved(device)),
            }
            if device.type == "cuda"
            else None
        ),
        "aggregates": aggregates,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output / "report.json", report)
    return report


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
