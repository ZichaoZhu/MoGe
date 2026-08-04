from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
from accelerate import Accelerator
from accelerate.utils import (
    DistributedDataParallelKwargs,
    InitProcessGroupKwargs,
)

from moge.model.v3 import MoGeModel
from moge.scripts.overfit_hypersim_v3 import (
    aligned_metrics,
    geometry_loss,
    moving_average,
)
from moge.scripts.train_hypersim_generalization_v3 import (
    METRIC_KEYS,
    stratified_subset,
)
from moge.scripts.train_hypersim_smallset_v3 import (
    aggregate_metrics,
    boundary_f1,
    load_raw_sample,
)
from moge.train.stability_v3 import (
    base_geometry_has_collapsed,
    maximum_saturation_fraction,
    raw_residual_abort_reason,
    raw_residual_peak_loss,
    raw_residual_percentiles,
    raw_residual_tail_loss,
    selection_score,
    updated_threshold_streak,
)
from moge.train.trainer_v3 import TrainingScheduleV3, build_v3_optimizer
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


@dataclass
class RawSample:
    sample_id: str
    split: str
    scene: str
    frame: int
    image: torch.Tensor
    gt_points: torch.Tensor
    crop_xyxy: tuple[int, int, int, int] | None = None
    structure_mask_config: Dict[str, object] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage joint MoGe-3 fine-tuning on disjoint Hypersim scenes"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument(
        "--resume",
        type=Path,
        help="Resume model, optimizer, RNG state, and histories exactly.",
    )
    checkpoint_group.add_argument(
        "--warm-start",
        type=Path,
        help="Load model weights and step only, with a fresh optimizer and histories.",
    )
    checkpoint_group.add_argument(
        "--transition-from",
        type=Path,
        help=(
            "Load model, optimizer and RNG state but start fresh histories in "
            "the new output directory. Used for stage changes and recovery."
        ),
    )
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ddp-backend", choices=("nccl", "gloo"), default="nccl")
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Keep the DINO backbone frozen for the complete run.",
    )
    parser.add_argument(
        "--train-refiner-only",
        action="store_true",
        help=(
            "Freeze the complete Base/2D Head and optimize only SSR from "
            "K=1..K losses."
        ),
    )
    parser.add_argument(
        "--fine-structure-rois",
        type=Path,
        help=(
            "Optional locked RGB/GT-only crop manifest used for periodic "
            "crop and structure evaluation."
        ),
    )
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument(
        "--ssr-normalization",
        choices=("batch_norm", "group_norm", "layer_norm"),
        default="batch_norm",
        help=(
            "Normalization inside sparse SSR residual blocks. BatchNorm keeps "
            "legacy checkpoint compatibility; the alternatives are batch-independent."
        ),
    )
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--microbatch-size", type=int, default=1)
    parser.add_argument("--refiner-detach-steps", type=int, default=500)
    parser.add_argument("--backbone-freeze-steps", type=int, default=100)
    parser.add_argument("--backbone-warmup-end", type=int, default=200)
    parser.add_argument("--ssr-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-7)
    parser.add_argument(
        "--learning-rate-schedule",
        choices=("constant", "cosine"),
        default="constant",
    )
    parser.add_argument(
        "--learning-rate-decay-start-step",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--learning-rate-decay-end-step",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--learning-rate-final-scale",
        type=float,
        default=1.0,
        help="Final/peak learning-rate ratio for the cosine schedule.",
    )
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--max-preclip-grad-norm",
        type=float,
        default=0.0,
        help="Abort before optimizer.step when the pre-clip norm exceeds this value; 0 disables.",
    )
    parser.add_argument(
        "--max-skipped-preclip-steps",
        type=int,
        default=0,
        help=(
            "Skip at most this many optimizer updates whose pre-clip gradient "
            "exceeds the configured limit; 0 preserves fail-fast behavior."
        ),
    )
    parser.add_argument(
        "--max-consecutive-skipped-preclip-steps",
        type=int,
        default=0,
        help=(
            "Abort after this many consecutive skipped pre-clip updates; "
            "required when --max-skipped-preclip-steps is positive."
        ),
    )
    parser.add_argument(
        "--max-abs-log-depth-residual",
        type=float,
        default=0.0,
        help="Abort before backward when any SSR log-depth residual exceeds this value; 0 disables.",
    )
    parser.add_argument(
        "--smooth-log-depth-residual-bound",
        type=float,
        default=0.0,
        help=(
            "Apply limit*tanh(raw/limit) to every SSR log-depth update before "
            "geometry update and re-voxelization; 0 disables."
        ),
    )
    parser.add_argument(
        "--max-abs-raw-log-depth-residual",
        type=float,
        default=0.0,
        help=(
            "Emergency abort before backward when the pre-bound SSR residual "
            "exceeds this value; 0 disables. Isolated values below this limit "
            "are constrained by the peak loss rather than treated as collapse."
        ),
    )
    parser.add_argument(
        "--max-raw-log-depth-residual-p999",
        type=float,
        default=0.0,
        help=(
            "Abort before backward when the P99.9 absolute raw residual "
            "exceeds this value; 0 disables."
        ),
    )
    parser.add_argument(
        "--raw-residual-warning-threshold",
        type=float,
        default=0.0,
        help="Record a warning when the maximum raw residual exceeds this value.",
    )
    parser.add_argument(
        "--raw-residual-tail-threshold",
        type=float,
        default=0.0,
        help="Raw residual magnitude below which the tail penalty is exactly zero.",
    )
    parser.add_argument(
        "--raw-residual-tail-weight",
        type=float,
        default=0.0,
        help="Weight of the raw residual tail penalty.",
    )
    parser.add_argument(
        "--raw-residual-peak-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the per-sample, per-cycle raw residual peak penalty."
        ),
    )
    parser.add_argument(
        "--max-bound-saturation-fraction",
        type=float,
        default=0.0,
        help="Abort after a persistent fraction of applied residuals reaches 95%% of the bound.",
    )
    parser.add_argument(
        "--max-consecutive-saturated-steps",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--max-voxel-depth-span",
        type=int,
        default=0,
        help="Reject a shell before SpConv construction when its depth span exceeds this value.",
    )
    parser.add_argument(
        "--max-refined-point-rel",
        type=float,
        default=0.0,
        help="Absolute periodic K-refined point Rel collapse threshold; 0 disables.",
    )
    parser.add_argument(
        "--max-refined-to-base-ratio",
        type=float,
        default=0.0,
        help="Periodic K-refined/K=0 point Rel collapse ratio; 0 disables.",
    )
    parser.add_argument(
        "--max-base-to-best-ratio",
        type=float,
        default=0.0,
        help="Abort when periodic K=0 point Rel exceeds this multiple of its historical best.",
    )
    parser.add_argument("--global-weight", type=float, default=1.0)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--local-scales", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument(
        "--full-eval-every",
        type=int,
        default=0,
        help="Evaluate all training samples and save a milestone every N steps; 0 disables.",
    )
    parser.add_argument("--periodic-train-samples", type=int, default=64)
    parser.add_argument(
        "--selection-split",
        choices=("train", "val"),
        default="val",
        help="Split whose K-refined point Rel selects checkpoint.pt.",
    )
    parser.add_argument(
        "--selection-scope",
        choices=("full", "crop", "structure", "composite"),
        default="full",
        help="Evaluation scope used to select checkpoint.pt; composite is full+structure point Rel.",
    )
    parser.add_argument(
        "--best-checkpoint-include-optimizer",
        action="store_true",
        help="Store optimizer and RNG state in checkpoint.pt for exact stage transition.",
    )
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--loss-smoothing-window", type=int, default=100)
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    parser.add_argument("--max-samples-per-split", type=int)
    parser.add_argument("--seed", type=int, default=61)
    return parser.parse_args()


def validate_joint_schedule(args: argparse.Namespace) -> None:
    if args.steps <= 0 or args.batch_size <= 0 or args.microbatch_size <= 0:
        raise ValueError("Steps and batch sizes must be positive")
    if args.microbatch_size > args.batch_size:
        raise ValueError("Microbatch size cannot exceed effective batch size")
    if not 1 <= args.refinement_steps <= 7:
        raise ValueError("Refinement steps must be in [1, 7]")
    if not (
        0 <= args.backbone_freeze_steps
        < args.backbone_warmup_end
        <= args.refiner_detach_steps
    ):
        raise ValueError(
            "Expected freeze < warmup <= detach; a short run may end before "
            "the detach boundary"
        )
    if (
        args.eval_every <= 0
        or args.full_eval_every < 0
        or args.periodic_train_samples <= 0
    ):
        raise ValueError("Evaluation intervals and sample counts must be positive")
    if args.ssr_learning_rate <= 0:
        raise ValueError("SSR learning rate must be positive")
    if args.train_refiner_only:
        if args.head_learning_rate != 0 or args.backbone_learning_rate != 0:
            raise ValueError(
                "Refiner-only training requires head and backbone learning rates 0"
            )
        if args.selection_scope != "full" and args.fine_structure_rois is None:
            raise ValueError(
                "Crop/structure selection requires --fine-structure-rois"
            )
    elif args.head_learning_rate <= 0:
        raise ValueError("Trainable heads require a positive learning rate")
    if not 0.0 < args.learning_rate_final_scale <= 1.0:
        raise ValueError("Final learning-rate scale must be in (0, 1]")
    if args.learning_rate_schedule == "cosine":
        if not (
            0
            <= args.learning_rate_decay_start_step
            < args.learning_rate_decay_end_step
        ):
            raise ValueError(
                "Cosine decay requires 0 <= start < end"
            )
    if min(
        args.max_preclip_grad_norm,
        args.max_abs_log_depth_residual,
        args.smooth_log_depth_residual_bound,
        args.max_abs_raw_log_depth_residual,
        args.max_raw_log_depth_residual_p999,
        args.raw_residual_warning_threshold,
        args.raw_residual_tail_threshold,
        args.raw_residual_tail_weight,
        args.raw_residual_peak_weight,
        args.max_bound_saturation_fraction,
        args.max_voxel_depth_span,
        args.max_refined_point_rel,
        args.max_refined_to_base_ratio,
        args.max_base_to_best_ratio,
    ) < 0:
        raise ValueError("Instability thresholds cannot be negative")
    if not math.isfinite(args.smooth_log_depth_residual_bound):
        raise ValueError("Smooth residual bound must be finite")
    if min(
        args.max_skipped_preclip_steps,
        args.max_consecutive_skipped_preclip_steps,
        args.max_consecutive_saturated_steps,
    ) < 0:
        raise ValueError("Skipped-step limits cannot be negative")
    if args.max_skipped_preclip_steps == 0:
        if args.max_consecutive_skipped_preclip_steps != 0:
            raise ValueError(
                "Consecutive skipped-step limit requires a positive total limit"
            )
    elif not (
        1
        <= args.max_consecutive_skipped_preclip_steps
        <= args.max_skipped_preclip_steps
    ):
        raise ValueError(
            "Consecutive skipped-step limit must be between 1 and the total limit"
        )
    if args.max_bound_saturation_fraction > 1:
        raise ValueError("Residual saturation fraction must lie in [0, 1]")
    if (args.max_bound_saturation_fraction > 0) != (
        args.max_consecutive_saturated_steps > 0
    ):
        raise ValueError(
            "Saturation fraction and consecutive-step limit must be enabled together"
        )
    if (
        args.selection_scope == "composite"
        and (
            args.selection_split != "train"
            or args.fine_structure_rois is None
            or args.full_eval_every <= 0
        )
    ):
        raise ValueError(
            "Composite selection requires train selection, fine-structure ROIs, "
            "and a positive full-eval interval"
        )
    if args.freeze_backbone or args.train_refiner_only:
        if args.backbone_learning_rate != 0:
            raise ValueError(
                "Frozen backbone requires --backbone-learning-rate 0"
            )
    elif args.backbone_learning_rate <= 0:
        raise ValueError("Trainable backbone learning rate must be positive")


def validate_distributed_batch(
    global_batch_size: int,
    microbatch_size: int,
    world_size: int,
) -> int:
    if world_size <= 0:
        raise ValueError("World size must be positive")
    if global_batch_size % world_size:
        raise ValueError(
            "Global batch size must be divisible by the number of processes"
        )
    local_batch_size = global_batch_size // world_size
    if microbatch_size > local_batch_size:
        raise ValueError(
            "Microbatch size cannot exceed the per-process batch size"
        )
    return local_batch_size


def shard_batch_indices(
    indices: Sequence[int],
    process_index: int,
    world_size: int,
) -> List[int]:
    if not 0 <= process_index < world_size:
        raise ValueError("Invalid distributed process index")
    if len(indices) % world_size:
        raise ValueError("Global batch indices cannot be evenly sharded")
    return list(indices[process_index::world_size])


def broadcast_main_object(value: Any, accelerator: Accelerator) -> Any:
    if accelerator.num_processes == 1:
        return value
    payload = [value if accelerator.is_main_process else None]
    torch.distributed.broadcast_object_list(payload, src=0)
    return payload[0]


def distributed_mean_dict(
    values: Dict[str, float],
    accelerator: Accelerator,
) -> Dict[str, float]:
    keys = sorted(values)
    tensor = torch.tensor(
        [float(values[key]) for key in keys],
        dtype=torch.float64,
        device=accelerator.device,
    )
    reduced = accelerator.reduce(tensor, reduction="mean")
    return {
        key: float(reduced[index].item())
        for index, key in enumerate(keys)
    }


def distributed_max_dict(
    values: Dict[str, float],
    accelerator: Accelerator,
) -> Dict[str, float]:
    keys = sorted(values)
    tensor = torch.tensor(
        [float(values[key]) for key in keys],
        dtype=torch.float64,
        device=accelerator.device,
    )
    reduced = accelerator.reduce(tensor, reduction="max")
    return {
        key: float(reduced[index].item())
        for index, key in enumerate(keys)
    }


def backbone_learning_rate(
    step: int,
    *,
    freeze_steps: int,
    warmup_end: int,
    peak_lr: float,
) -> float:
    if step <= freeze_steps:
        return 0.0
    if step < warmup_end:
        return peak_lr * (step - freeze_steps) / (warmup_end - freeze_steps)
    return peak_lr


def learning_rate_scale(
    step: int,
    *,
    schedule: str,
    decay_start_step: int,
    decay_end_step: int,
    final_scale: float,
) -> float:
    if schedule == "constant":
        return 1.0
    if schedule != "cosine":
        raise ValueError(f"Unsupported learning-rate schedule: {schedule}")
    if step <= decay_start_step:
        return 1.0
    if step >= decay_end_step:
        return final_scale
    progress = (
        (step - decay_start_step)
        / (decay_end_step - decay_start_step)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return final_scale + (1.0 - final_scale) * cosine


def training_stage(step: int, refiner_detach_steps: int) -> str:
    return "detached_warmup" if step <= refiner_detach_steps else "joint"


def effective_training_stage(
    step: int,
    refiner_detach_steps: int,
    *,
    train_refiner_only: bool,
) -> str:
    if train_refiner_only:
        return "refiner_only"
    return training_stage(step, refiner_detach_steps)


def refinement_has_collapsed(
    *,
    base_point_rel: float,
    refined_point_rel: float,
    absolute_threshold: float,
    ratio_threshold: float,
) -> bool:
    checks = []
    if absolute_threshold > 0:
        checks.append(refined_point_rel > absolute_threshold)
    if ratio_threshold > 0:
        checks.append(
            refined_point_rel
            > ratio_threshold * max(base_point_rel, 1e-12)
        )
    return bool(checks and all(checks))


def set_optimizer_learning_rates(
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    scale = learning_rate_scale(
        step,
        schedule=args.learning_rate_schedule,
        decay_start_step=args.learning_rate_decay_start_step,
        decay_end_step=args.learning_rate_decay_end_step,
        final_scale=args.learning_rate_final_scale,
    )
    rates = {
        "ssr": scale * args.ssr_learning_rate,
        "heads": scale * args.head_learning_rate,
        "backbone": (
            0.0
            if args.freeze_backbone or args.train_refiner_only
            else scale
            * backbone_learning_rate(
                step,
                freeze_steps=args.backbone_freeze_steps,
                warmup_end=args.backbone_warmup_end,
                peak_lr=args.backbone_learning_rate,
            )
        ),
    }
    for group in optimizer.param_groups:
        group["lr"] = rates[str(group["name"])]
    return rates


def set_backbone_trainable(model: MoGeModel, trainable: bool) -> None:
    for parameter in model.encoder.backbone.parameters():
        parameter.requires_grad_(trainable)


def set_refiner_only_trainable(model: MoGeModel) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("ssr."))


def base_parameters_are_frozen(model: MoGeModel) -> bool:
    return all(
        (name.startswith("ssr.") or not parameter.requires_grad)
        for name, parameter in model.named_parameters()
    )


def clear_backbone_gradients(model: MoGeModel) -> None:
    """Keep a zero-LR DDP backbone out of global gradient clipping."""
    for parameter in model.encoder.backbone.parameters():
        parameter.grad = None


def limited_manifest(
    manifest: Dict[str, object],
    maximum_per_split: int | None,
) -> Dict[str, object]:
    if maximum_per_split is None:
        return manifest
    if maximum_per_split <= 0:
        raise ValueError("--max-samples-per-split must be positive")
    samples = []
    for split in ("train", "val", "test"):
        candidates = [
            sample for sample in manifest["samples"] if sample["split"] == split
        ]
        samples.extend(candidates[:maximum_per_split])
    return {
        **manifest,
        "samples": samples,
        "counts": {
            split: sum(sample["split"] == split for sample in samples)
            for split in ("train", "val", "test")
        },
    }


def load_raw_samples(
    data: Path,
    manifest: Dict[str, object],
    *,
    height: int,
    width: int,
    safe_root: Path,
    splits: Sequence[str],
    fine_structure_rois: Dict[str, Dict[str, object]] | None = None,
) -> List[RawSample]:
    selected = []
    for metadata in manifest["samples"]:
        if metadata["split"] not in splits:
            continue
        image, gt_points = load_raw_sample(
            data,
            metadata,
            height,
            width,
            safe_root,
        )
        roi = (
            fine_structure_rois.get(str(metadata["id"]))
            if fine_structure_rois is not None
            else None
        )
        selected.append(
            RawSample(
                sample_id=str(metadata["id"]),
                split=str(metadata["split"]),
                scene=str(metadata["scene"]),
                frame=int(metadata["frame"]),
                image=image,
                gt_points=gt_points,
                crop_xyxy=(
                    tuple(int(value) for value in roi["crop_xyxy"])
                    if roi is not None
                    else None
                ),
                structure_mask_config=(
                    dict(roi["structure_mask"])
                    if roi is not None
                    else None
                ),
            )
        )
    return selected


def load_fine_structure_rois(
    path: Path | None,
    *,
    safe_root: Path,
    height: int,
    width: int,
) -> Dict[str, Dict[str, object]] | None:
    if path is None:
        return None
    source = assert_safe_path(path, safe_root=safe_root, must_exist=True)
    payload = json.loads(source.read_text(encoding="utf-8"))
    if payload.get("selection_inputs") != "RGB and ground truth only; no predictions":
        raise ValueError("Fine-structure ROI provenance must exclude predictions")
    if list(payload.get("shape", [])) != [height, width]:
        raise ValueError("Fine-structure ROI shape does not match training shape")
    entries: Dict[str, Dict[str, object]] = {}
    for entry in payload.get("entries", []):
        sample_id = str(entry["id"])
        if sample_id in entries:
            raise ValueError(f"Duplicate fine-structure ROI: {sample_id}")
        crop = tuple(int(value) for value in entry["crop_xyxy"])
        x0, y0, x1, y1 = crop
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"Invalid fine-structure crop for {sample_id}: {crop}")
        structure_mask = entry.get("structure_mask")
        if not isinstance(structure_mask, dict):
            raise ValueError(f"Missing structure mask for {sample_id}")
        entries[sample_id] = {
            **entry,
            "crop_xyxy": list(crop),
            "structure_mask": dict(structure_mask),
        }
    if not entries:
        raise ValueError("Fine-structure ROI manifest contains no entries")
    return entries


@torch.no_grad()
def evaluate_model(
    model: MoGeModel,
    samples: Sequence[RawSample],
    *,
    device: torch.device,
    num_tokens: int,
    refinement_steps: Sequence[int],
    batch_size: int,
    boundary_threshold: float,
    smooth_log_depth_residual_bound: float = 0.0,
    max_voxel_depth_span: int = 0,
    ssr_batch_norm_states: List[
        Dict[str, Dict[str, torch.Tensor]]
    ]
    | None = None,
) -> Dict[int, List[Dict[str, object]]]:
    from moge.scripts.overfit_hypersim_staged_v3 import (
        _scope_metrics,
        build_structure_mask,
    )

    requested = sorted(set(int(step) for step in refinement_steps))
    if not requested or requested[0] < 0 or requested[-1] > 7:
        raise ValueError("Evaluation refinement steps must be in [0, 7]")
    was_training = model.training
    model.eval()
    records = {step: [] for step in requested}
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        images = torch.stack([sample.image for sample in chunk]).to(device)
        gt = torch.stack([sample.gt_points for sample in chunk]).to(device)
        output = model(
            images,
            num_tokens=num_tokens,
            num_refinement_steps=requested[-1],
            return_intermediates=True,
            smooth_log_depth_residual_bound=smooth_log_depth_residual_bound,
            max_voxel_depth_span=(
                max_voxel_depth_span if max_voxel_depth_span > 0 else None
            ),
            ssr_batch_norm_states=ssr_batch_norm_states,
        )
        sequence = output["points_sequence"]
        for refinement_step in requested:
            prediction = sequence[refinement_step]
            for index, sample in enumerate(chunk):
                metrics, aligned_batch = aligned_metrics(
                    prediction[index : index + 1],
                    gt[index : index + 1],
                )
                aligned = aligned_batch[0]
                valid = torch.isfinite(gt[index]).all(dim=-1)
                metrics["boundary_f1"] = boundary_f1(
                    aligned[..., 2],
                    gt[index, ..., 2],
                    valid,
                    boundary_threshold,
                )
                if sample.crop_xyxy is not None:
                    x0, y0, x1, y1 = sample.crop_xyxy
                    crop_gt = gt[index, y0:y1, x0:x1]
                    crop_aligned = aligned[y0:y1, x0:x1]
                    crop_valid = (
                        torch.isfinite(crop_gt).all(dim=-1)
                        & (crop_gt[..., 2] > 0)
                    )
                    structure_mask = build_structure_mask(
                        gt[index],
                        sample.crop_xyxy,
                        sample.structure_mask_config
                        or {"type": "near_quantile", "quantile": 0.5},
                    )
                    for scope, scope_metrics in (
                        (
                            "crop",
                            _scope_metrics(
                                crop_aligned,
                                crop_gt,
                                crop_valid,
                                boundary_threshold=boundary_threshold,
                            ),
                        ),
                        (
                            "structure",
                            _scope_metrics(
                                crop_aligned,
                                crop_gt,
                                structure_mask,
                                boundary_threshold=boundary_threshold,
                            ),
                        ),
                    ):
                        metrics.update(
                            {
                                f"{scope}_{key}": value
                                for key, value in scope_metrics.items()
                                if key != "pixels"
                            }
                        )
                records[refinement_step].append(
                    {
                        "id": sample.sample_id,
                        "split": sample.split,
                        "scene": sample.scene,
                        "frame": sample.frame,
                        **metrics,
                    }
                )
        del images, gt, output, sequence
    if was_training:
        model.train()
    return records


def aggregate_evaluation(
    records: Dict[int, List[Dict[str, object]]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    aggregated: Dict[str, Dict[str, Dict[str, float]]] = {}
    for step, step_records in records.items():
        step_result: Dict[str, Dict[str, float]] = {}
        for split in sorted({str(record["split"]) for record in step_records}):
            split_records = [
                record for record in step_records if record["split"] == split
            ]
            values = aggregate_metrics(split_records)
            for scope in ("crop", "structure"):
                for metric in METRIC_KEYS:
                    key = f"{scope}_{metric}"
                    scoped = [
                        float(record[key])
                        for record in split_records
                        if key in record
                    ]
                    if scoped:
                        values[key] = float(np.mean(scoped))
            step_result[split] = values
        aggregated[str(step)] = step_result
    return aggregated


def selection_metric_key(scope: str) -> str:
    if scope == "full":
        return "point_rel"
    if scope not in {"crop", "structure"}:
        raise ValueError(f"Unsupported selection scope: {scope}")
    return f"{scope}_point_rel"


def periodic_selection_score(
    periodic: Dict[str, Dict[str, Dict[str, float]]],
    *,
    refinement_step: int,
    split: str,
    scope: str,
) -> float:
    return selection_score(
        periodic,
        refinement_step=refinement_step,
        split=split,
        scope=scope,
    )


def flatten_periodic_evaluation(
    step: int,
    periodic: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, object]:
    record: Dict[str, object] = {"step": step}
    for refinement_step, step_values in periodic.items():
        for split, metrics in step_values.items():
            for metric, value in metrics.items():
                record[f"{split}/k{refinement_step}_{metric}"] = value
    return record


def save_csv(path: Path, records: Iterable[Dict[str, object]]) -> None:
    rows = list(records)
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


def load_numeric_csv(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as file:
        for source in csv.DictReader(file):
            rows.append(
                {
                    key: (
                        source[key]
                        if key in {"stage"}
                        else int(source[key])
                        if key == "step"
                        else float(source[key])
                    )
                    for key in source
                }
            )
    return rows


def validate_resume_histories(
    *,
    start_step: int,
    training_history: Sequence[Dict[str, object]],
    evaluation_history: Sequence[Dict[str, object]],
) -> None:
    history_start_step = (
        int(evaluation_history[0]["step"]) if evaluation_history else 0
    )
    expected_length = start_step - history_start_step
    if expected_length < 0 or len(training_history) != expected_length:
        raise ValueError(
            "Training history does not match resume step and history origin"
        )


def atomic_torch_save(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.incomplete")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def atomic_json_save(payload: Dict[str, object], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def write_instability_event(
    output: Path,
    payload: Dict[str, object],
) -> None:
    temporary = output / "instability_event.json.incomplete"
    destination = output / "instability_event.json"
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def append_stability_event(
    output: Path,
    payload: Dict[str, object],
) -> None:
    with (output / "stability_events.jsonl").open(
        "a",
        encoding="utf-8",
    ) as file:
        file.write(json.dumps(payload, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())


def load_stability_events(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    events: List[Dict[str, object]] = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def skipped_preclip_state(
    events: Sequence[Dict[str, object]],
    *,
    start_step: int,
) -> tuple[int, int]:
    skipped_steps = sorted(
        int(event["step"])
        for event in events
        if event.get("event") == "preclip_gradient_limit_skipped"
        and int(event["step"]) <= start_step
    )
    total = len(skipped_steps)
    consecutive = 0
    expected_step = start_step
    for step in reversed(skipped_steps):
        if step != expected_step:
            break
        consecutive += 1
        expected_step -= 1
    return total, consecutive


def preclip_gradient_action(
    *,
    grad_norm: float,
    threshold: float,
    skipped_total: int,
    skipped_consecutive: int,
    max_skipped_total: int,
    max_skipped_consecutive: int,
) -> str:
    if threshold <= 0 or grad_norm <= threshold:
        return "apply"
    if (
        skipped_total < max_skipped_total
        and skipped_consecutive < max_skipped_consecutive
    ):
        return "skip"
    return "abort"


def skipped_gradient_budget_exhausted(
    *,
    skipped_total: int,
    skipped_consecutive: int,
    max_skipped_total: int,
    max_skipped_consecutive: int,
) -> bool:
    return (
        max_skipped_total > 0
        and (
            skipped_total >= max_skipped_total
            or skipped_consecutive >= max_skipped_consecutive
        )
    )


def checkpoint_payload(
    *,
    model: MoGeModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_step: int,
    best_score: float,
    args: argparse.Namespace,
    cpu_generator: torch.Generator,
    loss_generator: torch.Generator,
    initial_periodic: Dict[str, Dict[str, Dict[str, float]]],
    include_optimizer: bool,
    best_base_score: float | None = None,
    initial_full_evaluation: Dict[str, Dict[str, Dict[str, float]]] | None = None,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model": model.state_dict(),
        "step": step,
        "best_step": best_step,
        "pretrained": args.pretrained,
        "args": vars(args),
        "selection": {
            "metric": (
                f"{args.selection_scope} {args.selection_split} "
                f"K={args.refinement_steps} point_rel"
            ),
            "mode": "min",
            "score": best_score,
        },
        "cpu_generator_state": cpu_generator.get_state(),
        "loss_generator_state": loss_generator.get_state(),
        "initial_periodic": initial_periodic,
        "initial_full_evaluation": initial_full_evaluation,
        "best_base_point_rel": best_base_score,
        "training_mode": (
            "fixed-base SSR-only fine-tuning"
            if args.train_refiner_only
            else "two-stage joint fine-tuning"
        ),
        "normal_prediction": False,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def save_training_plot(
    output: Path,
    training_history: Sequence[Dict[str, object]],
    evaluation_history: Sequence[Dict[str, object]],
    *,
    smoothing_window: int,
    detach_step: int,
    refinement_step: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    losses = np.asarray(
        [float(record["loss"]) for record in training_history],
        dtype=np.float64,
    )
    steps = np.asarray(
        [int(record["step"]) for record in training_history],
        dtype=np.int64,
    )
    smooth = moving_average(
        losses,
        min(smoothing_window, max(1, len(losses))),
    )
    eval_steps = [int(record["step"]) for record in evaluation_history]
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 1, figsize=(11, 12))
    axes[0].plot(steps, losses, alpha=0.16, linewidth=0.7, color="#60A5FA")
    axes[0].plot(steps, smooth, linewidth=2.0, color="#1D4ED8")
    axes[0].axvline(detach_step, linestyle="--", color="#7C3AED")
    axes[0].set_xlabel("Optimization step")
    axes[0].set_ylabel("Geometry loss")
    axes[0].set_title("Two-stage joint fine-tuning loss")

    for split, color in (("train", "#2563EB"), ("val", "#DC2626")):
        for plotted_step, style in ((0, "--"), (refinement_step, "-")):
            axes[1].plot(
                eval_steps,
                [
                    float(record[f"{split}/k{plotted_step}_point_rel"])
                    for record in evaluation_history
                ],
                color=color,
                linestyle=style,
                marker="o",
                label=f"{split} K={plotted_step}",
            )
            axes[2].plot(
                eval_steps,
                [
                    float(record[f"{split}/k{plotted_step}_boundary_f1"])
                    for record in evaluation_history
                ],
                color=color,
                linestyle=style,
                marker="o",
                label=f"{split} K={plotted_step}",
            )
    for axis in axes[1:]:
        axis.axvline(detach_step, linestyle="--", color="#7C3AED")
        axis.set_xlabel("Optimization step")
        axis.legend(ncol=2, fontsize=8)
    axes[1].set_ylabel("Aligned point Rel")
    axes[1].set_title("Periodic point-map validation")
    axes[2].set_ylabel("Depth-boundary F1")
    axes[2].set_title("Periodic boundary validation")
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=180)
    figure.savefig(output / "training_curves.pdf")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    validate_joint_schedule(args)
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=[
            InitProcessGroupKwargs(backend=args.ddp_backend),
            DistributedDataParallelKwargs(find_unused_parameters=True),
        ],
    )
    local_batch_size = validate_distributed_batch(
        args.batch_size,
        args.microbatch_size,
        accelerator.num_processes,
    )
    data = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    if accelerator.is_main_process:
        output.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    manifest_path = assert_safe_path(
        data / "manifest.json",
        safe_root=args.safe_root,
        must_exist=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "moge3-hypersim-generalization-v1":
        raise ValueError(f"Unsupported manifest: {manifest.get('format')}")
    manifest = limited_manifest(manifest, args.max_samples_per_split)
    fine_structure_rois = load_fine_structure_rois(
        args.fine_structure_rois,
        safe_root=args.safe_root,
        height=args.height,
        width=args.width,
    )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = accelerator.device
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats(device)

    run_start = time.perf_counter()
    samples = load_raw_samples(
        data,
        manifest,
        height=args.height,
        width=args.width,
        safe_root=args.safe_root,
        splits=("train", "val"),
        fine_structure_rois=fine_structure_rois,
    )
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    if not train_samples or not val_samples:
        raise ValueError("Both train and validation samples are required")
    periodic_train = stratified_subset(
        train_samples,
        min(args.periodic_train_samples, len(train_samples)),
    )
    periodic_samples = periodic_train + val_samples

    with accelerator.main_process_first():
        model = MoGeModel.from_pretrained(
            args.pretrained,
            model_kwargs={"ssr": {"normalization": args.ssr_normalization}},
        )
    model.enable_gradient_checkpointing()
    model.train()
    schedule = TrainingScheduleV3(
        refiner_detach_steps=args.refiner_detach_steps,
        backbone_freeze_steps=args.backbone_freeze_steps,
        backbone_warmup_end=args.backbone_warmup_end,
        ssr_lr=args.ssr_learning_rate,
        head_lr=args.head_learning_rate,
        backbone_lr=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
    )
    optimizer = build_v3_optimizer(model, schedule)
    cpu_generator = torch.Generator().manual_seed(args.seed + 1)
    loss_generator = torch.Generator(device=device).manual_seed(args.seed + 2)

    start_step = 0
    best_step = 0
    resume_path = None
    warm_start_path = None
    transition_path = None
    resume_checkpoint = None
    warm_start_checkpoint = None
    transition_checkpoint = None
    if args.resume is not None:
        resume_path = assert_safe_path(
            args.resume,
            safe_root=args.safe_root,
            must_exist=True,
        )
        resume_checkpoint = torch.load(
            resume_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(resume_checkpoint["model"], strict=True)
        optimizer.load_state_dict(resume_checkpoint["optimizer"])
        cpu_generator.set_state(resume_checkpoint["cpu_generator_state"])
        start_step = int(resume_checkpoint["step"])
        best_step = int(resume_checkpoint["best_step"])
        best_score = float(resume_checkpoint["selection"]["score"])
        stored_best_base = resume_checkpoint.get("best_base_point_rel")
        best_base_score = (
            float(stored_best_base)
            if stored_best_base is not None
            else math.inf
        )
        initial_periodic = resume_checkpoint["initial_periodic"]
        initial_full_evaluation = resume_checkpoint.get(
            "initial_full_evaluation"
        )
        if args.steps <= start_step:
            raise ValueError("--steps must exceed the resumed checkpoint step")
        training_history = load_numeric_csv(output / "training_history.csv")
        evaluation_history = load_numeric_csv(output / "evaluation_history.csv")
        full_evaluation_history = load_numeric_csv(
            output / "full_evaluation_history.csv"
        )
        validate_resume_histories(
            start_step=start_step,
            training_history=training_history,
            evaluation_history=evaluation_history,
        )
    elif args.warm_start is not None:
        warm_start_path = assert_safe_path(
            args.warm_start,
            safe_root=args.safe_root,
            must_exist=True,
        )
        warm_start_checkpoint = torch.load(
            warm_start_path,
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(warm_start_checkpoint["model"], strict=True)
        start_step = int(warm_start_checkpoint["step"])
        if args.steps <= start_step:
            raise ValueError("--steps must exceed the warm-start checkpoint step")
    elif args.transition_from is not None:
        transition_path = assert_safe_path(
            args.transition_from,
            safe_root=args.safe_root,
            must_exist=True,
        )
        transition_checkpoint = torch.load(
            transition_path,
            map_location="cpu",
            weights_only=False,
        )
        if "optimizer" not in transition_checkpoint:
            raise ValueError(
                "Stage transition requires a checkpoint containing optimizer state"
            )
        model.load_state_dict(transition_checkpoint["model"], strict=True)
        optimizer.load_state_dict(transition_checkpoint["optimizer"])
        cpu_generator.set_state(transition_checkpoint["cpu_generator_state"])
        start_step = int(transition_checkpoint["step"])
        if args.steps <= start_step:
            raise ValueError("--steps must exceed the transition checkpoint step")

    if args.train_refiner_only:
        set_refiner_only_trainable(model)
        if not base_parameters_are_frozen(model):
            raise RuntimeError("Base parameters were not fully frozen")
    elif args.freeze_backbone:
        set_backbone_trainable(model, False)
    model, optimizer = accelerator.prepare(model, optimizer)
    raw_model = accelerator.unwrap_model(model)
    state_checkpoint = resume_checkpoint or transition_checkpoint
    if state_checkpoint is not None:
        loss_generator.set_state(state_checkpoint["loss_generator_state"])

    if resume_checkpoint is None:
        initial_periodic = None
        if accelerator.is_main_process:
            initial_records = evaluate_model(
                raw_model,
                periodic_samples,
                device=device,
                num_tokens=args.num_tokens,
                refinement_steps=(0, args.refinement_steps),
                batch_size=args.eval_batch_size,
                boundary_threshold=args.boundary_threshold,
                smooth_log_depth_residual_bound=(
                    args.smooth_log_depth_residual_bound
                ),
                max_voxel_depth_span=args.max_voxel_depth_span,
            )
            initial_periodic = aggregate_evaluation(initial_records)
        initial_periodic = broadcast_main_object(initial_periodic, accelerator)
        assert initial_periodic is not None
        initial_full_evaluation = None
        if accelerator.is_main_process and args.full_eval_every > 0:
            initial_full_records = evaluate_model(
                raw_model,
                train_samples,
                device=device,
                num_tokens=args.num_tokens,
                refinement_steps=(0, args.refinement_steps),
                batch_size=args.eval_batch_size,
                boundary_threshold=args.boundary_threshold,
                smooth_log_depth_residual_bound=(
                    args.smooth_log_depth_residual_bound
                ),
                max_voxel_depth_span=args.max_voxel_depth_span,
            )
            initial_full_evaluation = aggregate_evaluation(
                initial_full_records
            )
        initial_full_evaluation = broadcast_main_object(
            initial_full_evaluation,
            accelerator,
        )
        selection_evaluation = (
            initial_full_evaluation
            if args.selection_scope == "composite"
            else initial_periodic
        )
        if selection_evaluation is None:
            raise RuntimeError("Missing initial selection evaluation")
        best_score = periodic_selection_score(
            selection_evaluation,
            refinement_step=args.refinement_steps,
            split=args.selection_split,
            scope=args.selection_scope,
        )
        best_base_score = float(
            initial_periodic["0"][args.selection_split]["point_rel"]
        )
        best_step = start_step
        training_history = []
        evaluation_history = [
            flatten_periodic_evaluation(start_step, initial_periodic)
        ]
        full_evaluation_history = (
            [
                flatten_periodic_evaluation(
                    start_step,
                    initial_full_evaluation,
                )
            ]
            if initial_full_evaluation is not None
            else []
        )
        if accelerator.is_main_process:
            initial_payload = checkpoint_payload(
                model=raw_model,
                optimizer=optimizer,
                step=start_step,
                best_step=best_step,
                best_score=best_score,
                args=args,
                cpu_generator=cpu_generator,
                loss_generator=loss_generator,
                initial_periodic=initial_periodic,
                include_optimizer=args.best_checkpoint_include_optimizer,
                best_base_score=best_base_score,
                initial_full_evaluation=initial_full_evaluation,
            )
            atomic_torch_save(initial_payload, output / "checkpoint.pt")
            atomic_torch_save(
                initial_payload,
                output / "initial_checkpoint.pt",
            )
            atomic_json_save(
                {
                    "step": best_step,
                    "score": best_score,
                    "selection_scope": args.selection_scope,
                    "selection_split": args.selection_split,
                    "contains_optimizer": (
                        args.best_checkpoint_include_optimizer
                    ),
                },
                output / "checkpoint_metadata.json",
            )
            atomic_torch_save(
                checkpoint_payload(
                    model=raw_model,
                    optimizer=optimizer,
                    step=start_step,
                    best_step=best_step,
                    best_score=best_score,
                    args=args,
                    cpu_generator=cpu_generator,
                    loss_generator=loss_generator,
                    initial_periodic=initial_periodic,
                    include_optimizer=True,
                    best_base_score=best_base_score,
                    initial_full_evaluation=initial_full_evaluation,
                ),
                output / "resume_checkpoint.pt",
            )
            save_csv(output / "evaluation_history.csv", evaluation_history)
            save_csv(
                output / "full_evaluation_history.csv",
                full_evaluation_history,
            )
        accelerator.wait_for_everyone()

    stability_events = load_stability_events(output / "stability_events.jsonl")
    skipped_preclip_total, skipped_preclip_consecutive = skipped_preclip_state(
        stability_events,
        start_step=start_step,
    )
    saturation_streak = 0
    for historical_record in reversed(training_history):
        if (
            float(
                historical_record.get(
                    "ssr_max_bound_saturation_fraction",
                    0.0,
                )
            )
            <= args.max_bound_saturation_fraction
        ):
            break
        saturation_streak += 1
    raw_warning_active = bool(
        training_history
        and int(training_history[-1].get("ssr_raw_residual_warning", 0))
    )
    train_start = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        step_start = time.perf_counter()
        if args.train_refiner_only:
            set_refiner_only_trainable(raw_model)
        elif args.freeze_backbone:
            set_backbone_trainable(raw_model, False)
        elif accelerator.num_processes == 1:
            set_backbone_trainable(
                raw_model,
                step > args.backbone_freeze_steps,
            )
        learning_rates = set_optimizer_learning_rates(optimizer, step, args)
        stage = effective_training_stage(
            step,
            args.refiner_detach_steps,
            train_refiner_only=args.train_refiner_only,
        )
        global_indices = torch.randint(
            len(train_samples),
            (args.batch_size,),
            generator=cpu_generator,
        ).tolist()
        indices = shard_batch_indices(
            global_indices,
            accelerator.process_index,
            accelerator.num_processes,
        )
        if len(indices) != local_batch_size:
            raise RuntimeError("Distributed batch shard has the wrong size")
        batch = [train_samples[index] for index in indices]

        optimizer.zero_grad(set_to_none=True)
        local_loss_value = 0.0
        local_terms: Dict[str, float] = defaultdict(float)
        step_maxima = {
            "ssr_max_abs_log_depth_residual": 0.0,
            "ssr_max_abs_raw_log_depth_residual": 0.0,
            "ssr_max_bound_saturation_fraction": 0.0,
            "ssr_max_depth_span": 0.0,
            "ssr_raw_p95": 0.0,
            "ssr_raw_p99": 0.0,
            "ssr_raw_p999": 0.0,
        }
        for microbatch_start in range(
            0,
            len(batch),
            args.microbatch_size,
        ):
            microbatch = batch[
                microbatch_start : microbatch_start + args.microbatch_size
            ]
            weight = len(microbatch) / len(batch)
            images = torch.stack([sample.image for sample in microbatch]).to(device)
            gt = torch.stack([sample.gt_points for sample in microbatch]).to(device)
            output_dict = model(
                images,
                num_tokens=args.num_tokens,
                num_refinement_steps=args.refinement_steps,
                return_intermediates=True,
                detach_base_from_refiner=(
                    args.train_refiner_only
                    or step <= args.refiner_detach_steps
                ),
                smooth_log_depth_residual_bound=(
                    args.smooth_log_depth_residual_bound
                ),
                max_voxel_depth_span=(
                    args.max_voxel_depth_span
                    if args.max_voxel_depth_span > 0
                    else None
                ),
            )
            raw_residuals = output_dict["raw_log_depth_residuals"]
            saturation_fraction = maximum_saturation_fraction(
                output_dict["log_depth_residuals"],
                bound=args.smooth_log_depth_residual_bound,
            )
            percentile_values = raw_residual_percentiles(raw_residuals)
            microbatch_maxima = distributed_max_dict(
                {
                    "ssr_max_abs_log_depth_residual": max(
                        float(residual.detach().abs().amax())
                        for residual in output_dict["log_depth_residuals"]
                    ),
                    "ssr_max_abs_raw_log_depth_residual": max(
                        float(residual.detach().abs().amax())
                        for residual in raw_residuals
                    ),
                    "ssr_max_bound_saturation_fraction": saturation_fraction,
                    "ssr_max_depth_span": max(
                        float(stats["depth_span"].detach().amax())
                        for stats in output_dict["voxel_stats"]
                    ),
                    **percentile_values,
                },
                accelerator,
            )
            for key, value in microbatch_maxima.items():
                step_maxima[key] = max(step_maxima[key], value)
            if (
                args.max_abs_log_depth_residual > 0
                and microbatch_maxima["ssr_max_abs_log_depth_residual"]
                > args.max_abs_log_depth_residual
            ):
                if accelerator.is_main_process:
                    write_instability_event(
                        output,
                        {
                            "event": "log_depth_residual_limit",
                            "step": step,
                            "stage": stage,
                            "threshold": args.max_abs_log_depth_residual,
                            **microbatch_maxima,
                            "action": "aborted before backward and optimizer step",
                        },
                    )
                raise RuntimeError(
                    "SSR log-depth residual exceeded the configured safety limit"
                )
            raw_abort_reason = raw_residual_abort_reason(
                maximum=microbatch_maxima[
                    "ssr_max_abs_raw_log_depth_residual"
                ],
                emergency_limit=args.max_abs_raw_log_depth_residual,
                p999=microbatch_maxima["ssr_raw_p999"],
                p999_limit=args.max_raw_log_depth_residual_p999,
            )
            if raw_abort_reason is not None:
                threshold = (
                    args.max_abs_raw_log_depth_residual
                    if raw_abort_reason
                    == "raw_log_depth_residual_emergency_limit"
                    else args.max_raw_log_depth_residual_p999
                )
                if accelerator.is_main_process:
                    write_instability_event(
                        output,
                        {
                            "event": raw_abort_reason,
                            "step": step,
                            "stage": stage,
                            "threshold": threshold,
                            **microbatch_maxima,
                            "action": (
                                "aborted before backward and optimizer step"
                            ),
                        },
                    )
                raise RuntimeError(
                    "Raw SSR log-depth residual distribution exceeded the "
                    "configured safety limit"
                )
            sequence = output_dict["points_sequence"]
            base_loss, base_terms = geometry_loss(
                sequence[:1],
                gt,
                global_weight=args.global_weight,
                local_weight=args.local_weight,
                edge_weight=args.edge_weight,
                local_scales=tuple(args.local_scales),
                generator=loss_generator,
            )
            refined_loss, refined_terms = geometry_loss(
                sequence[1:],
                gt,
                global_weight=args.global_weight,
                local_weight=args.local_weight,
                edge_weight=args.edge_weight,
                local_scales=tuple(args.local_scales),
                generator=loss_generator,
            )
            tail_loss = raw_residual_tail_loss(
                raw_residuals,
                threshold=args.raw_residual_tail_threshold,
            )
            peak_loss = raw_residual_peak_loss(
                raw_residuals,
                threshold=args.raw_residual_tail_threshold,
            )
            geometry_total = (
                refined_loss
                if args.train_refiner_only
                else base_loss + refined_loss
            )
            loss = (
                geometry_total
                + args.raw_residual_tail_weight * tail_loss
                + args.raw_residual_peak_weight * peak_loss
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at step {step}")
            local_loss_value += weight * float(loss.detach())
            for prefix, terms in (("base", base_terms), ("refined", refined_terms)):
                for key, value in terms.items():
                    local_terms[f"{prefix}/{key}"] += weight * value
            local_terms["stability/raw_tail_loss"] += (
                weight * float(tail_loss.detach())
            )
            local_terms["stability/raw_peak_loss"] += (
                weight * float(peak_loss.detach())
            )
            accelerator.backward(weight * loss)
            del (
                images,
                gt,
                output_dict,
                sequence,
                base_loss,
                refined_loss,
                geometry_total,
                tail_loss,
                peak_loss,
                loss,
            )

        warning_now = bool(
            args.raw_residual_warning_threshold > 0
            and step_maxima["ssr_max_abs_raw_log_depth_residual"]
            > args.raw_residual_warning_threshold
        )
        if warning_now and not raw_warning_active and accelerator.is_main_process:
            append_stability_event(
                output,
                {
                    "event": "raw_log_depth_residual_warning",
                    "step": step,
                    "stage": stage,
                    "threshold": args.raw_residual_warning_threshold,
                    **step_maxima,
                    "action": "warning_only",
                },
            )
        raw_warning_active = warning_now
        saturation_streak = updated_threshold_streak(
            saturation_streak,
            value=step_maxima["ssr_max_bound_saturation_fraction"],
            threshold=args.max_bound_saturation_fraction,
        )
        if (
            args.max_consecutive_saturated_steps > 0
            and saturation_streak >= args.max_consecutive_saturated_steps
        ):
            if accelerator.is_main_process:
                write_instability_event(
                    output,
                    {
                        "event": "residual_saturation_streak",
                        "step": step,
                        "stage": stage,
                        "threshold": args.max_bound_saturation_fraction,
                        "consecutive_steps": saturation_streak,
                        **step_maxima,
                        "action": "aborted before optimizer step",
                    },
                )
            raise RuntimeError(
                "SSR residual saturation persisted beyond the configured limit"
            )

        if learning_rates["backbone"] == 0.0:
            clear_backbone_gradients(raw_model)
        grad_norm = accelerator.clip_grad_norm_(
            (
                parameter
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            args.gradient_clip_norm,
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        gradient_action = preclip_gradient_action(
            grad_norm=float(grad_norm),
            threshold=args.max_preclip_grad_norm,
            skipped_total=skipped_preclip_total,
            skipped_consecutive=skipped_preclip_consecutive,
            max_skipped_total=args.max_skipped_preclip_steps,
            max_skipped_consecutive=args.max_consecutive_skipped_preclip_steps,
        )
        optimizer_step_skipped = gradient_action == "skip"
        if gradient_action == "abort":
            if accelerator.is_main_process:
                write_instability_event(
                    output,
                    {
                        "event": "preclip_gradient_limit",
                        "step": step,
                        "stage": stage,
                        "threshold": args.max_preclip_grad_norm,
                        "grad_norm": float(grad_norm),
                        "skipped_preclip_total": skipped_preclip_total,
                        "skipped_preclip_consecutive": (
                            skipped_preclip_consecutive
                        ),
                        "max_skipped_preclip_steps": (
                            args.max_skipped_preclip_steps
                        ),
                        "max_consecutive_skipped_preclip_steps": (
                            args.max_consecutive_skipped_preclip_steps
                        ),
                        **step_maxima,
                        "action": "aborted before optimizer step",
                    },
                )
            raise RuntimeError(
                "Pre-clip gradient norm exceeded the configured safety limit"
            )
        if optimizer_step_skipped:
            skipped_preclip_total += 1
            skipped_preclip_consecutive += 1
            if accelerator.is_main_process:
                append_stability_event(
                    output,
                    {
                        "event": "preclip_gradient_limit_skipped",
                        "step": step,
                        "stage": stage,
                        "threshold": args.max_preclip_grad_norm,
                        "grad_norm": float(grad_norm),
                        "skipped_preclip_total": skipped_preclip_total,
                        "skipped_preclip_consecutive": (
                            skipped_preclip_consecutive
                        ),
                        **step_maxima,
                        "action": "optimizer step skipped after gradient clipping",
                    },
                )
            optimizer.zero_grad(set_to_none=True)
            if skipped_gradient_budget_exhausted(
                skipped_total=skipped_preclip_total,
                skipped_consecutive=skipped_preclip_consecutive,
                max_skipped_total=args.max_skipped_preclip_steps,
                max_skipped_consecutive=(
                    args.max_consecutive_skipped_preclip_steps
                ),
            ):
                if accelerator.is_main_process:
                    write_instability_event(
                        output,
                        {
                            "event": "preclip_gradient_skip_budget_exhausted",
                            "step": step,
                            "stage": stage,
                            "threshold": args.max_preclip_grad_norm,
                            "grad_norm": float(grad_norm),
                            "skipped_preclip_total": skipped_preclip_total,
                            "skipped_preclip_consecutive": (
                                skipped_preclip_consecutive
                            ),
                            **step_maxima,
                            "action": "aborted after the configured skipped update",
                        },
                    )
                raise RuntimeError(
                    "Pre-clip gradient skip budget was exhausted"
                )
        else:
            optimizer.step()
            skipped_preclip_consecutive = 0
        distributed_values = distributed_mean_dict(
            {
                "loss": local_loss_value,
                "grad_norm": float(grad_norm),
                **local_terms,
            },
            accelerator,
        )
        loss_value = distributed_values.pop("loss")
        reduced_grad_norm = distributed_values.pop("grad_norm")
        record: Dict[str, object] = {
            "step": step,
            "stage": stage,
            "loss": loss_value,
            "grad_norm": reduced_grad_norm,
            "lr_ssr": learning_rates["ssr"],
            "lr_heads": learning_rates["heads"],
            "lr_backbone": learning_rates["backbone"],
            **distributed_values,
            **step_maxima,
            "ssr_raw_residual_warning": int(warning_now),
            "ssr_saturation_streak": saturation_streak,
            "optimization_step_seconds": time.perf_counter() - step_start,
        }
        training_history.append(record)

        if step % args.eval_every == 0 or step == args.steps:
            accelerator.wait_for_everyone()
            periodic = None
            if accelerator.is_main_process:
                periodic_records = evaluate_model(
                    raw_model,
                    periodic_samples,
                    device=device,
                    num_tokens=args.num_tokens,
                    refinement_steps=(0, args.refinement_steps),
                    batch_size=args.eval_batch_size,
                    boundary_threshold=args.boundary_threshold,
                    smooth_log_depth_residual_bound=(
                        args.smooth_log_depth_residual_bound
                    ),
                    max_voxel_depth_span=args.max_voxel_depth_span,
                )
                periodic = aggregate_evaluation(periodic_records)
            periodic = broadcast_main_object(periodic, accelerator)
            assert periodic is not None
            evaluation_history.append(
                flatten_periodic_evaluation(step, periodic)
            )
            base_score = float(
                periodic["0"][args.selection_split]["point_rel"]
            )
            refined_score = float(
                periodic[str(args.refinement_steps)][args.selection_split][
                    "point_rel"
                ]
            )
            if base_geometry_has_collapsed(
                current_point_rel=base_score,
                best_point_rel=best_base_score,
                ratio_threshold=args.max_base_to_best_ratio,
            ):
                if accelerator.is_main_process:
                    save_csv(output / "training_history.csv", training_history)
                    save_csv(output / "evaluation_history.csv", evaluation_history)
                    write_instability_event(
                        output,
                        {
                            "event": "periodic_base_collapse",
                            "step": step,
                            "stage": stage,
                            "selection_split": args.selection_split,
                            "base_point_rel": base_score,
                            "historical_best_base_point_rel": best_base_score,
                            "ratio_threshold": args.max_base_to_best_ratio,
                            "action": (
                                "aborted before overwriting the last safe "
                                "resume checkpoint"
                            ),
                        },
                    )
                raise RuntimeError(
                    "Periodic base point Rel indicates geometry collapse"
                )
            if refinement_has_collapsed(
                base_point_rel=base_score,
                refined_point_rel=refined_score,
                absolute_threshold=args.max_refined_point_rel,
                ratio_threshold=args.max_refined_to_base_ratio,
            ):
                if accelerator.is_main_process:
                    save_csv(output / "training_history.csv", training_history)
                    save_csv(output / "evaluation_history.csv", evaluation_history)
                    write_instability_event(
                        output,
                        {
                            "event": "periodic_refinement_collapse",
                            "step": step,
                            "stage": stage,
                            "selection_split": args.selection_split,
                            "base_point_rel": base_score,
                            "refined_point_rel": refined_score,
                            "refined_to_base_ratio": (
                                refined_score / max(base_score, 1e-12)
                            ),
                            "absolute_threshold": args.max_refined_point_rel,
                            "ratio_threshold": args.max_refined_to_base_ratio,
                            "action": (
                                "aborted before overwriting the last safe "
                                "resume checkpoint"
                            ),
                        },
                    )
                raise RuntimeError(
                    "Periodic refined point Rel indicates SSR collapse"
                )
            best_base_score = min(best_base_score, base_score)

            full_periodic = None
            full_eval_due = (
                args.full_eval_every > 0
                and (
                    step % args.full_eval_every == 0
                    or step == args.steps
                )
            )
            if accelerator.is_main_process and full_eval_due:
                full_records = evaluate_model(
                    raw_model,
                    train_samples,
                    device=device,
                    num_tokens=args.num_tokens,
                    refinement_steps=(0, args.refinement_steps),
                    batch_size=args.eval_batch_size,
                    boundary_threshold=args.boundary_threshold,
                    smooth_log_depth_residual_bound=(
                        args.smooth_log_depth_residual_bound
                    ),
                    max_voxel_depth_span=args.max_voxel_depth_span,
                )
                full_periodic = aggregate_evaluation(full_records)
            full_periodic = broadcast_main_object(
                full_periodic,
                accelerator,
            )
            if full_periodic is not None:
                full_evaluation_history.append(
                    flatten_periodic_evaluation(step, full_periodic)
                )

            selection_evaluation = (
                full_periodic
                if args.selection_scope == "composite"
                else periodic
            )
            score = None
            if selection_evaluation is not None:
                score = periodic_selection_score(
                    selection_evaluation,
                    refinement_step=args.refinement_steps,
                    split=args.selection_split,
                    scope=args.selection_scope,
                )
            if score is not None and score < best_score:
                best_score = score
                best_step = step
                if accelerator.is_main_process:
                    atomic_torch_save(
                        checkpoint_payload(
                            model=raw_model,
                            optimizer=optimizer,
                            step=step,
                            best_step=best_step,
                            best_score=best_score,
                            args=args,
                            cpu_generator=cpu_generator,
                            loss_generator=loss_generator,
                            initial_periodic=initial_periodic,
                            include_optimizer=(
                                args.best_checkpoint_include_optimizer
                            ),
                            best_base_score=best_base_score,
                            initial_full_evaluation=(
                                initial_full_evaluation
                            ),
                        ),
                        output / "checkpoint.pt",
                    )
                    atomic_json_save(
                        {
                            "step": best_step,
                            "score": best_score,
                            "selection_scope": args.selection_scope,
                            "selection_split": args.selection_split,
                            "contains_optimizer": (
                                args.best_checkpoint_include_optimizer
                            ),
                        },
                        output / "checkpoint_metadata.json",
                    )
            if accelerator.is_main_process:
                save_csv(output / "training_history.csv", training_history)
                save_csv(output / "evaluation_history.csv", evaluation_history)
                save_csv(
                    output / "full_evaluation_history.csv",
                    full_evaluation_history,
                )
                atomic_torch_save(
                    checkpoint_payload(
                        model=raw_model,
                        optimizer=optimizer,
                        step=step,
                        best_step=best_step,
                        best_score=best_score,
                        args=args,
                        cpu_generator=cpu_generator,
                        loss_generator=loss_generator,
                        initial_periodic=initial_periodic,
                        include_optimizer=True,
                        best_base_score=best_base_score,
                        initial_full_evaluation=initial_full_evaluation,
                    ),
                    output / "resume_checkpoint.pt",
                )
                if full_periodic is not None:
                    milestones = output / "milestones"
                    milestones.mkdir(parents=True, exist_ok=True)
                    atomic_torch_save(
                        checkpoint_payload(
                            model=raw_model,
                            optimizer=optimizer,
                            step=step,
                            best_step=best_step,
                            best_score=best_score,
                            args=args,
                            cpu_generator=cpu_generator,
                            loss_generator=loss_generator,
                            initial_periodic=initial_periodic,
                            include_optimizer=True,
                            best_base_score=best_base_score,
                            initial_full_evaluation=(
                                initial_full_evaluation
                            ),
                        ),
                        milestones / f"step_{step:06d}.pt",
                    )
            accelerator.wait_for_everyone()

        if (
            accelerator.is_main_process
            and (step == 1 or step % args.log_every == 0 or step == args.steps)
        ):
            print(
                json.dumps(
                    {
                        "step": step,
                        "stage": stage,
                        "loss": loss_value,
                        "grad_norm": reduced_grad_norm,
                        "learning_rates": learning_rates,
                        "best_step": best_step,
                        "selection_split": args.selection_split,
                        "best_selection_score": best_score,
                        "optimizer_step_skipped": optimizer_step_skipped,
                        "skipped_preclip_total": skipped_preclip_total,
                        "elapsed_seconds": time.perf_counter() - train_start,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    train_elapsed = time.perf_counter() - train_start
    accelerator.wait_for_everyone()
    local_peak_memory = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0
    )
    peak_memory_by_process = accelerator.gather(
        torch.tensor([local_peak_memory], device=device, dtype=torch.int64)
    ).cpu().tolist()
    if accelerator.is_main_process:
        atomic_torch_save(
            checkpoint_payload(
                model=raw_model,
                optimizer=optimizer,
                step=args.steps,
                best_step=best_step,
                best_score=best_score,
                args=args,
                cpu_generator=cpu_generator,
                loss_generator=loss_generator,
                initial_periodic=initial_periodic,
                include_optimizer=True,
                best_base_score=best_base_score,
                initial_full_evaluation=initial_full_evaluation,
            ),
            output / "latest_checkpoint.pt",
        )
        save_csv(output / "training_history.csv", training_history)
        save_csv(output / "evaluation_history.csv", evaluation_history)
        save_training_plot(
            output,
            training_history,
            evaluation_history,
            smoothing_window=args.loss_smoothing_window,
            detach_step=args.refiner_detach_steps,
            refinement_step=args.refinement_steps,
        )
        report = {
            "status": "complete",
            "experiment": (
                "Hypersim fixed-base SSR-only fine-tuning"
                if args.train_refiner_only
                else "Hypersim two-stage joint MoGe-3 fine-tuning"
            ),
            "paper_alignment": {
                "base_and_ssr_jointly_optimized": not args.train_refiner_only,
                "refiner_detached_during_warmup": (
                    not args.train_refiner_only
                ),
                "fixed_base_diagnostic_ablation": args.train_refiner_only,
                "dino_bfloat16_geometry_float32": True,
                "normal_prediction": False,
                "synthetic_only_controlled_screen": True,
                "real_data_routing_exercised": False,
            },
            "distributed": {
                "backend": args.ddp_backend,
                "world_size": accelerator.num_processes,
                "global_batch_size": args.batch_size,
                "local_batch_size": local_batch_size,
            },
            "counts": manifest["counts"],
            "shape": [args.height, args.width],
            "num_tokens": args.num_tokens,
            "ssr_normalization": args.ssr_normalization,
            "smooth_log_depth_residual_bound": (
                args.smooth_log_depth_residual_bound
            ),
            "optimization_steps": args.steps,
            "refiner_detach_steps": args.refiner_detach_steps,
            "backbone_frozen": args.freeze_backbone or args.train_refiner_only,
            "complete_base_and_2d_heads_frozen": args.train_refiner_only,
            "backbone_freeze_steps": args.backbone_freeze_steps,
            "backbone_warmup_end": args.backbone_warmup_end,
            "learning_rates": {
                "ssr": args.ssr_learning_rate,
                "heads": args.head_learning_rate,
                "backbone": args.backbone_learning_rate,
                "schedule": args.learning_rate_schedule,
                "decay_start_step": (
                    args.learning_rate_decay_start_step
                ),
                "decay_end_step": args.learning_rate_decay_end_step,
                "final_scale": args.learning_rate_final_scale,
            },
            "batch_size": args.batch_size,
            "microbatch_size": args.microbatch_size,
            "best_step": best_step,
            "selection_split": args.selection_split,
            "selection_scope": args.selection_scope,
            "best_selection_score": best_score,
            "initial_periodic": initial_periodic,
            "latest_periodic": {
                str(refinement_step): {
                    split: {
                        metric: evaluation_history[-1][
                            f"{split}/k{refinement_step}_{metric}"
                        ]
                        for metric in METRIC_KEYS
                    }
                    for split in ("train", "val")
                }
                for refinement_step in (0, args.refinement_steps)
            },
            "initial_loss": float(training_history[0]["loss"]),
            "final_loss": float(training_history[-1]["loss"]),
            "training_seconds_including_periodic_eval": train_elapsed,
            "total_seconds": time.perf_counter() - run_start,
            "peak_memory_bytes": max(peak_memory_by_process),
            "peak_memory_bytes_by_process": peak_memory_by_process,
            "stability": {
                "max_preclip_grad_norm": args.max_preclip_grad_norm,
                "max_abs_applied_log_depth_residual": (
                    args.max_abs_log_depth_residual
                ),
                "max_abs_raw_log_depth_residual": (
                    args.max_abs_raw_log_depth_residual
                ),
                "max_raw_log_depth_residual_p999": (
                    args.max_raw_log_depth_residual_p999
                ),
                "raw_residual_warning_threshold": (
                    args.raw_residual_warning_threshold
                ),
                "raw_residual_tail_threshold": (
                    args.raw_residual_tail_threshold
                ),
                "raw_residual_tail_weight": args.raw_residual_tail_weight,
                "raw_residual_peak_weight": args.raw_residual_peak_weight,
                "max_bound_saturation_fraction": (
                    args.max_bound_saturation_fraction
                ),
                "max_consecutive_saturated_steps": (
                    args.max_consecutive_saturated_steps
                ),
                "max_voxel_depth_span": args.max_voxel_depth_span,
                "max_base_to_best_ratio": args.max_base_to_best_ratio,
                "skipped_preclip_steps": skipped_preclip_total,
                "max_skipped_preclip_steps": (
                    args.max_skipped_preclip_steps
                ),
                "max_consecutive_skipped_preclip_steps": (
                    args.max_consecutive_skipped_preclip_steps
                ),
            },
            "resumed_from": str(resume_path) if resume_path else None,
            "warm_started_from": (
                str(warm_start_path) if warm_start_path else None
            ),
            "transitioned_from": (
                str(transition_path) if transition_path else None
            ),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "requested_device": args.device,
            "artifacts": {
                "best_checkpoint": "checkpoint.pt",
                "latest_checkpoint": "latest_checkpoint.pt",
                "resume_checkpoint": "resume_checkpoint.pt",
                "training_history": "training_history.csv",
                "evaluation_history": "evaluation_history.csv",
                "full_evaluation_history": "full_evaluation_history.csv",
                "milestones": "milestones/",
                "training_curves": "training_curves.png",
                "stability_events": "stability_events.jsonl",
            },
        }
        (output / "report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False))
    accelerator.wait_for_everyone()
    accelerator.end_training()


if __name__ == "__main__":
    main()
