from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
import traceback
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch

from moge.model.v3 import MoGeModel
from moge.scripts.overfit_hypersim_v3 import geometry_loss, moving_average
from moge.scripts.train_hypersim_smallset_v3 import boundary_f1, load_raw_sample
from moge.train.losses_v3 import solve_global_affine_alignment
from moge.train.trainer_v3 import TrainingScheduleV3, build_v3_optimizer
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


EVALUATION_STEPS = (0, 1, 3, 5)
METRIC_NAMES = (
    "point_rel",
    "depth_rel",
    "depth_delta_1.01",
    "depth_delta_1.25",
    "boundary_f1",
)
SCOPE_NAMES = ("full", "crop", "structure")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paper-style detached-then-joint overfitting of MoGe-3 on one "
            "locked Hypersim fine-structure frame"
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument("--training-refinement-steps", type=int, default=3)
    parser.add_argument(
        "--evaluation-refinement-steps",
        type=int,
        nargs="+",
        default=list(EVALUATION_STEPS),
    )
    parser.add_argument("--stage1-min-steps", type=int, default=2000)
    parser.add_argument("--stage1-max-steps", type=int, default=5000)
    parser.add_argument("--joint-min-steps", type=int, default=2000)
    parser.add_argument("--joint-max-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--plateau-patience-evals", type=int, default=5)
    parser.add_argument(
        "--plateau-relative-improvement",
        type=float,
        default=0.005,
    )
    parser.add_argument("--loss-plateau-window", type=int, default=500)
    parser.add_argument("--loss-plateau-chunk", type=int, default=100)
    parser.add_argument("--success-patience-evals", type=int, default=3)
    parser.add_argument("--success-full-depth-rel", type=float, default=0.01)
    parser.add_argument(
        "--success-full-delta-1.01",
        dest="success_full_delta_1_01",
        type=float,
        default=0.95,
    )
    parser.add_argument("--success-structure-point-rel", type=float, default=0.02)
    parser.add_argument("--backbone-freeze-steps", type=int, default=1000)
    parser.add_argument("--backbone-warmup-end", type=int, default=2000)
    parser.add_argument("--ssr-learning-rate", type=float, default=2e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-5)
    parser.add_argument("--backbone-learning-rate", type=float, default=5e-7)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--global-weight", type=float, default=1.0)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--local-scales", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--loss-smoothing-window", type=int, default=100)
    parser.add_argument("--seed", type=int, default=91)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(args.height, args.width, args.num_tokens) <= 0:
        raise ValueError("Image shape and token count must be positive")
    if args.training_refinement_steps != 3:
        raise ValueError("Exp9 trains exactly K=3")
    requested = tuple(sorted(set(args.evaluation_refinement_steps)))
    if requested != EVALUATION_STEPS:
        raise ValueError(f"Exp9 evaluates exactly K={EVALUATION_STEPS}")
    if not (
        0 < args.stage1_min_steps <= args.stage1_max_steps
        and 0 < args.joint_min_steps <= args.joint_max_steps
    ):
        raise ValueError("Stage minimums must be positive and not exceed maximums")
    if not (
        0
        <= args.backbone_freeze_steps
        < args.backbone_warmup_end
        <= args.stage1_min_steps
    ):
        raise ValueError(
            "Expected backbone freeze < warmup <= stage-one minimum"
        )
    if args.eval_every <= 0 or args.plateau_patience_evals <= 0:
        raise ValueError("Evaluation interval and patience must be positive")
    if args.loss_plateau_window < 2 * args.loss_plateau_chunk:
        raise ValueError("Loss plateau window must contain two comparison chunks")
    if args.loss_plateau_chunk <= 0:
        raise ValueError("Loss plateau chunk must be positive")
    if not 0 < args.plateau_relative_improvement < 1:
        raise ValueError("Plateau relative improvement must be in (0, 1)")
    if min(
        args.ssr_learning_rate,
        args.head_learning_rate,
        args.backbone_learning_rate,
    ) <= 0:
        raise ValueError("All peak learning rates must be positive")


def relative_improvement(previous: float, current: float) -> float:
    if not math.isfinite(previous) or not math.isfinite(current):
        return float("-inf")
    return (previous - current) / max(abs(previous), 1e-12)


def loss_window_improvement(
    losses: Sequence[float],
    *,
    window: int,
    chunk: int,
) -> float | None:
    if len(losses) < window:
        return None
    values = np.asarray(losses[-window:], dtype=np.float64)
    first = float(values[:chunk].mean())
    last = float(values[-chunk:].mean())
    return relative_improvement(first, last)


def meaningful_best_update(
    best: float | None,
    value: float,
    threshold: float,
) -> Tuple[float, bool]:
    if best is None or not math.isfinite(best):
        return value, True
    if relative_improvement(best, value) >= threshold:
        return value, True
    return best, False


def plateau_reached(
    stale_evaluations: Mapping[str, int],
    *,
    patience: int,
    loss_improvement: float | None = None,
    threshold: float = 0.005,
) -> bool:
    if not stale_evaluations:
        return False
    metrics_stale = all(value >= patience for value in stale_evaluations.values())
    loss_stale = (
        True
        if loss_improvement is None
        else loss_improvement < threshold
    )
    return metrics_stale and loss_stale


def backbone_learning_rate(
    step: int,
    *,
    freeze_steps: int,
    warmup_end: int,
    peak_lr: float,
    decay_steps: int = 25_000,
) -> float:
    if step <= freeze_steps:
        return 0.0
    if step < warmup_end:
        return peak_lr * (step - freeze_steps) / (warmup_end - freeze_steps)
    return peak_lr * 0.5 ** ((step - warmup_end) // decay_steps)


def set_learning_rates(
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
) -> Dict[str, float]:
    rates = {
        "ssr": args.ssr_learning_rate * 0.5 ** (step // 10_000),
        "heads": args.head_learning_rate
        * 0.5 ** (max(0, step - args.backbone_warmup_end) // 25_000),
        "backbone": backbone_learning_rate(
            step,
            freeze_steps=args.backbone_freeze_steps,
            warmup_end=args.backbone_warmup_end,
            peak_lr=args.backbone_learning_rate,
        ),
    }
    for group in optimizer.param_groups:
        group["lr"] = rates[str(group["name"])]
    return rates


def configure_depth_trainable_modules(model: MoGeModel) -> Dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = {
        "ssr": model.ssr,
        "neck": model.neck,
        "points_head": model.points_head,
        "backbone": model.encoder.backbone,
    }
    counts = {}
    for name, module in modules.items():
        for parameter in module.parameters():
            parameter.requires_grad_(True)
        counts[name] = sum(parameter.numel() for parameter in module.parameters())
    return counts


def set_backbone_trainable(model: MoGeModel, trainable: bool) -> None:
    for parameter in model.encoder.backbone.parameters():
        parameter.requires_grad_(trainable)


def load_sample_and_selection(
    data: Path,
    selection_path: Path,
    sample_id: str,
    *,
    height: int,
    width: int,
    safe_root: Path,
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, object], Dict[str, object]]:
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    samples = {
        str(sample["id"]): sample for sample in manifest.get("samples", [])
    }
    if sample_id not in samples:
        raise ValueError(f"Sample is absent from manifest: {sample_id}")
    metadata = samples[sample_id]
    if metadata.get("split") != "train":
        raise ValueError(f"Exp9 requires a training sample: {sample_id}")

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("status") != "locked-before-rendering":
        raise ValueError("Selection must be locked before rendering")
    if selection.get("selection_inputs") != "RGB and ground truth only; no predictions":
        raise ValueError("Selection provenance must exclude predictions")
    if list(selection.get("shape", [])) != [height, width]:
        raise ValueError("Selection shape does not match training shape")
    entries = {
        str(entry["id"]): entry for entry in selection.get("entries", [])
    }
    if sample_id not in entries:
        raise ValueError(f"Sample is absent from locked selection: {sample_id}")
    entry = entries[sample_id]
    crop = tuple(int(value) for value in entry["crop_xyxy"])
    x0, y0, x1, y1 = crop
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Invalid crop for {sample_id}: {crop}")
    display_mask = entry.get("display_mask")
    if not isinstance(display_mask, dict):
        raise ValueError(f"Selection has no display mask for {sample_id}")

    image, gt_points = load_raw_sample(
        data,
        metadata,
        height,
        width,
        safe_root,
    )
    return image, gt_points, metadata, {**entry, "crop_xyxy": list(crop)}


def build_structure_mask(
    gt_points: torch.Tensor,
    crop: Sequence[int],
    config: Mapping[str, object],
) -> torch.Tensor:
    x0, y0, x1, y1 = (int(value) for value in crop)
    gt_crop = gt_points[y0:y1, x0:x1]
    valid = torch.isfinite(gt_crop).all(dim=-1) & (gt_crop[..., 2] > 0)
    values = gt_crop[..., 2][valid]
    if not values.numel():
        raise ValueError("Locked crop contains no valid ground-truth points")
    mask_type = str(config.get("type"))
    if mask_type == "near_quantile":
        quantile = float(config["quantile"])
        if not 0 < quantile < 1:
            raise ValueError("near_quantile must be in (0, 1)")
        maximum = torch.quantile(values.float(), quantile)
        mask = valid & (gt_crop[..., 2] <= maximum)
    elif mask_type == "depth_range":
        minimum = float(config.get("minimum_depth_m", 0.0))
        maximum = float(config["maximum_depth_m"])
        mask = (
            valid
            & (gt_crop[..., 2] >= minimum)
            & (gt_crop[..., 2] <= maximum)
        )
    else:
        raise ValueError(f"Unsupported structure mask: {mask_type}")
    if int(mask.sum()) < 16:
        raise ValueError("Locked structure mask contains fewer than 16 pixels")
    return mask


def _scope_metrics(
    aligned: torch.Tensor,
    gt: torch.Tensor,
    mask: torch.Tensor,
    *,
    boundary_threshold: float,
) -> Dict[str, float]:
    valid = (
        mask
        & torch.isfinite(aligned).all(dim=-1)
        & torch.isfinite(gt).all(dim=-1)
        & (gt[..., 2] > 0)
    )
    if not bool(valid.any()):
        raise ValueError("Evaluation scope contains no valid points")
    safe_gt = torch.where(valid[..., None], gt, torch.ones_like(gt))
    denominator = safe_gt[..., 2].clamp_min(1e-6)
    point_rel = (aligned - safe_gt).norm(dim=-1) / denominator
    depth_rel = (aligned[..., 2] - safe_gt[..., 2]).abs() / denominator
    ratio = torch.maximum(
        aligned[..., 2] / denominator,
        denominator / aligned[..., 2].clamp_min(1e-6),
    )
    return {
        "pixels": int(valid.sum().item()),
        "point_rel": float(point_rel[valid].mean().item()),
        "depth_rel": float(depth_rel[valid].mean().item()),
        "depth_delta_1.01": float((ratio[valid] < 1.01).float().mean().item()),
        "depth_delta_1.25": float((ratio[valid] < 1.25).float().mean().item()),
        "boundary_f1": boundary_f1(
            aligned[..., 2],
            gt[..., 2],
            valid,
            boundary_threshold,
        ),
    }


@torch.no_grad()
def evaluate_single_image(
    model: MoGeModel,
    image: torch.Tensor,
    gt_points: torch.Tensor,
    *,
    crop: Sequence[int],
    structure_mask: torch.Tensor,
    num_tokens: int,
    refinement_steps: Sequence[int],
    boundary_threshold: float,
) -> Tuple[Dict[str, object], Dict[int, torch.Tensor]]:
    requested = tuple(sorted(set(int(value) for value in refinement_steps)))
    was_training = model.training
    model.eval()
    output = model(
        image[None],
        num_tokens=num_tokens,
        num_refinement_steps=max(requested),
        return_intermediates=True,
    )
    sequence = output["points_sequence"]
    gt = gt_points.to(image.device)
    x0, y0, x1, y1 = (int(value) for value in crop)
    full_valid = torch.isfinite(gt).all(dim=-1) & (gt[..., 2] > 0)
    crop_valid = full_valid[y0:y1, x0:x1]
    structure = structure_mask.to(image.device)
    metrics: Dict[str, object] = {}
    aligned_by_k: Dict[int, torch.Tensor] = {}
    for k in requested:
        prediction = sequence[k][0].float()
        alignment = solve_global_affine_alignment(prediction[None], gt[None])
        aligned = alignment.apply(prediction[None])[0]
        aligned_by_k[k] = aligned.detach().cpu()
        metrics[f"k{k}"] = {
            "full": _scope_metrics(
                aligned,
                gt,
                full_valid,
                boundary_threshold=boundary_threshold,
            ),
            "crop": _scope_metrics(
                aligned[y0:y1, x0:x1],
                gt[y0:y1, x0:x1],
                crop_valid,
                boundary_threshold=boundary_threshold,
            ),
            "structure": _scope_metrics(
                aligned[y0:y1, x0:x1],
                gt[y0:y1, x0:x1],
                structure,
                boundary_threshold=boundary_threshold,
            ),
            "alignment": {
                "scale": float(alignment.scale.mean().item()),
                "z_shift": float(alignment.shift[..., 2].mean().item()),
            },
        }
    if was_training:
        model.train()
    return metrics, aligned_by_k


def zero_identity_error(aligned_by_k: Mapping[int, torch.Tensor]) -> float:
    base = aligned_by_k[0]
    return max(
        float((aligned_by_k[k] - base).abs().max().item())
        for k in aligned_by_k
        if k
    )


def flatten_metrics(
    snapshot: str,
    step: int,
    stage: str,
    metrics: Mapping[str, object],
) -> Dict[str, object]:
    row: Dict[str, object] = {
        "snapshot": snapshot,
        "step": step,
        "stage": stage,
    }
    for k_name, scope_values in metrics.items():
        for scope in SCOPE_NAMES:
            values = scope_values[scope]
            for metric in METRIC_NAMES:
                row[f"{k_name}/{scope}/{metric}"] = values[metric]
    return row


def save_csv(path: Path, rows: Iterable[Mapping[str, object]]) -> None:
    materialized = list(rows)
    if not materialized:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(materialized[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(materialized)


def atomic_torch_save(payload: Mapping[str, object], path: Path) -> None:
    temporary = path.with_name(f"{path.name}.incomplete")
    torch.save(dict(payload), temporary)
    os.replace(temporary, path)


def checkpoint_payload(
    *,
    model: MoGeModel,
    optimizer: torch.optim.Optimizer,
    step: int,
    phase: str,
    stage1_end_step: int | None,
    best_joint_step: int | None,
    best_joint_score: float | None,
    training_history: Sequence[Mapping[str, object]],
    evaluation_history: Sequence[Mapping[str, object]],
    trackers: Mapping[str, object],
    cpu_generator: torch.Generator,
    loss_generator: torch.Generator,
    args: argparse.Namespace,
    include_optimizer: bool,
) -> Dict[str, object]:
    payload: Dict[str, object] = {
        "model": model.state_dict(),
        "step": step,
        "phase": phase,
        "stage1_end_step": stage1_end_step,
        "best_joint_step": best_joint_step,
        "best_joint_score": best_joint_score,
        "training_history": list(training_history),
        "evaluation_history": list(evaluation_history),
        "trackers": dict(trackers),
        "cpu_generator_state": cpu_generator.get_state(),
        "loss_generator_state": loss_generator.get_state(),
        "pretrained": args.pretrained,
        "args": vars(args),
        "training_mode": "single-image detached-then-joint depth overfit",
        "normal_prediction": False,
    }
    if include_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    return payload


def save_snapshot(
    metrics_dir: Path,
    name: str,
    *,
    sample_id: str,
    step: int,
    stage: str,
    stop_reason: str,
    metrics: Mapping[str, object],
    extra: Mapping[str, object] | None = None,
) -> Dict[str, object]:
    payload = {
        "status": "complete",
        "sample_id": sample_id,
        "snapshot": name,
        "step": step,
        "stage": stage,
        "stop_reason": stop_reason,
        "metrics": metrics,
        **(dict(extra) if extra else {}),
    }
    (metrics_dir / f"{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return payload


def save_training_plot(
    artifacts_dir: Path,
    training_history: Sequence[Mapping[str, object]],
    evaluation_history: Sequence[Mapping[str, object]],
    *,
    stage1_end_step: int,
    smoothing_window: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    steps = np.asarray([row["step"] for row in training_history], dtype=np.int64)
    losses = np.asarray([row["loss"] for row in training_history], dtype=np.float64)
    smooth = moving_average(
        losses,
        min(max(1, smoothing_window), len(losses)),
    )
    eval_steps = np.asarray(
        [row["step"] for row in evaluation_history],
        dtype=np.int64,
    )

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(3, 1, figsize=(11, 13))
    axes[0].plot(steps, losses, alpha=0.15, linewidth=0.7, color="#60A5FA")
    axes[0].plot(steps, smooth, linewidth=2.0, color="#1D4ED8")
    axes[0].set_ylabel("Geometry loss")
    axes[0].set_title("Detached-to-joint single-image training")
    for k, style in ((0, "--"), (3, "-")):
        axes[1].plot(
            eval_steps,
            [row[f"k{k}/full/point_rel"] for row in evaluation_history],
            linestyle=style,
            marker="o",
            label=f"Full K={k}",
        )
        axes[2].plot(
            eval_steps,
            [row[f"k{k}/structure/point_rel"] for row in evaluation_history],
            linestyle=style,
            marker="o",
            label=f"Structure K={k}",
        )
    for axis in axes:
        axis.axvline(stage1_end_step, color="#7C3AED", linestyle="--")
        axis.set_xlabel("Optimization step")
    axes[1].set_ylabel("Aligned point Rel")
    axes[1].set_title("Full-frame deterministic metrics")
    axes[2].set_ylabel("Aligned point Rel")
    axes[2].set_title("Locked fine-structure metrics")
    axes[1].legend()
    axes[2].legend()
    figure.tight_layout()
    figure.savefig(artifacts_dir / "training_curves.png", dpi=180)
    figure.savefig(artifacts_dir / "training_curves.pdf")
    plt.close(figure)


def success_reached(metrics: Mapping[str, object], args: argparse.Namespace) -> bool:
    k0 = metrics["k0"]
    k3 = metrics["k3"]
    return (
        k3["full"]["depth_rel"] < args.success_full_depth_rel
        and k3["full"]["depth_delta_1.01"] > args.success_full_delta_1_01
        and k3["structure"]["point_rel"] < args.success_structure_point_rel
        and k3["full"]["point_rel"] < k0["full"]["point_rel"]
        and k3["structure"]["point_rel"] < k0["structure"]["point_rel"]
    )


def run(args: argparse.Namespace) -> None:
    validate_args(args)
    data = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    selection = assert_safe_path(
        args.selection,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    metrics_dir = output / "metrics"
    artifacts_dir = output / "artifacts"
    checkpoints_dir = output / "checkpoints"
    for directory in (metrics_dir, artifacts_dir, checkpoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.reset_peak_memory_stats()

    image, gt_points, metadata, selection_entry = load_sample_and_selection(
        data,
        selection,
        args.sample_id,
        height=args.height,
        width=args.width,
        safe_root=args.safe_root,
    )
    crop = selection_entry["crop_xyxy"]
    structure_mask = build_structure_mask(
        gt_points,
        crop,
        selection_entry["display_mask"],
    )
    image = image.to(device)
    gt_points = gt_points.to(device)

    model = MoGeModel.from_pretrained(args.pretrained).to(device)
    parameter_counts = configure_depth_trainable_modules(model)
    model.enable_gradient_checkpointing()
    schedule = TrainingScheduleV3(
        refiner_detach_steps=args.stage1_max_steps,
        backbone_freeze_steps=args.backbone_freeze_steps,
        backbone_warmup_end=args.backbone_warmup_end,
        ssr_lr=args.ssr_learning_rate,
        head_lr=args.head_learning_rate,
        backbone_lr=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        gradient_clip_norm=args.gradient_clip_norm,
    )
    optimizer = build_v3_optimizer(model, schedule)
    cpu_generator = torch.Generator().manual_seed(args.seed + 1)
    loss_generator = torch.Generator(device=device).manual_seed(args.seed + 2)

    step = 0
    phase = "detached"
    stage1_end_step: int | None = None
    best_joint_step: int | None = None
    best_joint_score: float | None = None
    training_history: list[Dict[str, object]] = []
    evaluation_history: list[Dict[str, object]] = []
    trackers: MutableMapping[str, Any] = {
        "stage1_best": {"k0": None, "k3": None},
        "stage1_stale": {"k0": 0, "k3": 0},
        "joint_best_score": None,
        "joint_stale": 0,
        "success_streak": 0,
    }

    if args.resume is not None:
        resume_path = assert_safe_path(
            args.resume,
            safe_root=args.safe_root,
            must_exist=True,
        )
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        step = int(checkpoint["step"])
        phase = str(checkpoint["phase"])
        stage1_end_step = checkpoint["stage1_end_step"]
        best_joint_step = checkpoint["best_joint_step"]
        best_joint_score = checkpoint["best_joint_score"]
        training_history = list(checkpoint["training_history"])
        evaluation_history = list(checkpoint["evaluation_history"])
        trackers = dict(checkpoint["trackers"])
        cpu_generator.set_state(checkpoint["cpu_generator_state"])
        loss_generator.set_state(checkpoint["loss_generator_state"])
        del checkpoint
    else:
        initial_metrics, aligned = evaluate_single_image(
            model,
            image,
            gt_points,
            crop=crop,
            structure_mask=structure_mask,
            num_tokens=args.num_tokens,
            refinement_steps=args.evaluation_refinement_steps,
            boundary_threshold=args.boundary_threshold,
        )
        identity_error = zero_identity_error(aligned)
        if identity_error != 0.0:
            raise RuntimeError(
                f"Zero-initialized SSR is not an exact identity: {identity_error}"
            )
        save_snapshot(
            metrics_dir,
            "initial",
            sample_id=args.sample_id,
            step=0,
            stage="initial",
            stop_reason="before_training",
            metrics=initial_metrics,
            extra={"zero_identity_max_abs_error": identity_error},
        )
        evaluation_history.append(
            flatten_metrics("periodic", 0, "initial", initial_metrics)
        )
        atomic_torch_save(
            checkpoint_payload(
                model=model,
                optimizer=optimizer,
                step=0,
                phase=phase,
                stage1_end_step=None,
                best_joint_step=None,
                best_joint_score=None,
                training_history=training_history,
                evaluation_history=evaluation_history,
                trackers=trackers,
                cpu_generator=cpu_generator,
                loss_generator=loss_generator,
                args=args,
                include_optimizer=False,
            ),
            checkpoints_dir / "initial.pt",
        )

    start_time = time.perf_counter()
    stage1_stop_reason = None
    joint_stop_reason = None
    while True:
        if phase == "detached" and step >= args.stage1_max_steps:
            stage1_stop_reason = "maximum_steps"
        if phase == "joint" and stage1_end_step is not None:
            if step - stage1_end_step >= args.joint_max_steps:
                joint_stop_reason = "maximum_steps"
        if stage1_stop_reason is not None:
            metrics, _ = evaluate_single_image(
                model,
                image,
                gt_points,
                crop=crop,
                structure_mask=structure_mask,
                num_tokens=args.num_tokens,
                refinement_steps=args.evaluation_refinement_steps,
                boundary_threshold=args.boundary_threshold,
            )
            stage1_end_step = step
            save_snapshot(
                metrics_dir,
                "stage1",
                sample_id=args.sample_id,
                step=step,
                stage="detached",
                stop_reason=stage1_stop_reason,
                metrics=metrics,
                extra={"plateau_trackers": trackers},
            )
            atomic_torch_save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    phase="joint",
                    stage1_end_step=stage1_end_step,
                    best_joint_step=best_joint_step,
                    best_joint_score=best_joint_score,
                    training_history=training_history,
                    evaluation_history=evaluation_history,
                    trackers=trackers,
                    cpu_generator=cpu_generator,
                    loss_generator=loss_generator,
                    args=args,
                    include_optimizer=True,
                ),
                checkpoints_dir / "stage1.pt",
            )
            phase = "joint"
            stage1_stop_reason = None
            trackers["joint_best_score"] = None
            trackers["joint_stale"] = 0
            trackers["success_streak"] = 0
            continue
        if joint_stop_reason is not None:
            break

        step += 1
        set_backbone_trainable(
            model,
            step > args.backbone_freeze_steps,
        )
        learning_rates = set_learning_rates(optimizer, step, args)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        forward_output = model(
            image[None],
            num_tokens=args.num_tokens,
            num_refinement_steps=args.training_refinement_steps,
            return_intermediates=True,
            detach_base_from_refiner=phase == "detached",
        )
        sequence = forward_output["points_sequence"]
        base_loss, base_terms = geometry_loss(
            sequence[:1],
            gt_points[None],
            global_weight=args.global_weight,
            local_weight=args.local_weight,
            edge_weight=args.edge_weight,
            local_scales=tuple(args.local_scales),
            generator=loss_generator,
        )
        refined_loss, refined_terms = geometry_loss(
            sequence[1:],
            gt_points[None],
            global_weight=args.global_weight,
            local_weight=args.local_weight,
            edge_weight=args.edge_weight,
            local_scales=tuple(args.local_scales),
            generator=loss_generator,
        )
        loss = base_loss + refined_loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            (
                parameter
                for parameter in model.parameters()
                if parameter.grad is not None
            ),
            args.gradient_clip_norm,
        )
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        optimizer.step()
        record = {
            "step": step,
            "stage": phase,
            "loss": float(loss.detach().item()),
            "base_loss": float(base_loss.detach().item()),
            "refined_loss": float(refined_loss.detach().item()),
            "grad_norm": float(grad_norm.item()),
            "lr_ssr": learning_rates["ssr"],
            "lr_heads": learning_rates["heads"],
            "lr_backbone": learning_rates["backbone"],
            **{f"base/{key}": value for key, value in base_terms.items()},
            **{f"refined/{key}": value for key, value in refined_terms.items()},
        }
        training_history.append(record)

        should_evaluate = step % args.eval_every == 0
        if should_evaluate:
            metrics, _ = evaluate_single_image(
                model,
                image,
                gt_points,
                crop=crop,
                structure_mask=structure_mask,
                num_tokens=args.num_tokens,
                refinement_steps=args.evaluation_refinement_steps,
                boundary_threshold=args.boundary_threshold,
            )
            evaluation_history.append(
                flatten_metrics("periodic", step, phase, metrics)
            )
            if phase == "detached":
                for key, value in (
                    ("k0", metrics["k0"]["full"]["point_rel"]),
                    ("k3", metrics["k3"]["full"]["point_rel"]),
                ):
                    best, improved = meaningful_best_update(
                        trackers["stage1_best"][key],
                        float(value),
                        args.plateau_relative_improvement,
                    )
                    trackers["stage1_best"][key] = best
                    trackers["stage1_stale"][key] = (
                        0
                        if improved
                        else int(trackers["stage1_stale"][key]) + 1
                    )
                stage_losses = [
                    float(row["loss"])
                    for row in training_history
                    if row["stage"] == "detached"
                ]
                loss_improvement = loss_window_improvement(
                    stage_losses,
                    window=args.loss_plateau_window,
                    chunk=args.loss_plateau_chunk,
                )
                if (
                    step >= args.stage1_min_steps
                    and plateau_reached(
                        trackers["stage1_stale"],
                        patience=args.plateau_patience_evals,
                        loss_improvement=loss_improvement,
                        threshold=args.plateau_relative_improvement,
                    )
                ):
                    stage1_stop_reason = "detected_plateau"
            else:
                score = float(
                    metrics["k3"]["full"]["point_rel"]
                    + metrics["k3"]["structure"]["point_rel"]
                )
                best, improved = meaningful_best_update(
                    trackers["joint_best_score"],
                    score,
                    args.plateau_relative_improvement,
                )
                trackers["joint_best_score"] = best
                trackers["joint_stale"] = (
                    0 if improved else int(trackers["joint_stale"]) + 1
                )
                if best_joint_score is None or score < best_joint_score:
                    best_joint_score = score
                    best_joint_step = step
                    atomic_torch_save(
                        checkpoint_payload(
                            model=model,
                            optimizer=optimizer,
                            step=step,
                            phase=phase,
                            stage1_end_step=stage1_end_step,
                            best_joint_step=best_joint_step,
                            best_joint_score=best_joint_score,
                            training_history=training_history,
                            evaluation_history=evaluation_history,
                            trackers=trackers,
                            cpu_generator=cpu_generator,
                            loss_generator=loss_generator,
                            args=args,
                            include_optimizer=False,
                        ),
                        checkpoints_dir / "best_joint.pt",
                    )
                joint_steps = step - int(stage1_end_step)
                trackers["success_streak"] = (
                    int(trackers["success_streak"]) + 1
                    if success_reached(metrics, args)
                    else 0
                )
                if joint_steps >= args.joint_min_steps:
                    if int(trackers["success_streak"]) >= args.success_patience_evals:
                        joint_stop_reason = "success_threshold"
                    elif int(trackers["joint_stale"]) >= args.plateau_patience_evals:
                        joint_stop_reason = "detected_plateau"

            save_csv(metrics_dir / "history.csv", evaluation_history)
            save_csv(metrics_dir / "training_history.csv", training_history)
            atomic_torch_save(
                checkpoint_payload(
                    model=model,
                    optimizer=optimizer,
                    step=step,
                    phase=phase,
                    stage1_end_step=stage1_end_step,
                    best_joint_step=best_joint_step,
                    best_joint_score=best_joint_score,
                    training_history=training_history,
                    evaluation_history=evaluation_history,
                    trackers=trackers,
                    cpu_generator=cpu_generator,
                    loss_generator=loss_generator,
                    args=args,
                    include_optimizer=True,
                ),
                checkpoints_dir / "resume.pt",
            )

        if step == 1 or step % args.log_every == 0:
            print(
                json.dumps(
                    {
                        "sample_id": args.sample_id,
                        "step": step,
                        "stage": phase,
                        "loss": record["loss"],
                        "grad_norm": record["grad_norm"],
                        "learning_rates": learning_rates,
                        "stage1_end_step": stage1_end_step,
                        "best_joint_step": best_joint_step,
                        "elapsed_seconds": time.perf_counter() - start_time,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    atomic_torch_save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            step=step,
            phase=phase,
            stage1_end_step=stage1_end_step,
            best_joint_step=best_joint_step,
            best_joint_score=best_joint_score,
            training_history=training_history,
            evaluation_history=evaluation_history,
            trackers=trackers,
            cpu_generator=cpu_generator,
            loss_generator=loss_generator,
            args=args,
            include_optimizer=True,
        ),
        checkpoints_dir / "last.pt",
    )
    last_metrics, _ = evaluate_single_image(
        model,
        image,
        gt_points,
        crop=crop,
        structure_mask=structure_mask,
        num_tokens=args.num_tokens,
        refinement_steps=args.evaluation_refinement_steps,
        boundary_threshold=args.boundary_threshold,
    )
    save_snapshot(
        metrics_dir,
        "last",
        sample_id=args.sample_id,
        step=step,
        stage="joint",
        stop_reason=str(joint_stop_reason),
        metrics=last_metrics,
    )

    best_checkpoint = torch.load(
        checkpoints_dir / "best_joint.pt",
        map_location=device,
        weights_only=False,
    )
    model.load_state_dict(best_checkpoint["model"], strict=True)
    final_metrics, _ = evaluate_single_image(
        model,
        image,
        gt_points,
        crop=crop,
        structure_mask=structure_mask,
        num_tokens=args.num_tokens,
        refinement_steps=args.evaluation_refinement_steps,
        boundary_threshold=args.boundary_threshold,
    )
    save_snapshot(
        metrics_dir,
        "final",
        sample_id=args.sample_id,
        step=int(best_checkpoint["step"]),
        stage="joint_best",
        stop_reason="minimum_full_plus_structure_k3_point_rel",
        metrics=final_metrics,
        extra={
            "last_step": step,
            "joint_stop_reason": joint_stop_reason,
            "selection_score": best_joint_score,
        },
    )
    atomic_torch_save(
        checkpoint_payload(
            model=model,
            optimizer=optimizer,
            step=int(best_checkpoint["step"]),
            phase="joint_best",
            stage1_end_step=stage1_end_step,
            best_joint_step=best_joint_step,
            best_joint_score=best_joint_score,
            training_history=training_history,
            evaluation_history=evaluation_history,
            trackers=trackers,
            cpu_generator=cpu_generator,
            loss_generator=loss_generator,
            args=args,
            include_optimizer=False,
        ),
        checkpoints_dir / "final.pt",
    )
    save_csv(metrics_dir / "history.csv", evaluation_history)
    save_csv(metrics_dir / "training_history.csv", training_history)
    save_training_plot(
        artifacts_dir,
        training_history,
        evaluation_history,
        stage1_end_step=int(stage1_end_step),
        smoothing_window=args.loss_smoothing_window,
    )

    report = {
        "status": "complete",
        "experiment": "exp9 paper-style single-image fine-structure overfit",
        "sample_id": args.sample_id,
        "scene": metadata["scene"],
        "frame": metadata["frame"],
        "selection": selection_entry,
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "training_refinement_steps": args.training_refinement_steps,
        "evaluation_refinement_steps": args.evaluation_refinement_steps,
        "parameter_counts": parameter_counts,
        "normal_prediction": False,
        "augmentation": "disabled",
        "stage1_end_step": stage1_end_step,
        "best_joint_step": best_joint_step,
        "last_step": step,
        "joint_stop_reason": joint_stop_reason,
        "best_joint_score": best_joint_score,
        "elapsed_seconds": time.perf_counter() - start_time,
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device)
            if device.type == "cuda"
            else 0
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "initial": json.loads(
            (metrics_dir / "initial.json").read_text(encoding="utf-8")
        ),
        "stage1": json.loads(
            (metrics_dir / "stage1.json").read_text(encoding="utf-8")
        ),
        "final": json.loads(
            (metrics_dir / "final.json").read_text(encoding="utf-8")
        ),
        "last": json.loads(
            (metrics_dir / "last.json").read_text(encoding="utf-8")
        ),
    }
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


def main() -> None:
    args = parse_args()
    output: Path | None = None
    try:
        output = assert_safe_path(
            args.output,
            safe_root=args.safe_root,
            writable=True,
        )
        run(args)
    except BaseException as error:
        if output is not None:
            output.mkdir(parents=True, exist_ok=True)
            failure = {
                "status": "failed",
                "sample_id": getattr(args, "sample_id", None),
                "error_type": type(error).__name__,
                "error": str(error),
                "traceback": traceback.format_exc(),
            }
            (output / "failure_report.json").write_text(
                json.dumps(failure, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
