from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from moge.model.ssr import SelfGuidedSparseRefiner, factorize_points
from moge.model.v3 import MoGeModel
from moge.scripts.overfit_hypersim_v3 import (
    aligned_metrics,
    colorize_depth,
    geometry_loss,
    moving_average,
)
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


@dataclass
class CachedSample:
    sample_id: str
    split: str
    scene: str
    frame: int
    image: torch.Tensor
    gt_points: torch.Tensor
    base_points: torch.Tensor
    visual_features: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MoGe-3 SSR on a fixed small multi-scene Hypersim subset"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--global-weight", type=float, default=1.0)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--local-scales", type=int, nargs="*", default=[4, 16, 64])
    parser.add_argument("--eval-every", type=int, default=200)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--loss-smoothing-window", type=int, default=50)
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=23)
    return parser.parse_args()


def load_raw_sample(
    data_dir: Path,
    sample: Dict[str, object],
    height: int,
    width: int,
    safe_root: Path,
) -> Tuple[torch.Tensor, torch.Tensor]:
    matrix = np.asarray(sample["M_cam_from_uv"], dtype=np.float32)
    u = np.linspace(-1.0 + 1.0 / width, 1.0 - 1.0 / width, width, dtype=np.float32)
    v = np.linspace(-1.0 + 1.0 / height, 1.0 - 1.0 / height, height, dtype=np.float32)[::-1]
    grid_u, grid_v = np.meshgrid(u, v)
    uv1 = np.stack((grid_u, grid_v, np.ones_like(grid_u)), axis=-1)
    rays = uv1 @ matrix.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-8)
    rays *= np.asarray([1.0, -1.0, -1.0], dtype=np.float32)

    rgb_candidate = data_dir / sample["rgb"]["file"]
    depth_candidate = data_dir / sample["depth"]["file"]
    if rgb_candidate.is_symlink() or depth_candidate.is_symlink():
        raise PermissionError(f"Refusing symlinked experiment input: {sample['id']}")
    rgb_path = assert_safe_path(
        rgb_candidate,
        safe_root=safe_root,
        must_exist=True,
    )
    depth_path = assert_safe_path(
        depth_candidate,
        safe_root=safe_root,
        must_exist=True,
    )

    with Image.open(rgb_path) as image_file:
        image_file = image_file.convert("RGB").resize(
            (width, height),
            Image.Resampling.LANCZOS,
        )
        image = np.asarray(image_file, dtype=np.float32).copy() / 255.0
    with h5py.File(depth_path, "r") as file:
        radial_depth = file["dataset"][:].astype(np.float32)
    radial_depth = cv2.resize(
        radial_depth,
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    )
    valid = np.isfinite(radial_depth) & (radial_depth > 0)
    points = radial_depth[..., None] * rays
    points[~valid] = np.nan
    return (
        torch.from_numpy(image).permute(2, 0, 1).contiguous(),
        torch.from_numpy(points).contiguous(),
    )


def cache_base_predictions(
    data_dir: Path,
    manifest: Dict[str, object],
    model: MoGeModel,
    *,
    height: int,
    width: int,
    num_tokens: int,
    device: torch.device,
    safe_root: Path,
    cache_batch_size: int = 2,
) -> List[CachedSample]:
    raw = []
    for sample in manifest["samples"]:
        image, gt_points = load_raw_sample(
            data_dir,
            sample,
            height,
            width,
            safe_root,
        )
        raw.append((sample, image, gt_points))

    cached: List[CachedSample] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(raw), cache_batch_size):
            chunk = raw[start : start + cache_batch_size]
            images = torch.stack([entry[1] for entry in chunk]).to(device)
            output = model._forward_base(images, num_tokens)
            base_points = output["points"].float().cpu()
            visual_features = output["_visual_features"].float().cpu()
            for index, (sample, image, gt_points) in enumerate(chunk):
                cached.append(
                    CachedSample(
                        sample_id=str(sample["id"]),
                        split=str(sample["split"]),
                        scene=str(sample["scene"]),
                        frame=int(sample["frame"]),
                        image=image,
                        gt_points=gt_points,
                        base_points=base_points[index],
                        visual_features=visual_features[index],
                    )
                )
            del images, output
    return cached


def refine_cached(
    refiner: SelfGuidedSparseRefiner,
    base_points: torch.Tensor,
    visual_features: torch.Tensor,
    refinement_steps: int,
) -> List[torch.Tensor]:
    factorized = factorize_points(base_points.float())
    refined = base_points.float()
    sequence = []
    for _ in range(refinement_steps):
        residual, _ = refiner(factorized, visual_features.float())
        factorized = torch.cat(
            (factorized[..., :2], factorized[..., 2:3] + residual[..., None]),
            dim=-1,
        )
        refined = refined * residual.exp()[..., None]
        sequence.append(refined)
    return sequence


def depth_edge_map(
    depth: torch.Tensor,
    valid: torch.Tensor,
    threshold: float,
) -> torch.Tensor:
    safe = torch.where(valid, depth.clamp_min(1e-6), torch.ones_like(depth))
    log_depth = safe.log()
    horizontal = (log_depth[:, 1:] - log_depth[:, :-1]).abs()
    vertical = (log_depth[1:, :] - log_depth[:-1, :]).abs()
    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    vertical_valid = valid[1:, :] & valid[:-1, :]
    edges = torch.zeros_like(valid)
    edges[:, 1:] |= (horizontal > threshold) & horizontal_valid
    edges[:, :-1] |= (horizontal > threshold) & horizontal_valid
    edges[1:, :] |= (vertical > threshold) & vertical_valid
    edges[:-1, :] |= (vertical > threshold) & vertical_valid
    return edges


def boundary_f1(
    pred_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid: torch.Tensor,
    threshold: float,
) -> float:
    pred_edge = depth_edge_map(pred_depth, valid, threshold)
    gt_edge = depth_edge_map(gt_depth, valid, threshold)
    pred_count = int(pred_edge.sum())
    gt_count = int(gt_edge.sum())
    if pred_count == 0 or gt_count == 0:
        return 1.0 if pred_count == gt_count else 0.0
    pred_dilated = F.max_pool2d(
        pred_edge.float()[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0].bool()
    gt_dilated = F.max_pool2d(
        gt_edge.float()[None, None],
        kernel_size=3,
        stride=1,
        padding=1,
    )[0, 0].bool()
    precision = (pred_edge & gt_dilated).sum().float() / pred_edge.sum().clamp_min(1)
    recall = (gt_edge & pred_dilated).sum().float() / gt_edge.sum().clamp_min(1)
    return float((2 * precision * recall / (precision + recall).clamp_min(1e-8)).cpu())


def aggregate_metrics(records: Sequence[Dict[str, object]]) -> Dict[str, float]:
    keys = ("point_rel", "depth_rel", "depth_delta_1.01", "depth_delta_1.25", "boundary_f1")
    return {
        key: float(np.mean([float(record[key]) for record in records]))
        for key in keys
    }


@torch.no_grad()
def zero_initialization_error(
    refiner: SelfGuidedSparseRefiner,
    samples: Sequence[CachedSample],
    *,
    device: torch.device,
    refinement_steps: int,
    batch_size: int = 2,
) -> float:
    maximum = 0.0
    refiner.eval()
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        base = torch.stack([sample.base_points for sample in chunk]).to(device)
        visual = torch.stack([sample.visual_features for sample in chunk]).to(device)
        refined = refine_cached(refiner, base, visual, refinement_steps)[-1]
        maximum = max(maximum, float((refined - base).abs().max()))
    refiner.train()
    return maximum


@torch.no_grad()
def evaluate(
    refiner: SelfGuidedSparseRefiner,
    samples: Sequence[CachedSample],
    *,
    device: torch.device,
    refinement_steps: int,
    boundary_threshold: float,
    use_refiner: bool,
    batch_size: int = 2,
) -> Tuple[List[Dict[str, object]], Dict[str, torch.Tensor]]:
    refiner.eval()
    records: List[Dict[str, object]] = []
    predictions: Dict[str, torch.Tensor] = {}
    for start in range(0, len(samples), batch_size):
        chunk = samples[start : start + batch_size]
        base = torch.stack([sample.base_points for sample in chunk]).to(device)
        visual = torch.stack([sample.visual_features for sample in chunk]).to(device)
        gt = torch.stack([sample.gt_points for sample in chunk]).to(device)
        pred = (
            refine_cached(refiner, base, visual, refinement_steps)[-1]
            if use_refiner
            else base
        )
        for index, sample in enumerate(chunk):
            metrics, aligned_batch = aligned_metrics(
                pred[index : index + 1],
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
            records.append(
                {
                    "id": sample.sample_id,
                    "split": sample.split,
                    "scene": sample.scene,
                    "frame": sample.frame,
                    **metrics,
                }
            )
            predictions[sample.sample_id] = aligned.cpu()
    refiner.train()
    return records, predictions


def save_csv(path: Path, records: Iterable[Dict[str, object]]) -> None:
    records = list(records)
    if not records:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(records[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(records)


def save_training_plot(
    output: Path,
    losses: Sequence[float],
    eval_history: Sequence[Dict[str, float]],
    baseline: Dict[str, Dict[str, float]],
    smoothing_window: int,
    start_step: int = 0,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    values = np.asarray(losses, dtype=np.float64)
    steps = np.arange(start_step + 1, start_step + len(values) + 1)
    window = min(smoothing_window, max(1, len(values)))
    smooth = moving_average(values, window)
    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=False)

    axes[0].plot(steps, values, color="#5B8FF9", alpha=0.18, linewidth=0.7, label="Per-step loss")
    axes[0].plot(steps, smooth, color="#1D4ED8", linewidth=2.0, label=f"{window}-step mean")
    axes[0].set_xlabel("Optimization step")
    axes[0].set_ylabel("Geometry loss")
    axes[0].set_title("MoGe-3 Hypersim Small-Set SSR Training")
    axes[0].legend()

    eval_steps = [record["step"] for record in eval_history]
    for split, color in (("train", "#1D4ED8"), ("val", "#D94841")):
        axes[1].plot(
            eval_steps,
            [record[f"{split}/depth_rel"] for record in eval_history],
            color=color,
            marker="o",
            linewidth=2,
            label=f"{split} K=3",
        )
        axes[1].axhline(
            baseline[split]["depth_rel"],
            color=color,
            linestyle="--",
            alpha=0.65,
            label=f"{split} K=0 baseline",
        )
    axes[1].set_xlabel("Optimization step")
    axes[1].set_ylabel("Aligned depth Rel")
    axes[1].set_title("Train and held-out scene evaluation")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=180)
    figure.savefig(output / "training_curves.pdf")
    plt.close(figure)


def error_color(error: np.ndarray, valid: np.ndarray, maximum: float = 0.05) -> np.ndarray:
    normalized = np.clip(error / maximum, 0.0, 1.0)
    normalized[~valid] = 0
    colored = cv2.applyColorMap(
        np.round(255 * normalized).astype(np.uint8),
        cv2.COLORMAP_MAGMA,
    )
    colored[~valid] = 0
    return colored


def label_panel(panel: np.ndarray, text: str) -> np.ndarray:
    panel = panel.copy()
    cv2.putText(
        panel,
        text,
        (7, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return panel


def save_qualitative(
    output: Path,
    selected: Sequence[CachedSample],
    baseline_predictions: Dict[str, torch.Tensor],
    final_predictions: Dict[str, torch.Tensor],
) -> None:
    rows = []
    for sample in selected:
        image = (sample.image.permute(1, 2, 0).numpy().clip(0, 1)[:, :, ::-1] * 255).astype(np.uint8)
        gt = sample.gt_points[..., 2].numpy()
        valid = np.isfinite(gt)
        base = baseline_predictions[sample.sample_id][..., 2].numpy()
        final = final_predictions[sample.sample_id][..., 2].numpy()
        lo, hi = np.percentile(gt[valid], (2, 98))
        base_error = np.abs(base - gt) / np.clip(gt, 1e-6, None)
        final_error = np.abs(final - gt) / np.clip(gt, 1e-6, None)
        panels = [
            label_panel(image, f"{sample.split}: {sample.scene}/{sample.frame:04d}"),
            label_panel(colorize_depth(gt, valid, lo, hi), "GT depth"),
            label_panel(colorize_depth(base, valid, lo, hi), "K=0 aligned"),
            label_panel(colorize_depth(final, valid, lo, hi), "K=3 trained"),
            label_panel(error_color(base_error, valid), "K=0 relative error"),
            label_panel(error_color(final_error, valid), "K=3 relative error"),
        ]
        rows.append(np.concatenate(panels, axis=1))
    cv2.imwrite(str(output / "qualitative_comparison.png"), np.concatenate(rows, axis=0))


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.steps <= 0:
        raise ValueError("Batch size and steps must be positive")
    if not 1 <= args.refinement_steps <= 7:
        raise ValueError("Refinement steps must be in [1, 7]")
    data_dir = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "moge3-hypersim-smallset-v1":
        raise ValueError(f"Unsupported manifest: {manifest.get('format')}")
    model = MoGeModel.from_pretrained(args.pretrained).to(device).eval()
    samples = cache_base_predictions(
        data_dir,
        manifest,
        model,
        height=args.height,
        width=args.width,
        num_tokens=args.num_tokens,
        device=device,
        safe_root=args.safe_root,
    )
    train_samples = [sample for sample in samples if sample.split == "train"]
    val_samples = [sample for sample in samples if sample.split == "val"]
    if not train_samples or not val_samples:
        raise ValueError("Both train and val samples are required")

    refiner = model.ssr
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    refiner.to(device).train()
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    start_step = 0
    resume_path = None
    if args.resume is not None:
        resume_path = assert_safe_path(
            args.resume,
            safe_root=args.safe_root,
            must_exist=True,
        )
        checkpoint = torch.load(
            resume_path,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("pretrained") != args.pretrained:
            raise ValueError(
                "Checkpoint base model does not match --pretrained: "
                f"{checkpoint.get('pretrained')} != {args.pretrained}"
            )
        refiner.load_state_dict(checkpoint["ssr"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["step"])
        if args.steps <= start_step:
            raise ValueError(
                f"--steps must exceed resumed step {start_step}, got {args.steps}"
            )
    cpu_generator = torch.Generator().manual_seed(args.seed + 1)
    loss_generator = torch.Generator(device=device).manual_seed(args.seed + 2)

    baseline_records, baseline_predictions = evaluate(
        refiner,
        samples,
        device=device,
        refinement_steps=args.refinement_steps,
        boundary_threshold=args.boundary_threshold,
        use_refiner=False,
        batch_size=args.batch_size,
    )
    identity_error = zero_initialization_error(
        refiner,
        samples,
        device=device,
        refinement_steps=args.refinement_steps,
        batch_size=args.batch_size,
    )
    zero_records, _ = evaluate(
        refiner,
        samples,
        device=device,
        refinement_steps=args.refinement_steps,
        boundary_threshold=args.boundary_threshold,
        use_refiner=True,
        batch_size=args.batch_size,
    )
    baseline = {
        split: aggregate_metrics([record for record in baseline_records if record["split"] == split])
        for split in ("train", "val")
    }
    zero = {
        split: aggregate_metrics([record for record in zero_records if record["split"] == split])
        for split in ("train", "val")
    }

    losses: List[float] = []
    training_records: List[Dict[str, object]] = []
    eval_history: List[Dict[str, float]] = [
        {
            "step": start_step,
            **{
                f"{split}/{key}": value
                for split in ("train", "val")
                for key, value in zero[split].items()
            },
        }
    ]
    start_time = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        indices = torch.randint(
            len(train_samples),
            (args.batch_size,),
            generator=cpu_generator,
        ).tolist()
        batch = [train_samples[index] for index in indices]
        base = torch.stack([sample.base_points for sample in batch]).to(device)
        visual = torch.stack([sample.visual_features for sample in batch]).to(device)
        gt = torch.stack([sample.gt_points for sample in batch]).to(device)

        optimizer.zero_grad(set_to_none=True)
        sequence = refine_cached(refiner, base, visual, args.refinement_steps)
        loss, terms = geometry_loss(
            sequence,
            gt,
            global_weight=args.global_weight,
            local_weight=args.local_weight,
            edge_weight=args.edge_weight,
            local_scales=tuple(args.local_scales),
            generator=loss_generator,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        optimizer.step()
        loss_value = float(loss.detach())
        losses.append(loss_value)
        training_records.append(
            {
                "step": step,
                "loss": loss_value,
                "grad_norm": float(grad_norm),
                **terms,
            }
        )

        if step % args.eval_every == 0 or step == args.steps:
            evaluation, _ = evaluate(
                refiner,
                samples,
                device=device,
                refinement_steps=args.refinement_steps,
                boundary_threshold=args.boundary_threshold,
                use_refiner=True,
                batch_size=args.batch_size,
            )
            aggregate = {
                split: aggregate_metrics(
                    [record for record in evaluation if record["split"] == split]
                )
                for split in ("train", "val")
            }
            eval_history.append(
                {
                    "step": step,
                    **{
                        f"{split}/{key}": value
                        for split in ("train", "val")
                        for key, value in aggregate[split].items()
                    },
                }
            )

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_value,
                        "grad_norm": float(grad_norm),
                        "elapsed_seconds": time.perf_counter() - start_time,
                        **terms,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - start_time
    final_records, final_predictions = evaluate(
        refiner,
        samples,
        device=device,
        refinement_steps=args.refinement_steps,
        boundary_threshold=args.boundary_threshold,
        use_refiner=True,
        batch_size=args.batch_size,
    )
    final = {
        split: aggregate_metrics([record for record in final_records if record["split"] == split])
        for split in ("train", "val")
    }

    baseline_by_id = {record["id"]: record for record in baseline_records}
    final_by_id = {record["id"]: record for record in final_records}
    per_frame = []
    for sample in samples:
        before = baseline_by_id[sample.sample_id]
        after = final_by_id[sample.sample_id]
        per_frame.append(
            {
                "id": sample.sample_id,
                "split": sample.split,
                "scene": sample.scene,
                "frame": sample.frame,
                **{f"before_{key}": before[key] for key in ("point_rel", "depth_rel", "depth_delta_1.01", "depth_delta_1.25", "boundary_f1")},
                **{f"after_{key}": after[key] for key in ("point_rel", "depth_rel", "depth_delta_1.01", "depth_delta_1.25", "boundary_f1")},
                "depth_rel_improved": bool(after["depth_rel"] < before["depth_rel"]),
            }
        )

    save_csv(output / "training_history.csv", training_records)
    save_csv(output / "evaluation_history.csv", eval_history)
    save_csv(output / "per_frame_metrics.csv", per_frame)
    save_training_plot(
        output,
        losses,
        eval_history,
        baseline,
        args.loss_smoothing_window,
        start_step,
    )
    selected = [
        train_samples[0],
        train_samples[len(train_samples) // 2],
        val_samples[0],
        val_samples[-1],
    ]
    save_qualitative(output, selected, baseline_predictions, final_predictions)

    checkpoint = {
        "ssr": refiner.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.steps,
        "pretrained": args.pretrained,
        "args": vars(args),
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    train_improved = [
        record for record in per_frame
        if record["split"] == "train" and record["depth_rel_improved"]
    ]
    val_improved = [
        record for record in per_frame
        if record["split"] == "val" and record["depth_rel_improved"]
    ]
    report = {
        "status": "complete",
        "experiment": "Hypersim multi-scene small-set SSR overfit",
        "counts": manifest["counts"],
        "shape": [args.height, args.width],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pretrained": args.pretrained,
        "base_frozen": True,
        "normal_prediction": False,
        "refinement_steps": args.refinement_steps,
        "optimization_steps": args.steps,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "resumed_step": start_step,
        "batch_size": args.batch_size,
        "identity_max_error": identity_error if start_step == 0 else None,
        "initial_k3_max_delta": identity_error,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "baseline": baseline,
        "initial_k3": zero,
        "zero_initialized_k3": zero if start_step == 0 else None,
        "final": final,
        "train_depth_rel_reduction": 1.0 - final["train"]["depth_rel"] / baseline["train"]["depth_rel"],
        "val_depth_rel_reduction": 1.0 - final["val"]["depth_rel"] / baseline["val"]["depth_rel"],
        "train_fraction_improved": len(train_improved) / len(train_samples),
        "val_fraction_improved": len(val_improved) / len(val_samples),
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / len(losses),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "boundary_definition": {
            "log_depth_jump_threshold": args.boundary_threshold,
            "matching_radius_pixels": 1,
        },
        "artifacts": {
            "checkpoint": "checkpoint.pt",
            "training_history": "training_history.csv",
            "evaluation_history": "evaluation_history.csv",
            "per_frame_metrics": "per_frame_metrics.csv",
            "training_curves": "training_curves.png",
            "qualitative_comparison": "qualitative_comparison.png",
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
