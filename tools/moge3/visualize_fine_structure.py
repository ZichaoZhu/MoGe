from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from moge.model.v3 import MoGeModel
from moge.scripts.train_hypersim_smallset_v3 import (
    CachedSample,
    boundary_f1,
    cache_base_predictions,
    refine_cached,
)
from moge.train.losses_v3 import solve_global_affine_alignment
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


@dataclass
class SelectedRegion:
    sample: CachedSample
    crop: Tuple[int, int, int, int]
    fine_mask: np.ndarray
    aligned_points: Dict[int, np.ndarray]
    metrics: Dict[int, Dict[str, float]]
    selection_score: float
    fine_pixels: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create paper-style MoGe-3 fine-structure 3D visualizations"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--max-k", type=int, default=5)
    parser.add_argument("--crop-height", type=int, default=96)
    parser.add_argument("--crop-width", type=int, default=96)
    parser.add_argument("--crop-stride", type=int, default=16)
    parser.add_argument("--min-fine-pixels", type=int, default=24)
    parser.add_argument("--yaw", type=float, default=28.0)
    parser.add_argument("--pitch", type=float, default=-8.0)
    parser.add_argument("--splits", nargs="+", default=["train", "val"])
    parser.add_argument("--orbit-splits", nargs="*", default=["train"])
    parser.add_argument("--train-sample-id")
    parser.add_argument("--val-sample-id")
    parser.add_argument("--test-sample-id")
    parser.add_argument(
        "--selection-file",
        type=Path,
        help="Optional pre-training JSON containing split-specific sample ids",
    )
    parser.add_argument(
        "--selection-provenance",
        choices=("pretraining_locked", "posthoc_rgb_gt_only", "posthoc_diagnostic"),
        help=(
            "How fixed samples were selected. Defaults to pretraining_locked when "
            "--selection-file is provided, otherwise posthoc_diagnostic."
        ),
    )
    parser.add_argument("--seed", type=int, default=29)
    return parser.parse_args()


def selection_description(
    split: str,
    *,
    has_fixed_sample: bool,
    provenance: str,
) -> str:
    if has_fixed_sample and provenance == "pretraining_locked":
        return (
            "fixed sample locked before training using RGB/ground truth only; "
            "within-sample window "
            + (
                "maximizes K=0 to K=3 point-Rel reduction"
                if split == "train"
                else "maximizes ground-truth fine-detail proposal support"
            )
        )
    if has_fixed_sample and provenance == "posthoc_rgb_gt_only":
        if split == "train":
            return (
                "post-hoc training illustration: sample chosen using RGB/ground "
                "truth only; within-sample window maximizes K=0 to K=3 point-Rel "
                "reduction"
            )
        return (
            "post-hoc qualitative sample and within-sample window chosen using "
            "RGB/ground truth only; model predictions were not used for selection"
        )
    if has_fixed_sample:
        return (
            "post-hoc diagnostic fixed sample; within-sample window "
            + (
                "maximizes K=0 to K=3 point-Rel reduction"
                if split == "train"
                else "maximizes ground-truth fine-detail proposal support"
            )
        )
    if split == "train":
        return "largest K=0 to K=3 point-Rel reduction among fine-detail windows"
    return (
        "largest ground-truth fine-detail proposal support, independent of "
        "prediction quality"
    )


def threshold_residual(residual: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = residual[valid].astype(np.float64)
    if values.size == 0:
        return np.zeros_like(valid)
    median = float(np.median(values))
    sigma = 1.4826 * float(np.median(np.abs(values - median)))
    tolerance = max(1e-8, 1e-6 * float(np.max(np.abs(values))))
    if sigma > tolerance:
        threshold = max(3.0 * sigma, median + tolerance)
    else:
        tail = values[values > median + tolerance]
        if tail.size == 0:
            return np.zeros_like(valid)
        tail_median = float(np.median(tail))
        tail_sigma = 1.4826 * float(np.median(np.abs(tail - tail_median)))
        threshold = median + max(3.0 * tail_sigma, tolerance)
    return (residual > threshold) & valid


def paper_coarse_fine_mask(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """
    Reproduce the coarse stage of paper Appendix B.1.

    SAM2 segment-aware expansion is intentionally left out because Hypersim
    visualization crops only need deterministic structure proposals. The
    resulting mask must not be reported as the paper's benchmark mask.
    """
    height, width = depth.shape
    safe_depth = np.where(valid, np.clip(depth, 1e-6, None), np.nan)
    finite_values = safe_depth[np.isfinite(safe_depth)]
    if finite_values.size == 0:
        return np.zeros_like(valid)
    fill_depth = float(np.median(finite_values))
    disparity = np.where(valid, 1.0 / safe_depth, 1.0 / fill_depth).astype(np.float32)

    mask = np.zeros_like(valid)
    for scale in (8, 16, 32):
        reduced_width = max(1, width // scale)
        reduced_height = max(1, height // scale)
        if min(reduced_width, reduced_height) < 4:
            continue
        small = cv2.resize(
            disparity,
            (reduced_width, reduced_height),
            interpolation=cv2.INTER_AREA,
        )
        restored = cv2.resize(
            small,
            (width, height),
            interpolation=cv2.INTER_LINEAR,
        )
        mask |= threshold_residual(np.abs(disparity - restored), valid)

    for size in (3, 5, 9, 17):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        opened = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(disparity, cv2.MORPH_CLOSE, kernel)
        mask |= threshold_residual(np.clip(disparity - opened, 0, None), valid)
        mask |= threshold_residual(np.clip(closed - disparity, 0, None), valid)
    return mask & valid


def sliding_positions(length: int, window: int, stride: int) -> List[int]:
    if window >= length:
        return [0]
    positions = list(range(0, length - window + 1, stride))
    if positions[-1] != length - window:
        positions.append(length - window)
    return positions


def select_crop(
    fine_mask: np.ndarray,
    base_error: np.ndarray,
    refined_error: np.ndarray,
    *,
    crop_height: int,
    crop_width: int,
    stride: int,
    min_fine_pixels: int,
    mode: str,
) -> Tuple[Tuple[int, int, int, int], float, int]:
    height, width = fine_mask.shape
    crop_height = min(crop_height, height)
    crop_width = min(crop_width, width)
    best = None
    for y0 in sliding_positions(height, crop_height, stride):
        for x0 in sliding_positions(width, crop_width, stride):
            y1, x1 = y0 + crop_height, x0 + crop_width
            local = fine_mask[y0:y1, x0:x1]
            count = int(local.sum())
            if count < min_fine_pixels:
                continue
            if mode == "improvement":
                before = float(base_error[y0:y1, x0:x1][local].mean())
                after = float(refined_error[y0:y1, x0:x1][local].mean())
                score = (before - after) * math.sqrt(count)
            elif mode == "structure":
                score = float(count)
            else:
                raise ValueError(f"Unsupported crop-selection mode: {mode}")
            candidate = (score, count, (x0, y0, x1, y1))
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    if best is None:
        x0 = max(0, (width - crop_width) // 2)
        y0 = max(0, (height - crop_height) // 2)
        return (x0, y0, x0 + crop_width, y0 + crop_height), 0.0, 0
    return best[2], float(best[0]), int(best[1])


def align_points(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    alignment = solve_global_affine_alignment(
        pred_points[None],
        gt_points[None],
    )
    return alignment.apply(pred_points[None])[0]


def relative_point_error(pred_points: np.ndarray, gt_points: np.ndarray) -> np.ndarray:
    valid = np.isfinite(gt_points).all(axis=-1)
    safe_gt = np.where(valid[..., None], gt_points, 1.0)
    error = np.linalg.norm(pred_points - safe_gt, axis=-1)
    error /= np.clip(safe_gt[..., 2], 1e-6, None)
    error[~valid] = np.nan
    return error


def region_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    fine_mask: np.ndarray,
    crop: Tuple[int, int, int, int],
    boundary_threshold: float = 0.03,
) -> Dict[str, float]:
    x0, y0, x1, y1 = crop
    pred = pred_points[y0:y1, x0:x1]
    gt = gt_points[y0:y1, x0:x1]
    selected = fine_mask[y0:y1, x0:x1]
    valid = np.isfinite(gt).all(axis=-1) & selected
    if not valid.any():
        return {
            "fine_pixels": 0,
            "point_rel": float("nan"),
            "depth_rel": float("nan"),
            "depth_delta_1.01": float("nan"),
            "boundary_f1": float("nan"),
        }
    safe_gt = np.where(valid[..., None], gt, 1.0)
    point_rel = np.linalg.norm(pred - safe_gt, axis=-1)
    point_rel /= np.clip(safe_gt[..., 2], 1e-6, None)
    depth_rel = np.abs(pred[..., 2] - safe_gt[..., 2])
    depth_rel /= np.clip(safe_gt[..., 2], 1e-6, None)
    ratio = np.maximum(
        pred[..., 2] / np.clip(safe_gt[..., 2], 1e-6, None),
        safe_gt[..., 2] / np.clip(pred[..., 2], 1e-6, None),
    )
    boundary_valid = np.isfinite(gt).all(axis=-1)
    return {
        "fine_pixels": int(valid.sum()),
        "point_rel": float(point_rel[valid].mean()),
        "depth_rel": float(depth_rel[valid].mean()),
        "depth_delta_1.01": float((ratio[valid] < 1.01).mean()),
        "boundary_f1": boundary_f1(
            torch.from_numpy(pred[..., 2]),
            torch.from_numpy(gt[..., 2]),
            torch.from_numpy(boundary_valid),
            boundary_threshold,
        ),
    }


def rotation_matrix(yaw_degrees: float, pitch_degrees: float) -> np.ndarray:
    yaw = math.radians(yaw_degrees)
    pitch = math.radians(pitch_degrees)
    rotate_y = np.asarray(
        [
            [math.cos(yaw), 0.0, math.sin(yaw)],
            [0.0, 1.0, 0.0],
            [-math.sin(yaw), 0.0, math.cos(yaw)],
        ],
        dtype=np.float32,
    )
    rotate_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(pitch), -math.sin(pitch)],
            [0.0, math.sin(pitch), math.cos(pitch)],
        ],
        dtype=np.float32,
    )
    return rotate_x @ rotate_y


def project_cloud(
    points: np.ndarray,
    colors: np.ndarray,
    valid: np.ndarray,
    center: np.ndarray,
    yaw: float,
    pitch: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    values = points[valid] - center
    projected = values @ rotation_matrix(yaw, pitch).T
    color_values = colors[valid]
    return projected[:, 0], -projected[:, 1], projected[:, 2], color_values


def projection_bounds(
    projections: Sequence[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Tuple[float, float, float, float]:
    x = np.concatenate([projection[0] for projection in projections])
    y = np.concatenate([projection[1] for projection in projections])
    x0, x1 = np.percentile(x, (0.5, 99.5))
    y0, y1 = np.percentile(y, (0.5, 99.5))
    width = max(float(x1 - x0), 1e-6)
    height = max(float(y1 - y0), 1e-6)
    side = 1.08 * max(width, height)
    cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    return cx - side / 2, cx + side / 2, cy - side / 2, cy + side / 2


def draw_cloud(
    axis,
    projection: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    bounds: Tuple[float, float, float, float],
) -> None:
    x, y, z, colors = projection
    order = np.argsort(z)[::-1]
    axis.scatter(
        x[order],
        y[order],
        c=colors[order],
        s=2.2,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xlim(bounds[0], bounds[1])
    axis.set_ylim(bounds[2], bounds[3])
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_facecolor("white")


def crop_arrays(
    region: SelectedRegion,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray], np.ndarray]:
    x0, y0, x1, y1 = region.crop
    image = region.sample.image.permute(1, 2, 0).numpy()[y0:y1, x0:x1]
    gt = region.sample.gt_points.numpy()[y0:y1, x0:x1]
    predictions = {
        step: points[y0:y1, x0:x1]
        for step, points in region.aligned_points.items()
    }
    fine_mask = region.fine_mask[y0:y1, x0:x1]
    return image, gt, predictions, fine_mask


def save_iteration_panel(
    output: Path,
    region: SelectedRegion,
    *,
    yaw: float,
    pitch: float,
    steps: Sequence[int] = (0, 1, 3, 5),
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    image, gt, predictions, _ = crop_arrays(region)
    columns = [("GT", gt), *[(f"K={step}", predictions[step]) for step in steps]]
    figure, axes = plt.subplots(2, len(columns) + 1, figsize=(19, 7.5))

    axes[0, 0].imshow(image)
    axes[0, 0].set_title("RGB crop")
    axes[0, 0].axis("off")
    full_image = region.sample.image.permute(1, 2, 0).numpy()
    x0, y0, x1, y1 = region.crop
    axes[1, 0].imshow(full_image)
    axes[1, 0].add_patch(
        Rectangle(
            (x0, y0),
            x1 - x0,
            y1 - y0,
            fill=False,
            edgecolor="red",
            linewidth=2.0,
            linestyle="--",
        )
    )
    axes[1, 0].set_title("Selected thin-structure ROI")
    axes[1, 0].axis("off")

    gt_valid = np.isfinite(gt).all(axis=-1)
    gt_disparity = np.where(gt_valid, 1.0 / np.clip(gt[..., 2], 1e-6, None), np.nan)
    disparity_values = gt_disparity[gt_valid]
    disparity_min, disparity_max = np.percentile(disparity_values, (2, 98))

    center = np.nanmedian(gt.reshape(-1, 3), axis=0)
    colors = image.reshape(image.shape)
    projections = []
    for _, points in columns:
        valid = gt_valid & np.isfinite(points).all(axis=-1)
        projections.append(
            project_cloud(points, colors, valid, center, yaw, pitch)
        )
    bounds = projection_bounds(projections)

    for column_index, ((label, points), projection) in enumerate(
        zip(columns, projections),
        start=1,
    ):
        valid = gt_valid & np.isfinite(points).all(axis=-1)
        disparity = np.where(
            valid,
            1.0 / np.clip(points[..., 2], 1e-6, None),
            np.nan,
        )
        axes[0, column_index].imshow(
            disparity,
            cmap="turbo",
            vmin=disparity_min,
            vmax=disparity_max,
        )
        if label == "GT":
            title = "GT disparity"
        else:
            step = int(label.split("=")[1])
            title = (
                f"{label} disparity\n"
                f"proposal depth Rel {100 * region.metrics[step]['depth_rel']:.2f}%"
            )
        axes[0, column_index].set_title(title)
        axes[0, column_index].axis("off")

        draw_cloud(axes[1, column_index], projection, bounds)
        if label == "GT":
            title = f"GT novel view\nyaw {yaw:.0f}°"
        else:
            step = int(label.split("=")[1])
            title = (
                f"{label} novel view\n"
                f"proposal point Rel {100 * region.metrics[step]['point_rel']:.2f}%"
            )
        axes[1, column_index].set_title(title)

    figure.suptitle(
        f"{region.sample.split}: {region.sample.sample_id} | "
        f"crop=({x0},{y0})-({x1},{y1})",
        fontsize=15,
    )
    figure.tight_layout()
    stem = f"{region.sample.split}_fine_structure"
    figure.savefig(output / f"{stem}.png", dpi=180)
    figure.savefig(output / f"{stem}.pdf")
    plt.close(figure)


def orbit_frame(
    region: SelectedRegion,
    yaw: float,
    pitch: float,
    steps: Sequence[int] = (0, 3),
) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    image, gt, predictions, _ = crop_arrays(region)
    columns = [("GT", gt), *[(f"K={step}", predictions[step]) for step in steps]]
    valid = np.isfinite(gt).all(axis=-1)
    center = np.nanmedian(gt.reshape(-1, 3), axis=0)
    projections = [
        project_cloud(points, image, valid & np.isfinite(points).all(axis=-1), center, yaw, pitch)
        for _, points in columns
    ]
    bounds = projection_bounds(projections)

    figure, axes = plt.subplots(1, len(columns), figsize=(10.5, 3.6))
    for axis, (label, _), projection in zip(axes, columns, projections):
        draw_cloud(axis, projection, bounds)
        axis.set_title(label)
    figure.suptitle(
        f"{region.sample.sample_id} | novel-view yaw {yaw:+.0f}°",
        fontsize=13,
    )
    figure.tight_layout()
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba).convert("RGB")


def save_orbit_gif(
    output: Path,
    region: SelectedRegion,
    *,
    pitch: float,
) -> None:
    yaw_values = list(np.linspace(-35.0, 35.0, 15))
    frames = [
        orbit_frame(region, float(yaw), pitch, steps=(0, 3))
        for yaw in yaw_values
    ]
    frames += list(reversed(frames[1:-1]))
    frames[0].save(
        output / f"{region.sample.split}_fine_structure_orbit.gif",
        save_all=True,
        append_images=frames[1:],
        duration=110,
        loop=0,
        optimize=True,
    )


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


def evaluate_sample(
    sample: CachedSample,
    refiner,
    *,
    device: torch.device,
    max_k: int,
    crop_height: int,
    crop_width: int,
    crop_stride: int,
    min_fine_pixels: int,
) -> Tuple[SelectedRegion, Dict[str, object]]:
    base = sample.base_points[None].to(device)
    visual = sample.visual_features[None].to(device)
    gt = sample.gt_points.to(device)
    refined = refine_cached(refiner, base, visual, max_k)
    raw_points = {0: base[0], **{step: refined[step - 1][0] for step in range(1, max_k + 1)}}
    aligned = {
        step: align_points(points, gt).cpu().numpy()
        for step, points in raw_points.items()
    }
    gt_numpy = gt.cpu().numpy()
    valid = np.isfinite(gt_numpy).all(axis=-1)
    fine_mask = paper_coarse_fine_mask(gt_numpy[..., 2], valid)
    base_error = relative_point_error(aligned[0], gt_numpy)
    refined_error = relative_point_error(aligned[min(3, max_k)], gt_numpy)
    mode = "improvement" if sample.split == "train" else "structure"
    crop, score, fine_pixels = select_crop(
        fine_mask,
        base_error,
        refined_error,
        crop_height=crop_height,
        crop_width=crop_width,
        stride=crop_stride,
        min_fine_pixels=min_fine_pixels,
        mode=mode,
    )
    metrics = {
        step: region_metrics(points, gt_numpy, fine_mask, crop)
        for step, points in aligned.items()
    }
    region = SelectedRegion(
        sample=sample,
        crop=crop,
        fine_mask=fine_mask,
        aligned_points=aligned,
        metrics=metrics,
        selection_score=score,
        fine_pixels=fine_pixels,
    )
    x0, y0, x1, y1 = crop
    record = {
        "id": sample.sample_id,
        "split": sample.split,
        "scene": sample.scene,
        "frame": sample.frame,
        "selection_mode": mode,
        "selection_score": score,
        "crop_x0": x0,
        "crop_y0": y0,
        "crop_x1": x1,
        "crop_y1": y1,
        "fine_pixels": fine_pixels,
        "k0_point_rel": metrics[0]["point_rel"],
        "k3_point_rel": metrics[min(3, max_k)]["point_rel"],
        "k0_depth_rel": metrics[0]["depth_rel"],
        "k3_depth_rel": metrics[min(3, max_k)]["depth_rel"],
        "point_rel_reduction": 1.0
        - metrics[min(3, max_k)]["point_rel"] / metrics[0]["point_rel"],
        "depth_rel_reduction": 1.0
        - metrics[min(3, max_k)]["depth_rel"] / metrics[0]["depth_rel"],
    }
    return region, record


def selected_report(region: SelectedRegion) -> Dict[str, object]:
    return {
        "id": region.sample.sample_id,
        "split": region.sample.split,
        "scene": region.sample.scene,
        "frame": region.sample.frame,
        "crop": list(region.crop),
        "fine_pixels": region.fine_pixels,
        "selection_score": region.selection_score,
        "metrics": {str(step): values for step, values in region.metrics.items()},
    }


def main() -> None:
    args = parse_args()
    if not 3 <= args.max_k <= 7:
        raise ValueError("--max-k must be in [3, 7]")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("--splits must not contain duplicates")
    supported_splits = {"train", "val", "test"}
    if not args.splits or not set(args.splits) <= supported_splits:
        raise ValueError(f"--splits must be drawn from {sorted(supported_splits)}")
    if not set(args.orbit_splits) <= set(args.splits):
        raise ValueError("--orbit-splits must be a subset of --splits")
    selection_provenance = args.selection_provenance
    if selection_provenance is None:
        selection_provenance = (
            "pretraining_locked"
            if args.selection_file is not None
            else "posthoc_diagnostic"
        )
    if args.selection_file is not None and selection_provenance != "pretraining_locked":
        raise ValueError(
            "--selection-file can only be used with pretraining_locked provenance"
        )
    if args.selection_file is None and selection_provenance == "pretraining_locked":
        raise ValueError(
            "pretraining_locked provenance requires a verified --selection-file"
        )
    data_dir = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    checkpoint_path = assert_safe_path(
        args.checkpoint,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.init()

    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if checkpoint.get("pretrained") != args.pretrained:
        raise ValueError(
            f"Checkpoint uses {checkpoint.get('pretrained')}, got {args.pretrained}"
        )

    model = MoGeModel.from_pretrained(args.pretrained).to(device).eval()
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"], strict=True)
    else:
        model.ssr.load_state_dict(checkpoint["ssr"], strict=True)
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
    refiner = model.ssr.eval()

    records: List[Dict[str, object]] = []
    selected: Dict[str, SelectedRegion] = {}
    selected_ids = {
        "train": args.train_sample_id,
        "val": args.val_sample_id,
        "test": args.test_sample_id,
    }
    if args.selection_file is not None:
        selection_path = assert_safe_path(
            args.selection_file,
            safe_root=args.safe_root,
            must_exist=True,
        )
        locked_selection = json.loads(selection_path.read_text(encoding="utf-8"))
        if locked_selection.get("status") != "locked-before-training":
            raise ValueError("Selection file was not locked before training")
        for split in args.splits:
            locked_id = locked_selection.get(split, {}).get("id")
            if not locked_id:
                raise ValueError(f"Selection file has no fixed sample for {split}")
            if selected_ids[split] is not None and selected_ids[split] != locked_id:
                raise ValueError(f"Conflicting fixed sample ids for {split}")
            selected_ids[split] = locked_id
    with torch.inference_mode():
        for sample in samples:
            if sample.split not in args.splits:
                continue
            selected_id = selected_ids[sample.split]
            if selected_id is not None and sample.sample_id != selected_id:
                continue
            region, record = evaluate_sample(
                sample,
                refiner,
                device=device,
                max_k=args.max_k,
                crop_height=args.crop_height,
                crop_width=args.crop_width,
                crop_stride=args.crop_stride,
                min_fine_pixels=args.min_fine_pixels,
            )
            records.append(record)
            current = selected.get(sample.split)
            if current is None or region.selection_score > current.selection_score:
                selected[sample.split] = region

    missing = [split for split in args.splits if split not in selected]
    if missing:
        raise RuntimeError(f"No eligible visualization sample for splits: {missing}")
    save_csv(output / "candidate_regions.csv", records)
    for split in args.splits:
        save_iteration_panel(
            output,
            selected[split],
            yaw=args.yaw,
            pitch=args.pitch,
        )
        if split in args.orbit_splits:
            save_orbit_gif(output, selected[split], pitch=args.pitch)

    report = {
        "status": "complete",
        "experiment": "MoGe-3 fine-structure 3D visualization",
        "paper_basis": {
            "figure_1_and_3": "off-axis point-map views reveal distortions hidden in 2D depth",
            "figure_4": "K=0/1/2/3/5 iteration comparison on thin structures",
            "appendix_b1": "coarse multi-scale residual and morphology proposal",
        },
        "mask_scope": (
            "Appendix B.1 coarse proposal only; SAM2 segment-aware expansion is "
            "omitted, so these are diagnostic crops rather than paper benchmark masks."
        ),
        "selection_provenance": selection_provenance,
        "selection": {
            split: selection_description(
                split,
                has_fixed_sample=selected_ids[split] is not None,
                provenance=selection_provenance,
            )
            for split in args.splits
        },
        "shape": [args.height, args.width],
        "max_k": args.max_k,
        "yaw_degrees": args.yaw,
        "pitch_degrees": args.pitch,
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        **{
            split: selected_report(selected[split])
            for split in args.splits
        },
        "artifacts": {
            "candidates": "candidate_regions.csv",
            **{
                f"{split}_panel": f"{split}_fine_structure.png"
                for split in args.splits
            },
            **{
                f"{split}_panel_pdf": f"{split}_fine_structure.pdf"
                for split in args.splits
            },
            **{
                f"{split}_orbit": f"{split}_fine_structure_orbit.gif"
                for split in args.orbit_splits
            },
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
