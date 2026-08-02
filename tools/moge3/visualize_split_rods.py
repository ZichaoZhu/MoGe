from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


Crop = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SeriesSpec:
    label: str
    checkpoint: Path
    refinement_steps: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render thin-structure point-cloud GIFs from each experiment's own "
            "train/val/test splits"
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument(
        "--series",
        nargs=3,
        action="append",
        required=True,
        metavar=("LABEL", "CHECKPOINT", "K"),
    )
    parser.add_argument("--experiment-label", required=True)
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num-tokens", type=int, required=True)
    parser.add_argument("--mask-dilation", type=int, default=2)
    parser.add_argument("--yaw-min", type=float, default=-40.0)
    parser.add_argument("--yaw-max", type=float, default=40.0)
    parser.add_argument("--orbit-frames", type=int, default=17)
    parser.add_argument("--pitch", type=float, default=-8.0)
    parser.add_argument(
        "--motion",
        choices=("ping-pong", "one-way"),
        default="ping-pong",
    )
    parser.add_argument("--frame-duration-ms", type=int, default=110)
    parser.add_argument("--seed", type=int, default=83)
    return parser.parse_args()


def normalize_manifest(manifest: Mapping[str, object]) -> Dict[str, object]:
    samples = list(manifest.get("samples", []))
    if not samples:
        raise ValueError("manifest contains no samples")
    if all(
        all(key in sample for key in ("id", "split", "scene", "M_cam_from_uv"))
        for sample in samples
    ):
        return dict(manifest)
    if len(samples) != 1:
        raise ValueError("Only a one-sample legacy manifest can be normalized")
    sample = dict(samples[0])
    scene = str(manifest["scene"])
    camera = str(manifest["camera"])
    frame = int(sample["frame"])
    sample.update(
        {
            "id": f"{scene}_{camera}_frame.{frame:04d}",
            "split": "train",
            "scene": scene,
            "camera": camera,
            "M_cam_from_uv": manifest["M_cam_from_uv"],
        }
    )
    return {**manifest, "samples": [sample]}


def validate_crop(crop: Sequence[int], height: int, width: int) -> Crop:
    if len(crop) != 4:
        raise ValueError("crop must contain x0 y0 x1 y1")
    x0, y0, x1, y1 = (int(value) for value in crop)
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(
            f"crop {(x0, y0, x1, y1)} is outside image shape {(height, width)}"
        )
    return x0, y0, x1, y1


def selection_entries(
    selection: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    height: int,
    width: int,
) -> Dict[str, Dict[str, object]]:
    if selection.get("status") != "locked-before-rendering":
        raise ValueError("selection must be locked-before-rendering")
    if selection.get("selection_inputs") != "RGB and ground truth only; no predictions":
        raise ValueError("selection provenance must exclude model predictions")
    raw_entries = selection.get("splits")
    if not isinstance(raw_entries, dict) or not raw_entries:
        raise ValueError("selection must contain a non-empty splits mapping")
    manifest_by_id = {
        str(sample["id"]): sample for sample in manifest["samples"]
    }
    validated: Dict[str, Dict[str, object]] = {}
    for split, raw_entry in raw_entries.items():
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unsupported split in selection: {split}")
        if not isinstance(raw_entry, dict):
            raise ValueError(f"Selection entry for {split} must be an object")
        sample_id = str(raw_entry["id"])
        if sample_id not in manifest_by_id:
            raise ValueError(f"Selected sample is absent from manifest: {sample_id}")
        sample = manifest_by_id[sample_id]
        if str(sample["split"]) != split:
            raise ValueError(
                f"Selected sample {sample_id} belongs to {sample['split']}, not {split}"
            )
        crop = validate_crop(raw_entry["crop_xyxy"], height, width)
        validated[split] = {
            **raw_entry,
            "id": sample_id,
            "crop_xyxy": list(crop),
        }
    return validated


def parse_series(raw_series: Sequence[Sequence[str]]) -> List[SeriesSpec]:
    labels = set()
    parsed = []
    for label, checkpoint, raw_k in raw_series:
        if not label or label in labels:
            raise ValueError("Series labels must be non-empty and unique")
        labels.add(label)
        refinement_steps = int(raw_k)
        if not 1 <= refinement_steps <= 7:
            raise ValueError("Every series K must be in [1, 7]")
        parsed.append(
            SeriesSpec(
                label=label,
                checkpoint=Path(checkpoint),
                refinement_steps=refinement_steps,
            )
        )
    return parsed


def masked_relative_metrics(
    pred_points: np.ndarray,
    gt_points: np.ndarray,
    mask: np.ndarray,
) -> Dict[str, float]:
    valid = (
        mask
        & np.isfinite(pred_points).all(axis=-1)
        & np.isfinite(gt_points).all(axis=-1)
        & (gt_points[..., 2] > 0)
    )
    if not valid.any():
        raise ValueError("display mask contains no valid predicted points")
    denominator = np.clip(gt_points[..., 2], 1e-6, None)
    point_rel = np.linalg.norm(pred_points - gt_points, axis=-1) / denominator
    depth_rel = np.abs(pred_points[..., 2] - gt_points[..., 2]) / denominator
    return {
        "pixels": int(valid.sum()),
        "point_rel": float(point_rel[valid].mean()),
        "depth_rel": float(depth_rel[valid].mean()),
    }


def build_display_mask(
    gt_points: np.ndarray,
    mask_config: Mapping[str, object],
) -> Tuple[np.ndarray, Dict[str, object]]:
    valid = (
        np.isfinite(gt_points).all(axis=-1)
        & (gt_points[..., 2] > 0)
    )
    values = gt_points[..., 2][valid]
    if values.size == 0:
        raise ValueError("crop contains no valid ground-truth depth")
    mask_type = str(mask_config.get("type"))
    if mask_type == "near_quantile":
        quantile = float(mask_config["quantile"])
        if not 0 < quantile < 1:
            raise ValueError("near_quantile must be in (0, 1)")
        maximum_depth = float(np.quantile(values, quantile))
        mask = valid & (gt_points[..., 2] <= maximum_depth)
        resolved = {
            "type": mask_type,
            "quantile": quantile,
            "resolved_maximum_depth_m": maximum_depth,
        }
    elif mask_type == "depth_range":
        minimum_depth = float(mask_config.get("minimum_depth_m", 0.0))
        maximum_depth = float(mask_config["maximum_depth_m"])
        if not 0 <= minimum_depth < maximum_depth:
            raise ValueError("invalid depth_range limits")
        mask = (
            valid
            & (gt_points[..., 2] >= minimum_depth)
            & (gt_points[..., 2] <= maximum_depth)
        )
        resolved = {
            "type": mask_type,
            "minimum_depth_m": minimum_depth,
            "maximum_depth_m": maximum_depth,
        }
    else:
        raise ValueError(f"Unsupported GT-only display mask: {mask_type}")
    return mask, resolved


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


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
    return projected[:, 0], -projected[:, 1], projected[:, 2], colors[valid]


def draw_cloud(axis, projection, radius: float) -> None:
    x, y, z, colors = projection
    order = np.argsort(z)[::-1]
    axis.scatter(
        x[order],
        y[order],
        c=colors[order],
        s=3.0,
        linewidths=0,
        rasterized=True,
    )
    axis.set_xlim(-radius, radius)
    axis.set_ylim(-radius, radius)
    axis.set_aspect("equal")
    axis.axis("off")
    axis.set_facecolor("white")


def highlighted_rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    highlighted = np.clip(image * 0.22 + 0.04, 0.0, 1.0)
    highlighted[mask] = image[mask]
    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    eroded = np.ones_like(mask)
    for offset_y in range(3):
        for offset_x in range(3):
            eroded &= padded[
                offset_y : offset_y + mask.shape[0],
                offset_x : offset_x + mask.shape[1],
            ]
    highlighted[mask & ~eroded] = np.asarray(
        [1.0, 0.12, 0.04],
        dtype=np.float32,
    )
    return highlighted


def source_rgb_with_crop(image: np.ndarray, crop: Crop) -> np.ndarray:
    marked = image.copy()
    x0, y0, x1, y1 = crop
    thickness = max(2, round(min(image.shape[:2]) / 128))
    red = np.asarray([1.0, 0.08, 0.03], dtype=np.float32)
    marked[y0 : min(y0 + thickness, y1), x0:x1] = red
    marked[max(y1 - thickness, y0) : y1, x0:x1] = red
    marked[y0:y1, x0 : min(x0 + thickness, x1)] = red
    marked[y0:y1, max(x1 - thickness, x0) : x1] = red
    return marked


def render_frame(
    *,
    experiment_label: str,
    split: str,
    full_image: np.ndarray,
    crop: Crop,
    image: np.ndarray,
    mask: np.ndarray,
    gt_points: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, float]],
    center: np.ndarray,
    radius: float,
    yaw: float,
    pitch: float,
) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    columns = [("GT", gt_points), *predictions.items()]
    figure, axes = plt.subplots(
        1,
        len(columns) + 2,
        figsize=(3.3 * (len(columns) + 2), 3.65),
    )
    axes[0].imshow(source_rgb_with_crop(full_image, crop))
    axes[0].set_title(f"{split} source RGB\nred box = selected crop")
    axes[0].axis("off")
    axes[1].imshow(highlighted_rgb(image, mask))
    axes[1].set_title(f"selected crop / GT-only mask\n{int(mask.sum())} pixels")
    axes[1].axis("off")
    for axis, (label, points) in zip(axes[2:], columns):
        valid = mask & np.isfinite(points).all(axis=-1)
        draw_cloud(
            axis,
            project_cloud(points, image, valid, center, yaw, pitch),
            radius,
        )
        if label == "GT":
            axis.set_title("Ground truth")
        else:
            axis.set_title(
                f"{label}\npoint Rel {100 * metrics[label]['point_rel']:.2f}%"
            )
    figure.suptitle(
        f"{experiment_label} | {split} thin structure | yaw {yaw:+.0f}°",
        fontsize=13,
        y=0.985,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.89))
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba).convert("RGB")


def save_preview(frames: Sequence[Image.Image], output: Path) -> None:
    selected = [frames[0], frames[len(frames) // 2], frames[-1]]
    target_width = min(1800, selected[0].width)
    resized = []
    for frame in selected:
        target_height = round(frame.height * target_width / frame.width)
        resized.append(
            frame.resize(
                (target_width, target_height),
                Image.Resampling.LANCZOS,
            )
        )
    canvas = Image.new(
        "RGB",
        (target_width, sum(frame.height for frame in resized)),
        "white",
    )
    y = 0
    for frame in resized:
        canvas.paste(frame, (0, y))
        y += frame.height
    canvas.save(output, optimize=True)


def align_points(pred_points: torch.Tensor, gt_points: torch.Tensor) -> torch.Tensor:
    from moge.train.losses_v3 import solve_global_affine_alignment

    alignment = solve_global_affine_alignment(
        pred_points[None],
        gt_points[None],
    )
    return alignment.apply(pred_points[None])[0]


def main() -> None:
    from moge.model.v3 import MoGeModel
    from moge.scripts.train_hypersim_smallset_v3 import (
        cache_base_predictions,
        refine_cached,
    )
    args = parse_args()
    if args.height <= 0 or args.width <= 0 or args.num_tokens <= 0:
        raise ValueError("shape and token arguments must be positive")
    if args.mask_dilation < 0:
        raise ValueError("--mask-dilation must be non-negative")
    if (
        args.orbit_frames < 3
        or args.yaw_min >= args.yaw_max
        or args.frame_duration_ms <= 0
    ):
        raise ValueError("invalid orbit range")
    series = parse_series(args.series)

    data_dir = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    selection_path = assert_safe_path(
        args.selection,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    validated_series = [
        SeriesSpec(
            label=spec.label,
            checkpoint=assert_safe_path(
                spec.checkpoint,
                safe_root=args.safe_root,
                must_exist=True,
            ),
            refinement_steps=spec.refinement_steps,
        )
        for spec in series
    ]

    manifest_path = assert_safe_path(
        data_dir / "manifest.json",
        safe_root=args.safe_root,
        must_exist=True,
    )
    manifest = normalize_manifest(
        json.loads(manifest_path.read_text(encoding="utf-8"))
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    entries = selection_entries(
        selection,
        manifest,
        height=args.height,
        width=args.width,
    )
    selected_ids = {entry["id"] for entry in entries.values()}
    selected_manifest = {
        **manifest,
        "samples": [
            sample
            for sample in manifest["samples"]
            if sample["id"] in selected_ids
        ],
    }
    if len(selected_manifest["samples"]) != len(entries):
        raise ValueError("Selections must map one-to-one to manifest samples")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.init()

    samples_by_split = {}
    aligned_by_split: Dict[str, Dict[str, np.ndarray]] = {
        split: {} for split in entries
    }
    checkpoint_records = []
    for series_index, spec in enumerate(validated_series):
        checkpoint = torch.load(
            spec.checkpoint,
            map_location=device,
            weights_only=False,
        )
        if checkpoint.get("pretrained") != args.pretrained:
            raise ValueError(
                f"{spec.label} uses {checkpoint.get('pretrained')}, "
                f"expected {args.pretrained}"
            )
        model = MoGeModel.from_pretrained(args.pretrained).to(device).eval()
        checkpoint_kind = "full_model" if "model" in checkpoint else "ssr_only"
        if checkpoint_kind == "full_model":
            model.load_state_dict(checkpoint["model"], strict=True)
        else:
            model.ssr.load_state_dict(checkpoint["ssr"], strict=True)
        cached = cache_base_predictions(
            data_dir,
            selected_manifest,
            model,
            height=args.height,
            width=args.width,
            num_tokens=args.num_tokens,
            device=device,
            safe_root=args.safe_root,
            cache_batch_size=1,
        )
        cached_by_id = {sample.sample_id: sample for sample in cached}
        with torch.inference_mode():
            for split, entry in entries.items():
                sample = cached_by_id[str(entry["id"])]
                base = sample.base_points[None].to(device)
                visual = sample.visual_features[None].to(device)
                gt = sample.gt_points.to(device)
                refined = refine_cached(
                    model.ssr.eval(),
                    base,
                    visual,
                    spec.refinement_steps,
                )[-1][0]
                if series_index == 0:
                    samples_by_split[split] = sample
                    aligned_by_split[split]["K=0"] = (
                        align_points(base[0], gt).cpu().numpy()
                    )
                aligned_by_split[split][spec.label] = (
                    align_points(refined, gt).cpu().numpy()
                )
        checkpoint_records.append(
            {
                "label": spec.label,
                "path": str(spec.checkpoint),
                "sha256": sha256_file(spec.checkpoint),
                "step": int(checkpoint["step"]),
                "kind": checkpoint_kind,
                "refinement_steps": spec.refinement_steps,
            }
        )
        del cached, checkpoint, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    import cv2

    yaw_values = np.linspace(args.yaw_min, args.yaw_max, args.orbit_frames)
    angular_step = (args.yaw_max - args.yaw_min) / (args.orbit_frames - 1)
    total_frames = (
        args.orbit_frames
        if args.motion == "one-way"
        else 2 * args.orbit_frames - 2
    )
    split_reports = {}
    for split, entry in entries.items():
        sample = samples_by_split[split]
        x0, y0, x1, y1 = validate_crop(
            entry["crop_xyxy"],
            args.height,
            args.width,
        )
        full_image = sample.image.permute(1, 2, 0).numpy()
        image = full_image[y0:y1, x0:x1]
        full_gt = sample.gt_points.numpy()
        gt = full_gt[y0:y1, x0:x1]
        mask_config = entry.get("display_mask")
        if not isinstance(mask_config, dict):
            raise ValueError(f"{split} selection has no display_mask object")
        mask, resolved_mask = build_display_mask(gt, mask_config)
        if args.mask_dilation:
            size = 2 * args.mask_dilation + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
            mask = cv2.dilate(mask.astype(np.uint8), kernel) > 0
        mask &= np.isfinite(gt).all(axis=-1)
        if int(mask.sum()) < 16:
            raise ValueError(f"{split} display mask is too small: {int(mask.sum())}")

        full_predictions = aligned_by_split[split]
        predictions = {
            label: points[y0:y1, x0:x1]
            for label, points in full_predictions.items()
        }
        metrics = {
            label: masked_relative_metrics(points, gt, mask)
            for label, points in predictions.items()
        }
        center = np.median(gt[mask], axis=0)
        radius = 1.08 * float(
            np.percentile(np.linalg.norm(gt[mask] - center, axis=-1), 99.5)
        )
        if not math.isfinite(radius) or radius <= 0:
            raise ValueError(f"Could not derive camera radius for {split}")

        forward_frames = [
            render_frame(
                experiment_label=args.experiment_label,
                split=split,
                full_image=full_image,
                crop=(x0, y0, x1, y1),
                image=image,
                mask=mask,
                gt_points=gt,
                predictions=predictions,
                metrics=metrics,
                center=center,
                radius=radius,
                yaw=float(yaw),
                pitch=args.pitch,
            )
            for yaw in yaw_values
        ]
        if args.motion == "one-way":
            frames = forward_frames
        else:
            frames = forward_frames + list(reversed(forward_frames[1:-1]))
        gif_name = f"{split}_rod_orbit.gif"
        frames[0].save(
            output / gif_name,
            save_all=True,
            append_images=frames[1:],
            duration=args.frame_duration_ms,
            loop=0,
            optimize=True,
        )
        preview_name = f"{split}_rod_preview.png"
        save_preview(forward_frames, output / preview_name)
        split_reports[split] = {
            "sample": {
                "id": sample.sample_id,
                "scene": sample.scene,
                "frame": sample.frame,
            },
            "crop_xyxy": [x0, y0, x1, y1],
            "display_mask_pixels": int(mask.sum()),
            "display_mask": resolved_mask,
            "metrics": metrics,
            "fixed_camera_radius_m": radius,
            "artifacts": {
                "orbit": gif_name,
                "preview": preview_name,
            },
        }

    report = {
        "status": "complete",
        "experiment": args.experiment_label,
        "selection": {
            "path": str(selection_path),
            "status": selection["status"],
            "inputs": selection["selection_inputs"],
        },
        "shape": [args.height, args.width],
        "display_mask": {
            "definition": (
                "split-specific ground-truth depth rule locked before model "
                "predictions and used only to isolate the displayed structure"
            ),
            "dilation_pixels": args.mask_dilation,
            "scope": (
                "Visualization-only diagnostic mask; not used for training, "
                "checkpoint selection, or paper Local benchmark evaluation."
            ),
        },
        "checkpoints": checkpoint_records,
        "render": {
            "yaw_degrees": [args.yaw_min, args.yaw_max],
            "pitch_degrees": args.pitch,
            "motion": args.motion,
            "frame_duration_ms": args.frame_duration_ms,
            "angular_step_degrees": angular_step,
            "angular_speed_degrees_per_second": (
                1000.0 * angular_step / args.frame_duration_ms
            ),
            "forward_frames": args.orbit_frames,
            "total_frames": total_frames,
            "loop_duration_ms": total_frames * args.frame_duration_ms,
            "camera_basis": "selected ground-truth fine-structure points only",
        },
        "splits": split_reports,
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
