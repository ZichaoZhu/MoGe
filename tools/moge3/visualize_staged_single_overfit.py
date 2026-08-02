from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from moge.model.v3 import MoGeModel
from moge.scripts.overfit_hypersim_staged_v3 import (
    EVALUATION_STEPS,
    build_structure_mask,
    evaluate_single_image,
    load_sample_and_selection,
)
from moge.scripts.overfit_hypersim_v3 import colorize_depth
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.visualize_split_rods import (
    draw_cloud,
    highlighted_rgb,
    project_cloud,
    source_rgb_with_crop,
)


Crop = Tuple[int, int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render final and stage-comparison artifacts for one exp9 run"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--experiment-output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument("--yaw-min", type=float, default=-90.0)
    parser.add_argument("--yaw-max", type=float, default=90.0)
    parser.add_argument("--orbit-frames", type=int, default=46)
    parser.add_argument("--pitch", type=float, default=-8.0)
    parser.add_argument("--frame-duration-ms", type=int, default=110)
    parser.add_argument("--seed", type=int, default=97)
    return parser.parse_args()


def validate_render_args(args: argparse.Namespace) -> None:
    if min(args.height, args.width, args.num_tokens) <= 0:
        raise ValueError("Shape and token count must be positive")
    if args.yaw_min >= args.yaw_max or args.orbit_frames < 3:
        raise ValueError("Invalid one-way yaw sequence")
    if args.frame_duration_ms <= 0:
        raise ValueError("Frame duration must be positive")


def load_model(
    checkpoint_path: Path,
    *,
    pretrained: str,
    device: torch.device,
) -> Tuple[MoGeModel, Mapping[str, object]]:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("pretrained") != pretrained:
        raise ValueError(
            f"Checkpoint uses {checkpoint.get('pretrained')}, expected {pretrained}"
        )
    model = MoGeModel.from_pretrained(pretrained).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


def camera_parameters(
    gt_points: np.ndarray,
    mask: np.ndarray,
) -> Tuple[np.ndarray, float]:
    values = gt_points[mask & np.isfinite(gt_points).all(axis=-1)]
    if values.shape[0] < 16:
        raise ValueError("Visualization mask has too few points")
    center = np.median(values, axis=0)
    radius = 1.08 * float(
        np.percentile(np.linalg.norm(values - center, axis=-1), 99.5)
    )
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Could not derive a finite visualization radius")
    return center, radius


def render_orbit_frame(
    *,
    label: str,
    full_image: np.ndarray,
    crop: Crop,
    crop_image: np.ndarray,
    gt_points: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, float]],
    mask: np.ndarray,
    structure_mask: np.ndarray,
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
        figsize=(3.05 * (len(columns) + 2), 3.65),
    )
    axes[0].imshow(source_rgb_with_crop(full_image, crop))
    axes[0].set_title("Source RGB\nred box = locked crop")
    axes[0].axis("off")
    if label == "structure":
        axes[1].imshow(highlighted_rgb(crop_image, structure_mask))
        axes[1].set_title(
            f"GT-locked structure\n{int(structure_mask.sum())} pixels"
        )
    else:
        axes[1].imshow(crop_image)
        axes[1].set_title(f"Complete crop\n{int(mask.sum())} valid pixels")
    axes[1].axis("off")

    for axis, (name, points) in zip(axes[2:], columns):
        valid = mask & np.isfinite(points).all(axis=-1)
        draw_cloud(
            axis,
            project_cloud(points, crop_image, valid, center, yaw, pitch),
            radius,
        )
        if name == "GT":
            axis.set_title("Ground truth")
        else:
            axis.set_title(
                f"{name}\npoint Rel {100 * metrics[name]['point_rel']:.2f}%"
            )
    figure.suptitle(
        f"exp9 {label} orbit | yaw {yaw:+.0f}°",
        fontsize=13,
        y=0.985,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba).convert("RGB")


def save_gif(
    output: Path,
    *,
    label: str,
    full_image: np.ndarray,
    crop: Crop,
    crop_image: np.ndarray,
    gt_points: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    metrics: Mapping[str, Mapping[str, float]],
    mask: np.ndarray,
    structure_mask: np.ndarray,
    yaw_values: Sequence[float],
    pitch: float,
    duration_ms: int,
) -> Dict[str, object]:
    center, radius = camera_parameters(gt_points, mask)
    frames = [
        render_orbit_frame(
            label=label,
            full_image=full_image,
            crop=crop,
            crop_image=crop_image,
            gt_points=gt_points,
            predictions=predictions,
            metrics=metrics,
            mask=mask,
            structure_mask=structure_mask,
            center=center,
            radius=radius,
            yaw=float(yaw),
            pitch=pitch,
        )
        for yaw in yaw_values
    ]
    palette_frames = [
        frame.quantize(
            colors=256,
            method=Image.Quantize.MEDIANCUT,
            dither=Image.Dither.NONE,
        )
        for frame in frames
    ]
    palette_frames[0].save(
        output,
        save_all=True,
        append_images=palette_frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
        disposal=2,
    )
    preview = [frames[0], frames[len(frames) // 2], frames[-1]]
    target_width = min(2000, preview[0].width)
    resized = [
        frame.resize(
            (target_width, round(frame.height * target_width / frame.width)),
            Image.Resampling.LANCZOS,
        )
        for frame in preview
    ]
    canvas = Image.new(
        "RGB",
        (target_width, sum(frame.height for frame in resized)),
        "white",
    )
    y = 0
    for frame in resized:
        canvas.paste(frame, (0, y))
        y += frame.height
    canvas.save(output.with_suffix(".png"), optimize=True)
    return {
        "file": output.name,
        "preview": output.with_suffix(".png").name,
        "frames": len(frames),
        "duration_ms": duration_ms,
        "radius_m": radius,
    }


def scope_metrics_from_report(
    report: Mapping[str, object],
    scope: str,
) -> Dict[str, Mapping[str, float]]:
    return {
        f"Final K={k}": report[f"k{k}"][scope]
        for k in EVALUATION_STEPS
    }


def save_stage_comparison(
    output: Path,
    image: torch.Tensor,
    gt_points: torch.Tensor,
    snapshots: Mapping[str, Mapping[int, torch.Tensor]],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    rgb = image.permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0)
    gt = gt_points.cpu().numpy()
    valid = np.isfinite(gt).all(axis=-1)
    lo, hi = np.percentile(gt[..., 2][valid], [2.0, 98.0])
    panels = [("RGB", rgb), ("GT", colorize_depth(gt[..., 2], valid, lo, hi)[:, :, ::-1] / 255.0)]
    for stage in ("initial", "stage1", "final"):
        for k in (0, 3):
            depth = snapshots[stage][k].numpy()[..., 2]
            panels.append(
                (
                    f"{stage} K={k}",
                    colorize_depth(depth, valid, lo, hi)[:, :, ::-1] / 255.0,
                )
            )
    figure, axes = plt.subplots(2, 4, figsize=(16, 8))
    for axis, (title, panel) in zip(axes.flat, panels):
        axis.imshow(panel)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    figure.savefig(output.with_suffix(".pdf"))
    plt.close(figure)


def save_stage_point_error_comparison(
    output: Path,
    gt_points: torch.Tensor,
    snapshots: Mapping[str, Mapping[int, torch.Tensor]],
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    gt = gt_points.cpu().numpy()
    valid = np.isfinite(gt).all(axis=-1) & (gt[..., 2] > 0)
    gt_norm = np.linalg.norm(gt, axis=-1).clip(1e-6)
    errors: Dict[tuple[str, int], np.ndarray] = {}
    finite_errors = []
    for stage in ("initial", "stage1", "final"):
        for k in (0, 3):
            predicted = snapshots[stage][k].numpy()
            error = np.linalg.norm(predicted - gt, axis=-1) / gt_norm
            error[~valid | ~np.isfinite(error)] = np.nan
            errors[(stage, k)] = error
            finite_errors.append(error[np.isfinite(error)])

    all_errors = np.concatenate(finite_errors)
    color_max = max(float(np.percentile(all_errors, 98.0)), 1e-3)
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    image_handle = None
    for column, stage in enumerate(("initial", "stage1", "final")):
        for row, k in enumerate((0, 3)):
            axis = axes[row, column]
            image_handle = axis.imshow(
                errors[(stage, k)],
                cmap="magma",
                vmin=0.0,
                vmax=color_max,
            )
            axis.set_title(f"{stage} K={k}")
            axis.axis("off")
    assert image_handle is not None
    colorbar = figure.colorbar(
        image_handle,
        ax=axes.ravel().tolist(),
        fraction=0.025,
        pad=0.02,
    )
    colorbar.set_label("Per-pixel relative 3D point error")
    figure.suptitle("Point-map error with a shared color scale")
    figure.savefig(output, dpi=180, bbox_inches="tight")
    figure.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def run(args: argparse.Namespace) -> None:
    validate_render_args(args)
    data = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    selection = assert_safe_path(
        args.selection,
        safe_root=args.safe_root,
        must_exist=True,
    )
    experiment_output = assert_safe_path(
        args.experiment_output,
        safe_root=args.safe_root,
        must_exist=True,
        writable=True,
    )
    artifacts = experiment_output / "artifacts"
    checkpoints = experiment_output / "checkpoints"
    artifacts.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    image, gt_points, _, entry = load_sample_and_selection(
        data,
        selection,
        args.sample_id,
        height=args.height,
        width=args.width,
        safe_root=args.safe_root,
    )
    crop = tuple(int(value) for value in entry["crop_xyxy"])
    structure = build_structure_mask(
        gt_points,
        crop,
        entry["display_mask"],
    )
    image_device = image.to(device)
    gt_device = gt_points.to(device)

    snapshots: Dict[str, Mapping[int, torch.Tensor]] = {}
    final_metrics: Mapping[str, object] | None = None
    final_aligned: Mapping[int, torch.Tensor] | None = None
    checkpoint_records = []
    for stage, filename in (
        ("initial", "initial.pt"),
        ("stage1", "stage1.pt"),
        ("final", "final.pt"),
    ):
        path = assert_safe_path(
            checkpoints / filename,
            safe_root=args.safe_root,
            must_exist=True,
        )
        model, checkpoint = load_model(
            path,
            pretrained=args.pretrained,
            device=device,
        )
        metrics, aligned = evaluate_single_image(
            model,
            image_device,
            gt_device,
            crop=crop,
            structure_mask=structure,
            num_tokens=args.num_tokens,
            refinement_steps=EVALUATION_STEPS,
            boundary_threshold=0.03,
        )
        snapshots[stage] = aligned
        if stage == "final":
            final_metrics = metrics
            final_aligned = aligned
        checkpoint_records.append(
            {
                "stage": stage,
                "file": filename,
                "step": int(checkpoint["step"]),
            }
        )
        del model, checkpoint
        if device.type == "cuda":
            torch.cuda.empty_cache()

    assert final_metrics is not None and final_aligned is not None
    x0, y0, x1, y1 = crop
    full_image = image.permute(1, 2, 0).numpy()
    crop_image = full_image[y0:y1, x0:x1]
    gt_crop = gt_points.numpy()[y0:y1, x0:x1]
    crop_valid = np.isfinite(gt_crop).all(axis=-1) & (gt_crop[..., 2] > 0)
    structure_np = structure.numpy()
    predictions = {
        f"Final K={k}": final_aligned[k].numpy()[y0:y1, x0:x1]
        for k in EVALUATION_STEPS
    }
    yaw_values = np.linspace(args.yaw_min, args.yaw_max, args.orbit_frames)
    crop_gif = save_gif(
        artifacts / "final_crop_orbit.gif",
        label="complete crop",
        full_image=full_image,
        crop=crop,
        crop_image=crop_image,
        gt_points=gt_crop,
        predictions=predictions,
        metrics=scope_metrics_from_report(final_metrics, "crop"),
        mask=crop_valid,
        structure_mask=structure_np,
        yaw_values=yaw_values,
        pitch=args.pitch,
        duration_ms=args.frame_duration_ms,
    )
    structure_gif = save_gif(
        artifacts / "final_structure_orbit.gif",
        label="locked structure",
        full_image=full_image,
        crop=crop,
        crop_image=crop_image,
        gt_points=gt_crop,
        predictions=predictions,
        metrics=scope_metrics_from_report(final_metrics, "structure"),
        mask=structure_np,
        structure_mask=structure_np,
        yaw_values=yaw_values,
        pitch=args.pitch,
        duration_ms=args.frame_duration_ms,
    )
    save_stage_comparison(
        artifacts / "stage_depth_comparison.png",
        image,
        gt_points,
        snapshots,
    )
    save_stage_point_error_comparison(
        artifacts / "stage_point_error_comparison.png",
        gt_points,
        snapshots,
    )
    report = {
        "status": "complete",
        "sample_id": args.sample_id,
        "crop_xyxy": list(crop),
        "checkpoints": checkpoint_records,
        "render": {
            "yaw_degrees": [args.yaw_min, args.yaw_max],
            "motion": "one-way",
            "frames": args.orbit_frames,
            "duration_ms": args.frame_duration_ms,
            "pitch_degrees": args.pitch,
        },
        "artifacts": {
            "complete_crop": crop_gif,
            "locked_structure": structure_gif,
            "stage_depth_comparison": "stage_depth_comparison.png",
            "stage_point_error_comparison": "stage_point_error_comparison.png",
        },
    }
    (artifacts / "visualization_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
