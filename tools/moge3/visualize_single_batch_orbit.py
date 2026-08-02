from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from PIL import Image

from moge.model.v3 import MoGeModel
from moge.scripts.overfit_hypersim_v3 import aligned_metrics, load_batch
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.visualize_fine_structure import (
    draw_cloud,
    project_cloud,
    projection_bounds,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a GT/K=0/K=3 orbit GIF for the single-batch experiment"
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
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--yaw-min", type=float, default=-35.0)
    parser.add_argument("--yaw-max", type=float, default=35.0)
    parser.add_argument("--yaw-frames", type=int, default=15)
    parser.add_argument("--pitch", type=float, default=-8.0)
    parser.add_argument("--duration-ms", type=int, default=110)
    parser.add_argument("--point-stride", type=int, default=1)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def orbit_yaw_sequence(
    yaw_min: float,
    yaw_max: float,
    yaw_frames: int,
) -> List[float]:
    if yaw_frames < 2:
        raise ValueError("yaw_frames must be at least 2")
    if yaw_min >= yaw_max:
        raise ValueError("yaw_min must be smaller than yaw_max")
    forward = [float(value) for value in np.linspace(yaw_min, yaw_max, yaw_frames)]
    return forward + list(reversed(forward[1:-1]))


def render_orbit_frame(
    point_maps: Dict[str, np.ndarray],
    colors: np.ndarray,
    valid: np.ndarray,
    center: np.ndarray,
    *,
    yaw: float,
    pitch: float,
) -> Image.Image:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    projections = [
        project_cloud(
            points,
            colors,
            valid & np.isfinite(points).all(axis=-1),
            center,
            yaw,
            pitch,
        )
        for points in point_maps.values()
    ]
    bounds = projection_bounds(projections)
    figure, axes = plt.subplots(1, len(point_maps), figsize=(10.5, 3.6))
    for axis, label, projection in zip(axes, point_maps, projections):
        draw_cloud(axis, projection, bounds)
        axis.set_title(label)
    figure.suptitle(
        f"exp1 single-frame overfit | aligned point maps | yaw {yaw:+.0f}°",
        fontsize=13,
    )
    figure.tight_layout()
    figure.canvas.draw()
    rgba = np.asarray(figure.canvas.buffer_rgba()).copy()
    plt.close(figure)
    return Image.fromarray(rgba).convert("RGB")


def save_orbit_gif(
    output: Path,
    point_maps: Dict[str, np.ndarray],
    colors: np.ndarray,
    valid: np.ndarray,
    *,
    yaw_values: Sequence[float],
    pitch: float,
    duration_ms: int,
) -> None:
    frames = [
        render_orbit_frame(
            point_maps,
            colors,
            valid,
            np.nanmedian(point_maps["GT"].reshape(-1, 3), axis=0),
            yaw=yaw,
            pitch=pitch,
        )
        for yaw in yaw_values
    ]
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("Image dimensions must be positive")
    if not 1 <= args.refinement_steps <= 7:
        raise ValueError("--refinement-steps must be in [1, 7]")
    if args.duration_ms <= 0:
        raise ValueError("--duration-ms must be positive")
    if args.point_stride <= 0:
        raise ValueError("--point-stride must be positive")

    data_dir = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    checkpoint_path = assert_safe_path(
        args.checkpoint,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    image, gt_points, _, manifest = load_batch(
        data_dir,
        args.height,
        args.width,
    )
    image = image.to(device)
    gt_points = gt_points.to(device)
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
    model.ssr.load_state_dict(checkpoint["ssr"], strict=True)
    with torch.inference_mode():
        prediction = model(
            image,
            num_tokens=args.num_tokens,
            num_refinement_steps=args.refinement_steps,
            return_intermediates=True,
            detach_base_from_refiner=True,
        )
        k0_metrics, k0_aligned = aligned_metrics(
            prediction["points_sequence"][0],
            gt_points,
        )
        refined_metrics, refined_aligned = aligned_metrics(
            prediction["points_sequence"][args.refinement_steps],
            gt_points,
        )

    stride = args.point_stride
    gt = gt_points[0].cpu().numpy()[::stride, ::stride]
    k0 = k0_aligned[0].cpu().numpy()[::stride, ::stride]
    refined = refined_aligned[0].cpu().numpy()[::stride, ::stride]
    colors = (
        image[0]
        .permute(1, 2, 0)
        .cpu()
        .numpy()[::stride, ::stride]
        .clip(0.0, 1.0)
    )
    valid = np.isfinite(gt).all(axis=-1)
    point_maps = {
        "GT": gt,
        "K=0 base": k0,
        f"K={args.refinement_steps} refined": refined,
    }
    yaw_values = orbit_yaw_sequence(
        args.yaw_min,
        args.yaw_max,
        args.yaw_frames,
    )
    gif_name = "single_batch_point_map_orbit.gif"
    save_orbit_gif(
        output / gif_name,
        point_maps,
        colors,
        valid,
        yaw_values=yaw_values,
        pitch=args.pitch,
        duration_ms=args.duration_ms,
    )

    report = {
        "status": "complete",
        "experiment": "exp1 Hypersim single-batch point-map orbit",
        "scene": manifest["scene"],
        "camera": manifest["camera"],
        "frame": manifest["samples"][0]["frame"],
        "shape": [args.height, args.width],
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "pretrained": args.pretrained,
        "refinement_steps": args.refinement_steps,
        "metrics": {
            "k0": k0_metrics,
            f"k{args.refinement_steps}": refined_metrics,
        },
        "rendering": {
            "columns": list(point_maps),
            "yaw_min_degrees": args.yaw_min,
            "yaw_max_degrees": args.yaw_max,
            "unique_yaw_frames": args.yaw_frames,
            "output_frames": len(yaw_values),
            "pitch_degrees": args.pitch,
            "duration_ms": args.duration_ms,
            "point_stride": args.point_stride,
            "affine_aligned_before_rendering": True,
        },
        "artifact": gif_name,
    }
    report_path = output / "single_batch_point_map_orbit_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
