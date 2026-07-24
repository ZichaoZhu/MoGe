from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import h5py
import numpy as np
import torch
from PIL import Image

from moge.model.v3 import MoGeModel
from moge.train.losses import edge_loss
from moge.train.losses_v3 import (
    affine_invariant_global_loss_v3,
    radial_partition_local_loss,
    solve_global_affine_alignment,
)
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit MoGe-3 SSR on one fixed real Hypersim batch"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--global-weight", type=float, default=1.0)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--local-scales", type=int, nargs="*", default=[4, 16, 64])
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_batch(
    data_dir: Path,
    height: int,
    width: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, object]]:
    manifest = json.loads((data_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "moge3-hypersim-single-batch-v1":
        raise ValueError(f"Unsupported manifest format: {manifest.get('format')}")
    matrix = np.asarray(manifest["M_cam_from_uv"], dtype=np.float32)

    half_du = 1.0 / width
    half_dv = 1.0 / height
    u = np.linspace(-1.0 + half_du, 1.0 - half_du, width, dtype=np.float32)
    v = np.linspace(-1.0 + half_dv, 1.0 - half_dv, height, dtype=np.float32)[::-1]
    grid_u, grid_v = np.meshgrid(u, v)
    uv1 = np.stack((grid_u, grid_v, np.ones_like(grid_u)), axis=-1)
    rays = uv1 @ matrix.T
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True).clip(1e-8)
    rays = rays * np.asarray([1.0, -1.0, -1.0], dtype=np.float32)

    images: List[np.ndarray] = []
    points: List[np.ndarray] = []
    masks: List[np.ndarray] = []
    for sample in manifest["samples"]:
        rgb_path = data_dir / sample["rgb"]["file"]
        depth_path = data_dir / sample["depth"]["file"]
        with Image.open(rgb_path) as image:
            image = image.convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
            rgb = np.asarray(image, dtype=np.float32) / 255.0
        with h5py.File(depth_path, "r") as file:
            radial_depth = file["dataset"][:].astype(np.float32)
        radial_depth = cv2.resize(
            radial_depth,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
        valid = np.isfinite(radial_depth) & (radial_depth > 0.0)
        point_map = radial_depth[..., None] * rays
        point_map[~valid] = np.nan
        images.append(rgb)
        points.append(point_map)
        masks.append(valid)

    image_tensor = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2)
    point_tensor = torch.from_numpy(np.stack(points))
    mask_tensor = torch.from_numpy(np.stack(masks))
    return image_tensor, point_tensor, mask_tensor, manifest


def geometry_loss(
    points_sequence: List[torch.Tensor],
    gt_points: torch.Tensor,
    *,
    global_weight: float,
    local_weight: float,
    edge_weight: float,
    local_scales: Tuple[int, ...],
    generator: torch.Generator,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    total = gt_points.new_zeros(())
    records: Dict[str, float] = {}
    for index, points in enumerate(points_sequence):
        global_per_image, alignment = affine_invariant_global_loss_v3(points, gt_points)
        global_term = global_per_image.mean()
        if local_weight:
            local_term = radial_partition_local_loss(
                points,
                gt_points,
                alignment,
                scales=local_scales,
                generator=generator,
            ).mean()
        else:
            local_term = global_term.new_zeros(())
        if edge_weight:
            edge_term = edge_loss(points, gt_points)[0].mean()
        else:
            edge_term = global_term.new_zeros(())
        total = total + (
            global_weight * global_term
            + local_weight * local_term
            + edge_weight * edge_term
        )
        records[f"k{index + 1}/global"] = float(global_term.detach())
        records[f"k{index + 1}/local"] = float(local_term.detach())
        records[f"k{index + 1}/edge"] = float(edge_term.detach())
    return total, records


@torch.no_grad()
def aligned_metrics(
    pred_points: torch.Tensor,
    gt_points: torch.Tensor,
) -> Tuple[Dict[str, float], torch.Tensor]:
    alignment = solve_global_affine_alignment(pred_points, gt_points)
    aligned = alignment.apply(pred_points)
    valid = torch.isfinite(gt_points).all(dim=-1)
    safe_gt = torch.where(valid[..., None], gt_points, torch.ones_like(gt_points))
    point_rel = (aligned - safe_gt).norm(dim=-1) / safe_gt[..., 2].clamp_min(1e-6)
    depth_rel = (aligned[..., 2] - safe_gt[..., 2]).abs() / safe_gt[..., 2].clamp_min(1e-6)
    ratio = torch.maximum(
        aligned[..., 2] / safe_gt[..., 2].clamp_min(1e-6),
        safe_gt[..., 2] / aligned[..., 2].clamp_min(1e-6),
    )
    return {
        "point_rel": float(point_rel[valid].mean()),
        "depth_rel": float(depth_rel[valid].mean()),
        "depth_delta_1.01": float((ratio[valid] < 1.01).float().mean()),
        "depth_delta_1.25": float((ratio[valid] < 1.25).float().mean()),
        "alignment_scale": float(alignment.scale.mean()),
        "alignment_z_shift": float(alignment.shift[..., 2].mean()),
    }, aligned


def colorize_depth(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    normalized = np.clip((depth - lo) / max(hi - lo, 1e-8), 0.0, 1.0)
    colored = cv2.applyColorMap(
        np.round(255.0 * (1.0 - normalized)).astype(np.uint8),
        cv2.COLORMAP_TURBO,
    )
    colored[~valid] = 0
    return colored


def save_visuals(
    output: Path,
    image: torch.Tensor,
    gt_points: torch.Tensor,
    before: torch.Tensor,
    after: torch.Tensor,
    losses: List[float],
) -> None:
    rgb = (
        image[0].detach().cpu().permute(1, 2, 0).numpy().clip(0, 1)[:, :, ::-1] * 255
    ).astype(np.uint8)
    gt_depth = gt_points[0, ..., 2].detach().cpu().numpy()
    before_depth = before[0, ..., 2].detach().cpu().numpy()
    after_depth = after[0, ..., 2].detach().cpu().numpy()
    valid = np.isfinite(gt_depth)
    lo, hi = np.percentile(gt_depth[valid], [2.0, 98.0])
    panels = [
        ("RGB", rgb),
        ("GT depth", colorize_depth(gt_depth, valid, lo, hi)),
        ("K=0 aligned", colorize_depth(before_depth, valid, lo, hi)),
        ("K refined aligned", colorize_depth(after_depth, valid, lo, hi)),
    ]
    for label, panel in panels:
        cv2.putText(
            panel,
            label,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    comparison = np.concatenate([panel for _, panel in panels], axis=1)
    cv2.imwrite(str(output / "comparison.png"), comparison)

    canvas = np.full((360, 720, 3), 255, dtype=np.uint8)
    if losses:
        values = np.log10(np.asarray(losses, dtype=np.float64).clip(1e-12))
        y_lo, y_hi = float(values.min()), float(values.max())
        xs = np.linspace(50, 690, len(values))
        ys = 320 - (values - y_lo) / max(y_hi - y_lo, 1e-12) * 280
        polyline = np.stack((xs, ys), axis=-1).round().astype(np.int32)
        cv2.polylines(canvas, [polyline], False, (40, 90, 220), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"log10(loss), {losses[0]:.5g} -> {losses[-1]:.5g}",
            (50, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output / "loss_curve.png"), canvas)
    np.save(output / "gt_depth.npy", gt_depth)
    np.save(output / "depth_k0_aligned.npy", before_depth)
    np.save(output / "depth_refined_aligned.npy", after_depth)


def main() -> None:
    args = parse_args()
    if args.height <= 0 or args.width <= 0:
        raise ValueError("Image dimensions must be positive")
    if not 1 <= args.refinement_steps <= 7:
        raise ValueError("--refinement-steps must be in [1, 7]")

    data_dir = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    image, gt_points, valid_mask, manifest = load_batch(
        data_dir,
        args.height,
        args.width,
    )
    image = image.to(device)
    gt_points = gt_points.to(device)
    valid_mask = valid_mask.to(device)

    model = MoGeModel.from_pretrained(args.pretrained).to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.ssr.parameters():
        parameter.requires_grad_(True)
    model.eval()
    model.ssr.train()
    optimizer = torch.optim.AdamW(
        model.ssr.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator(device=device).manual_seed(args.seed + 1)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.no_grad():
        initial_output = model(
            image,
            num_tokens=args.num_tokens,
            num_refinement_steps=args.refinement_steps,
            return_intermediates=True,
            detach_base_from_refiner=True,
        )
        initial_residual_max = max(
            float(value.abs().max())
            for value in initial_output["log_depth_residuals"]
        )
        before_metrics, before_aligned = aligned_metrics(
            initial_output["points_sequence"][0],
            gt_points,
        )

    losses: List[float] = []
    records: Dict[str, float] = {}
    start = time.perf_counter()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        output_dict = model(
            image,
            num_tokens=args.num_tokens,
            num_refinement_steps=args.refinement_steps,
            return_intermediates=True,
            detach_base_from_refiner=True,
        )
        loss, records = geometry_loss(
            output_dict["points_sequence"][1:],
            gt_points,
            global_weight=args.global_weight,
            local_weight=args.local_weight,
            edge_weight=args.edge_weight,
            local_scales=tuple(args.local_scales),
            generator=generator,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(loss)}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.ssr.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        optimizer.step()
        losses.append(float(loss.detach()))
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            elapsed = time.perf_counter() - start
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": losses[-1],
                        "grad_norm": float(grad_norm),
                        "seconds": elapsed,
                        **records,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    elapsed = time.perf_counter() - start
    model.eval()
    with torch.no_grad():
        final_output = model(
            image,
            num_tokens=args.num_tokens,
            num_refinement_steps=args.refinement_steps,
            return_intermediates=True,
            detach_base_from_refiner=True,
        )
        after_metrics, after_aligned = aligned_metrics(
            final_output["points"],
            gt_points,
        )
        final_residual_rms = math.sqrt(
            sum(float(value.square().mean()) for value in final_output["log_depth_residuals"])
            / len(final_output["log_depth_residuals"])
        )

    checkpoint = {
        "ssr": model.ssr.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": args.steps,
        "pretrained": args.pretrained,
        "ssr_config": model.ssr_config,
        "args": vars(args),
    }
    torch.save(checkpoint, output / "checkpoint.pt")
    save_visuals(
        output,
        image,
        gt_points,
        before_aligned,
        after_aligned,
        losses,
    )
    report = {
        "status": "complete",
        "experiment": "real Hypersim fixed-batch SSR overfit",
        "scene": manifest["scene"],
        "camera": manifest["camera"],
        "frames": [sample["frame"] for sample in manifest["samples"]],
        "batch_size": len(manifest["samples"]),
        "shape": [args.height, args.width],
        "valid_pixel_ratio": float(valid_mask.float().mean()),
        "pretrained": args.pretrained,
        "base_frozen": True,
        "normal_head_present": hasattr(model, "normal_head"),
        "refinement_steps": args.refinement_steps,
        "optimization_steps": args.steps,
        "loss_weights": {
            "global": args.global_weight,
            "local": args.local_weight,
            "edge": args.edge_weight,
        },
        "local_scales": args.local_scales,
        "initial_zero_residual_max": initial_residual_max,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "loss_reduction": losses[-1] / losses[0] if losses else None,
        "before": before_metrics,
        "after": after_metrics,
        "elapsed_seconds": elapsed,
        "seconds_per_step": elapsed / max(args.steps, 1),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "final_residual_rms": final_residual_rms,
        "artifacts": {
            "checkpoint": "checkpoint.pt",
            "comparison": "comparison.png",
            "loss_curve": "loss_curve.png",
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
