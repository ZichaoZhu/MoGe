from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


SELECTION_INPUTS = "RGB and ground truth only; no predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Lock one GT-depth-boundary crop for every Exp30 sample."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--grid-stride", type=int, default=16)
    return parser.parse_args()


def depth_boundary_score(points: torch.Tensor) -> torch.Tensor:
    depth = points[..., 2]
    valid = torch.isfinite(depth) & (depth > 0)
    safe_log_depth = torch.where(valid, depth.log(), torch.zeros_like(depth))
    horizontal = torch.zeros_like(depth)
    vertical = torch.zeros_like(depth)
    horizontal[:, 1:] = (
        (safe_log_depth[:, 1:] - safe_log_depth[:, :-1]).abs()
        * valid[:, 1:]
        * valid[:, :-1]
    )
    vertical[1:, :] = (
        (safe_log_depth[1:, :] - safe_log_depth[:-1, :]).abs()
        * valid[1:, :]
        * valid[:-1, :]
    )
    strength = (horizontal + vertical).clamp(max=1.0)
    return strength * valid


def select_crop(
    points: torch.Tensor,
    *,
    crop_size: int,
    grid_stride: int,
) -> tuple[list[int], float]:
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("Expected ground-truth points with shape [H,W,3]")
    height, width = points.shape[:2]
    if not 0 < crop_size <= min(height, width):
        raise ValueError("Crop size must fit within the image")
    if grid_stride <= 0:
        raise ValueError("Grid stride must be positive")

    strength = depth_boundary_score(points)
    pooled = F.avg_pool2d(
        strength[None, None],
        kernel_size=crop_size,
        stride=grid_stride,
    )[0, 0]
    flat_index = int(pooled.argmax().item())
    columns = pooled.shape[1]
    y0 = (flat_index // columns) * grid_stride
    x0 = (flat_index % columns) * grid_stride
    x0 = min(x0, width - crop_size)
    y0 = min(y0, height - crop_size)
    return [x0, y0, x0 + crop_size, y0 + crop_size], float(
        pooled.flatten()[flat_index].item()
    )


def build_manifest(
    data: Path,
    source: dict[str, Any],
    *,
    safe_root: Path,
    height: int,
    width: int,
    crop_size: int,
    grid_stride: int,
) -> dict[str, Any]:
    from moge.scripts.train_hypersim_smallset_v3 import load_raw_sample

    entries = []
    for metadata in source["samples"]:
        _, points = load_raw_sample(
            data,
            metadata,
            height,
            width,
            safe_root,
        )
        crop, score = select_crop(
            points,
            crop_size=crop_size,
            grid_stride=grid_stride,
        )
        entries.append(
            {
                "id": str(metadata["id"]),
                "split": str(metadata["split"]),
                "crop_xyxy": crop,
                "structure_mask": {
                    "type": "near_quantile",
                    "quantile": 0.5,
                },
                "gt_log_depth_boundary_score": score,
                "source": "exp30_automatic_gt_depth_boundary_crop",
            }
        )
    counts = {
        split: sum(entry["split"] == split for entry in entries)
        for split in ("train", "val", "test")
    }
    if counts != source["counts"]:
        raise RuntimeError("ROI counts do not match the source data manifest")
    return {
        "format": "moge3-fine-structure-rois-v1",
        "selection_inputs": SELECTION_INPUTS,
        "selection_rule": (
            "Select the 192x192 grid-aligned crop with the highest mean "
            "ground-truth log-depth boundary strength. Predictions are never used."
        ),
        "shape": [height, width],
        "crop_size": crop_size,
        "grid_stride": grid_stride,
        "counts": counts,
        "entries": entries,
    }


def main() -> None:
    args = parse_args()
    data = assert_safe_path(
        args.data,
        safe_root=args.safe_root,
        must_exist=True,
    )
    manifest_path = assert_safe_path(
        data / "manifest.json",
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(
        args.output,
        safe_root=args.safe_root,
        writable=True,
    )
    source = json.loads(manifest_path.read_text(encoding="utf-8"))
    if source.get("format") != "moge3-hypersim-generalization-v1":
        raise ValueError("Unsupported source data manifest")
    result = build_manifest(
        data,
        source,
        safe_root=args.safe_root,
        height=args.height,
        width=args.width,
        crop_size=args.crop_size,
        grid_stride=args.grid_stride,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), **result["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
