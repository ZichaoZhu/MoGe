from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import torch

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.export_exp9_pointclouds import (
    point_asset,
    read_binary_ply,
    write_binary_ply,
)


DEFAULT_MANIFESTS = (
    "manifest.json",
    "exp12/manifest.json",
    "exp20/manifest.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Add valid Hypersim ground-truth point clouds to existing viewer "
            "manifests without rerunning any model."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "experiment/exp11_hypersim_100_train_staged_joint_overfit/data"
        ),
    )
    parser.add_argument(
        "--viewer",
        type=Path,
        default=Path(
            "experiment/exp9_thin_structure_single_image_staged_overfit/viewer"
        ),
    )
    parser.add_argument(
        "--manifests",
        nargs="+",
        default=list(DEFAULT_MANIFESTS),
    )
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    return parser.parse_args()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def serialized_ground_truth(
    points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(points, dtype=np.float32)
    if value.ndim != 3 or value.shape[-1] != 3:
        raise ValueError("Ground-truth point map must have shape HxWx3")
    valid = np.isfinite(value).all(axis=-1) & (value[..., 2] > 0)
    if not valid.any():
        raise ValueError("Ground-truth point map has no valid pixels")
    serialized = value.copy()
    serialized[~valid] = 0.0
    return serialized, valid


def ground_truth_metrics(
    gt_points: torch.Tensor,
    sample: Mapping[str, Any],
    *,
    boundary_threshold: float,
) -> Dict[str, Any]:
    from moge.scripts.overfit_hypersim_staged_v3 import (
        _scope_metrics,
        build_structure_mask,
    )

    gt = gt_points.float()
    full_valid = torch.isfinite(gt).all(dim=-1) & (gt[..., 2] > 0)
    x0, y0, x1, y1 = (int(value) for value in sample["cropXYXY"])
    crop_valid = full_valid[y0:y1, x0:x1]
    structure_mask = build_structure_mask(
        gt,
        [x0, y0, x1, y1],
        sample["structureMask"],
    )
    return {
        "full": _scope_metrics(
            gt,
            gt,
            full_valid,
            boundary_threshold=boundary_threshold,
        ),
        "crop": _scope_metrics(
            gt[y0:y1, x0:x1],
            gt[y0:y1, x0:x1],
            crop_valid,
            boundary_threshold=boundary_threshold,
        ),
        "structure": _scope_metrics(
            gt[y0:y1, x0:x1],
            gt[y0:y1, x0:x1],
            structure_mask,
            boundary_threshold=boundary_threshold,
        ),
    }


def verify_ground_truth_ply(
    path: Path,
    *,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    decoded_points, decoded_colors, comments = read_binary_ply(path)
    np.testing.assert_array_equal(decoded_points, points.reshape(-1, 3))
    np.testing.assert_array_equal(decoded_colors, colors.reshape(-1, 3))
    if comments.get("vertex_order") != "row_major_one_vertex_per_pixel":
        raise ValueError("Ground-truth PLY did not preserve raster order")


def _public_path(public_root: Path, url: str) -> Path:
    if not url.startswith("/data/"):
        raise ValueError(f"Viewer asset URL is outside /data/: {url}")
    return public_root / url.removeprefix("/")


def _load_checksums(path: Path) -> list[Dict[str, Any]]:
    if not path.exists():
        return []
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Checksum manifest must be a list: {path}")
    return [
        dict(record)
        for record in value
        if record.get("stage") != "ground_truth"
    ]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from moge.scripts.train_hypersim_smallset_v3 import load_raw_sample

    if min(args.height, args.width) <= 0:
        raise ValueError("Image dimensions must be positive")
    project_root = assert_safe_path(
        args.project_root,
        safe_root=args.safe_root,
        must_exist=True,
    )
    data = assert_safe_path(
        args.data if args.data.is_absolute() else project_root / args.data,
        safe_root=args.safe_root,
        must_exist=True,
    )
    viewer = assert_safe_path(
        args.viewer if args.viewer.is_absolute() else project_root / args.viewer,
        safe_root=args.safe_root,
        must_exist=True,
        writable=True,
    )
    public_root = viewer / "public"
    data_root = public_root / "data"
    data_manifest = json.loads(
        (data / "manifest.json").read_text(encoding="utf-8")
    )
    metadata_by_id = {
        str(sample["id"]): sample for sample in data_manifest["samples"]
    }
    report_manifests: list[Dict[str, Any]] = []

    for relative in args.manifests:
        manifest_path = assert_safe_path(
            data_root / relative,
            safe_root=args.safe_root,
            must_exist=True,
            writable=True,
        )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        selected = [
            sample for sample in manifest["samples"]
            if sample.get("websiteEnabled")
        ]
        checksum_path = manifest_path.parent / "pointcloud_sha256.json"
        checksums = _load_checksums(checksum_path)
        exported: list[Dict[str, Any]] = []

        for sample in selected:
            sample_id = str(sample["id"])
            metadata = metadata_by_id.get(sample_id)
            if metadata is None:
                raise ValueError(f"{sample_id} is absent from the data manifest")
            image, gt_points = load_raw_sample(
                data,
                metadata,
                args.height,
                args.width,
                args.safe_root,
            )
            raw_gt = gt_points.float().cpu().numpy()
            serialized, valid = serialized_ground_truth(raw_gt)
            colors = (
                image.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
            ).round().astype(np.uint8)
            rgb_path = _public_path(public_root, str(sample["rgbUrl"]))
            output = rgb_path.parent / "ground_truth.ply"
            write_binary_ply(
                output,
                serialized,
                colors,
                width=args.width,
                height=args.height,
            )
            verify_ground_truth_ply(
                output,
                points=serialized,
                colors=colors,
            )
            metrics = ground_truth_metrics(
                gt_points,
                sample,
                boundary_threshold=args.boundary_threshold,
            )
            asset = point_asset(
                output,
                public_root=public_root,
                points=raw_gt,
                checkpoint_digest=array_sha256(raw_gt),
                alignment={"scale": 1.0, "zShift": 0.0},
                metrics=metrics,
            )
            asset["validPointCount"] = int(valid.sum())
            sample["groundTruth"] = asset
            record = {
                "sample": sample_id,
                "split": sample.get("split"),
                "stage": "ground_truth",
                "k": None,
                "url": asset["url"],
                "bytes": output.stat().st_size,
                "sha256": asset["sha256"],
                "validPointCount": asset["validPointCount"],
            }
            checksums.append(record)
            exported.append(record)

        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        checksum_path.write_text(
            json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        report_manifests.append(
            {
                "manifest": relative,
                "experiment": manifest["experiment"],
                "groundTruthCount": len(exported),
                "bytes": sum(int(record["bytes"]) for record in exported),
                "samples": exported,
            }
        )

    report = {
        "status": "complete",
        "format": "moge3_viewer_ground_truth_export_v1",
        "resolution": {"width": args.width, "height": args.height},
        "manifests": report_manifests,
        "groundTruthCount": sum(
            item["groundTruthCount"] for item in report_manifests
        ),
        "bytes": sum(item["bytes"] for item in report_manifests),
    }
    (data_root / "ground_truth_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
