from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from PIL import Image

from moge.model.ssr import capture_batch_norm_running_state
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.export_exp9_pointclouds import (
    EXPECTED_STEPS,
    VOXEL_DEPTH_SCALE,
    point_asset,
    read_binary_ply,
    write_binary_ply,
)
from tools.moge3.export_viewer_experiment import (
    batch_norm_state_sha256,
    batch_norm_states_equal,
    canonical_full_metrics,
    construct_checkpoint_model,
    load_reference_metrics,
    predict_and_measure,
    sha256_file,
    validate_full_metrics,
    validate_selection,
)


EXPERIMENT = "exp29_smooth_bounded_residual_long_joint"
SPLITS = ("train", "val", "test")
DEFAULT_RELATIVE_PATHS = {
    "data": (
        "experiment/exp11_hypersim_100_train_staged_joint_overfit/data"
    ),
    "checkpoint": (
        "experiment/stage3_stability_normalization/runs/"
        "exp29_smooth_bounded_residual_long_joint/artifacts/formal/checkpoint.pt"
    ),
    "viewer": (
        "experiment/exp9_thin_structure_single_image_staged_overfit/viewer"
    ),
    "report_output": (
        "experiment/stage3_stability_normalization/runs/"
        "exp29_smooth_bounded_residual_long_joint/results/viewer_export"
    ),
    "reference_metrics": (
        "experiment/stage3_stability_normalization/runs/"
        "exp29_smooth_bounded_residual_long_joint/results/viewer_evaluation/"
        "per_frame_metrics.csv"
    ),
    "selection_manifest": (
        "experiment/stage3_stability_normalization/runs/"
        "exp29_smooth_bounded_residual_long_joint/viewer_selection.json"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Exp29's three best improvements and two worst degradations "
            "per split as final-checkpoint browser point clouds."
        )
    )
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    for key in DEFAULT_RELATIVE_PATHS:
        parser.add_argument(f"--{key.replace('_', '-')}", type=Path)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--crop-stride", type=int, default=16)
    parser.add_argument("--edge-relative-threshold", type=float, default=0.05)
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    parser.add_argument("--metric-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--smooth-log-depth-residual-bound",
        type=float,
        default=0.1,
    )
    return parser.parse_args()


def _resolve(
    explicit: Path | None,
    *,
    project_root: Path,
    key: str,
) -> Path:
    return explicit or project_root / DEFAULT_RELATIVE_PATHS[key]


def _public_url(path: Path, public_root: Path) -> str:
    return "/" + path.relative_to(public_root).as_posix()


def _verify_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
) -> None:
    decoded_points, decoded_colors, comments = read_binary_ply(path)
    np.testing.assert_allclose(
        decoded_points,
        points.reshape(-1, 3),
        rtol=0.0,
        atol=0.0,
        equal_nan=True,
    )
    np.testing.assert_array_equal(decoded_colors, colors.reshape(-1, 3))
    if comments.get("vertex_order") != "row_major_one_vertex_per_pixel":
        raise ValueError("PLY raster order was not preserved")


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def _locked_crop(
    selected: Mapping[str, Any],
    *,
    width: int,
    height: int,
    expected_size: int,
) -> list[int]:
    raw = selected.get("cropXYXY")
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"{selected.get('id')} lacks a locked cropXYXY")
    crop = [int(value) for value in raw]
    x0, y0, x1, y1 = crop
    if (
        x0 < 0
        or y0 < 0
        or x1 > width
        or y1 > height
        or x1 - x0 != expected_size
        or y1 - y0 != expected_size
    ):
        raise ValueError(f"{selected.get('id')} has an invalid crop: {crop}")
    return crop


def _ground_truth_metrics(
    gt_points: torch.Tensor,
    *,
    crop: list[int],
    boundary_threshold: float,
) -> tuple[Dict[str, Any], int]:
    from moge.scripts.overfit_hypersim_staged_v3 import (
        _scope_metrics,
        build_structure_mask,
    )

    gt = gt_points.float()
    full_valid = torch.isfinite(gt).all(dim=-1) & (gt[..., 2] > 0)
    x0, y0, x1, y1 = crop
    crop_valid = full_valid[y0:y1, x0:x1]
    structure_mask = build_structure_mask(
        gt,
        crop,
        {"type": "near_quantile", "quantile": 0.5},
    )
    return (
        {
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
        },
        int(structure_mask.sum().item()),
    )


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from moge.scripts.train_hypersim_smallset_v3 import load_raw_sample

    if min(
        args.height,
        args.width,
        args.num_tokens,
        args.crop_size,
        args.crop_stride,
    ) <= 0:
        raise ValueError("Image, token and crop parameters must be positive")
    if args.smooth_log_depth_residual_bound <= 0:
        raise ValueError("Exp29 requires a positive smooth residual bound")

    project_root = assert_safe_path(
        args.project_root,
        safe_root=args.safe_root,
        must_exist=True,
    )
    resolved = {
        key: _resolve(
            getattr(args, key),
            project_root=project_root,
            key=key,
        )
        for key in DEFAULT_RELATIVE_PATHS
    }
    data = assert_safe_path(
        resolved["data"], safe_root=args.safe_root, must_exist=True
    )
    checkpoint_path = assert_safe_path(
        resolved["checkpoint"], safe_root=args.safe_root, must_exist=True
    )
    viewer = assert_safe_path(
        resolved["viewer"],
        safe_root=args.safe_root,
        must_exist=True,
        writable=True,
    )
    report_output = assert_safe_path(
        resolved["report_output"],
        safe_root=args.safe_root,
        writable=True,
    )
    reference_path = assert_safe_path(
        resolved["reference_metrics"],
        safe_root=args.safe_root,
        must_exist=True,
    )
    selection_path = assert_safe_path(
        resolved["selection_manifest"],
        safe_root=args.safe_root,
        must_exist=True,
    )
    report_output.mkdir(parents=True, exist_ok=True)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection, experiment=EXPERIMENT, splits=SPLITS)
    data_manifest = json.loads(
        (data / "manifest.json").read_text(encoding="utf-8")
    )
    metadata_by_id = {
        str(sample["id"]): sample for sample in data_manifest["samples"]
    }
    references = load_reference_metrics(reference_path)

    public_root = viewer / "public"
    output_root = public_root / "data" / "exp29"
    output_root.mkdir(parents=True, exist_ok=True)
    prepared: list[Dict[str, Any]] = []
    for split in SPLITS:
        for selected in selection["splits"][split]:
            sample_id = str(selected["id"])
            metadata = metadata_by_id.get(sample_id)
            if metadata is None or metadata.get("split") != split:
                raise ValueError(f"{sample_id} is absent from the {split} split")
            if sample_id not in references:
                raise ValueError(f"{sample_id} is absent from reference metrics")
            image, gt_points = load_raw_sample(
                data,
                metadata,
                args.height,
                args.width,
                args.safe_root,
            )
            crop = _locked_crop(
                selected,
                width=args.width,
                height=args.height,
                expected_size=args.crop_size,
            )
            colors = (
                image.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
            ).round().astype(np.uint8)
            sample_output = output_root / sample_id
            sample_output.mkdir(parents=True, exist_ok=True)
            rgb_output = sample_output / "source_rgb.jpg"
            Image.fromarray(colors).save(rgb_output, quality=95, subsampling=0)
            prepared.append(
                {
                    "id": sample_id,
                    "split": split,
                    "selection": dict(selected),
                    "image": image,
                    "gt_points": gt_points,
                    "crop": crop,
                    "colors": colors,
                    "rgb_url": _public_url(rgb_output, public_root),
                }
            )

    checkpoint_digest = sha256_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("step", -1)) != 2800:
        raise ValueError(
            f"Exp29 viewer requires checkpoint step 2800, got "
            f"{checkpoint.get('step')}"
        )
    checkpoint_bound = float(
        checkpoint.get("args", {}).get(
            "smooth_log_depth_residual_bound",
            0.0,
        )
    )
    if not math.isclose(
        checkpoint_bound,
        args.smooth_log_depth_residual_bound,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError(
            f"Residual bound mismatch: checkpoint={checkpoint_bound}, "
            f"export={args.smooth_log_depth_residual_bound}"
        )

    device = torch.device(args.device)
    model = construct_checkpoint_model(checkpoint, device=device)
    before_state = capture_batch_norm_running_state(model.ssr)
    before_sha = batch_norm_state_sha256(before_state)
    samples: list[Dict[str, Any]] = []
    checksums: list[Dict[str, Any]] = []
    metric_records: Dict[str, Any] = {}
    validation_differences: Dict[str, Dict[str, Dict[str, float]]] = {}

    for sample in prepared:
        gt_metrics, expected_structure_pixels = _ground_truth_metrics(
            sample["gt_points"],
            crop=sample["crop"],
            boundary_threshold=args.boundary_threshold,
        )
        gt_points = sample["gt_points"].float().cpu().numpy()
        gt_valid = (
            np.isfinite(gt_points).all(axis=-1)
            & (gt_points[..., 2] > 0)
        )
        serialized_gt_points = gt_points.copy()
        serialized_gt_points[~gt_valid] = 0.0
        ground_truth_output = (
            output_root / sample["id"] / "ground_truth.ply"
        )
        write_binary_ply(
            ground_truth_output,
            serialized_gt_points,
            sample["colors"],
            width=args.width,
            height=args.height,
        )
        _verify_ply(
            ground_truth_output,
            serialized_gt_points,
            sample["colors"],
        )
        ground_truth_asset = point_asset(
            ground_truth_output,
            public_root=public_root,
            points=gt_points,
            checkpoint_digest=_array_sha256(gt_points),
            alignment={"scale": 1.0, "zShift": 0.0},
            metrics=gt_metrics,
        )
        ground_truth_asset["validPointCount"] = int(gt_valid.sum())
        checksums.append(
            {
                "sample": sample["id"],
                "split": sample["split"],
                "stage": "ground_truth",
                "k": None,
                "url": ground_truth_asset["url"],
                "bytes": ground_truth_output.stat().st_size,
                "sha256": ground_truth_asset["sha256"],
            }
        )
        raw_by_k, metrics_by_k, structure_pixels = predict_and_measure(
            model,
            sample["image"].to(device),
            sample["gt_points"],
            crop=sample["crop"],
            num_tokens=args.num_tokens,
            steps=EXPECTED_STEPS,
            boundary_threshold=args.boundary_threshold,
            smooth_log_depth_residual_bound=(
                args.smooth_log_depth_residual_bound
            ),
        )
        if structure_pixels != expected_structure_pixels:
            raise ValueError(
                f"{sample['id']} structure mask changed between GT and prediction"
            )
        reference = references[sample["id"]]
        validation_differences[sample["id"]] = {}
        for step in EXPECTED_STEPS:
            validation_differences[sample["id"]][str(step)] = (
                validate_full_metrics(
                    sample_id=sample["id"],
                    step=step,
                    actual=metrics_by_k[step]["full"],
                    reference=reference,
                    tolerance=args.metric_tolerance,
                )
            )
            metrics_by_k[step]["full"] = canonical_full_metrics(
                reference,
                step=step,
                pixels=int(metrics_by_k[step]["full"]["pixels"]),
            )

        k0_rel = float(metrics_by_k[0]["full"]["point_rel"])
        stage_assets: Dict[str, Any] = {}
        for step in EXPECTED_STEPS:
            output = (
                output_root / sample["id"] / f"final_k{step}.ply"
            )
            write_binary_ply(
                output,
                raw_by_k[step],
                sample["colors"],
                width=args.width,
                height=args.height,
            )
            _verify_ply(output, raw_by_k[step], sample["colors"])
            metrics = metrics_by_k[step]
            asset = point_asset(
                output,
                public_root=public_root,
                points=raw_by_k[step],
                checkpoint_digest=checkpoint_digest,
                alignment=metrics["alignment"],
                metrics={
                    scope: metrics[scope]
                    for scope in ("full", "crop", "structure")
                },
            )
            current_rel = float(metrics["full"]["point_rel"])
            asset["pointRelReductionFromK0"] = (
                0.0
                if step == 0
                else (k0_rel - current_rel) / max(k0_rel, 1e-12)
            )
            stage_assets[str(step)] = asset
            checksums.append(
                {
                    "sample": sample["id"],
                    "split": sample["split"],
                    "stage": "final",
                    "k": step,
                    "url": asset["url"],
                    "bytes": output.stat().st_size,
                    "sha256": asset["sha256"],
                }
            )

        actual_improvement = float(
            stage_assets["3"]["pointRelReductionFromK0"]
        )
        expected_improvement = float(
            sample["selection"]["relativeImprovement"]
        )
        if not math.isclose(
            expected_improvement,
            actual_improvement,
            rel_tol=2e-4,
            abs_tol=5e-5,
        ):
            raise ValueError(
                f"{sample['id']} selected K=3 improvement "
                f"{expected_improvement} != {actual_improvement}"
            )
        outcome = "improved" if actual_improvement > 0 else "degraded"
        if sample["selection"].get("outcome") != outcome:
            raise ValueError(f"{sample['id']} outcome changed during export")
        selection_record = {
            "metric": str(selection["metric"]),
            "policy": str(selection["policy"][sample["split"]]),
            "rank": int(sample["selection"]["rank"]),
            "total": int(sample["selection"]["total"]),
            "relativeImprovement": actual_improvement,
            "outcome": outcome,
            "structureDescription": str(
                sample["selection"]["structureDescription"]
            ),
        }
        sample_record = {
            "id": sample["id"],
            "split": sample["split"],
            "label": f"图片 {int(sample['selection']['picture'])}",
            "order": int(sample["selection"]["picture"]),
            "description": (
                f"{selection_record['structureDescription']} · "
                f"{'改善样本' if outcome == 'improved' else '退化样本'}"
            ),
            "websiteEnabled": True,
            "cropXYXY": sample["crop"],
            "structureMask": {
                "type": "near_quantile",
                "quantile": 0.5,
                "pixels": structure_pixels,
            },
            "rgbUrl": sample["rgb_url"],
            "selection": selection_record,
            "groundTruth": ground_truth_asset,
            "stages": {"final": stage_assets},
        }
        samples.append(sample_record)
        metric_records[sample["id"]] = {
            "split": sample["split"],
            "cropXYXY": sample["crop"],
            "selection": selection_record,
            "groundTruth": ground_truth_asset["metrics"],
            "final": {
                step: stage_assets[step]["metrics"]
                for step in stage_assets
            },
        }

    after_state = capture_batch_norm_running_state(model.ssr)
    after_sha = batch_norm_state_sha256(after_state)
    buffers_unchanged = batch_norm_states_equal(before_state, after_state)
    if not buffers_unchanged or before_sha != after_sha:
        raise RuntimeError("SSR BatchNorm running buffers changed during export")
    del model, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    order_by_split = {
        split: [
            sample["id"] for sample in samples if sample["split"] == split
        ]
        for split in SPLITS
    }
    manifest = {
        "version": 2,
        "experiment": EXPERIMENT,
        "coordinateSpace": {
            "stored": "raw_moge_camera_xyz",
            "aligned": "P_aligned = scale * P_raw + (0, 0, zShift)",
            "threeDisplay": "[x, -y, -z]",
        },
        "voxelization": {
            "depthScale": VOXEL_DEPTH_SCALE,
            "spconvOrder": ["batch", "depth", "row", "column"],
            "depthCoordinate": "round(depthScale * log(Z_raw))",
        },
        "resolution": {"width": args.width, "height": args.height},
        "steps": list(EXPECTED_STEPS),
        "availableSplits": list(SPLITS),
        "defaultSplit": "train",
        "websiteSampleOrderBySplit": order_by_split,
        "availableStages": ["final"],
        "defaultStages": {"left": "final", "right": "final"},
        "samples": samples,
        "provenance": {
            "sourceExperiment": EXPERIMENT,
            "checkpointStep": 2800,
            "checkpointSha256": checkpoint_digest,
            "inferencePolicy": "stored_batch_norm_running_statistics",
            "selectionMetric": str(selection["metric"]),
            "selectionPolicy": (
                "训练前GT细杆显著性优先，并在每个划分保留改善与退化案例"
            ),
            "smoothLogDepthResidualBound": (
                args.smooth_log_depth_residual_bound
            ),
            "note": (
                "Exp29从Exp28 step 1000继续联合训练；step 2800由训练集"
                "细结构K=3 Point Rel选出。样本按训练前GT中的长细杆显著性"
                "重选；窗口A展示Hypersim真实点云，B/C比较同一检查点的"
                "不同K结果，不代表验证或测试泛化成功。"
            ),
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_path = output_root / "pointcloud_sha256.json"
    checksum_path.write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "manifest": str(manifest_path),
        "selectionManifest": str(selection_path),
        "splits": order_by_split,
        "checkpointStep": 2800,
        "checkpointSha256": checkpoint_digest,
        "inferencePolicy": "stored_batch_norm_running_statistics",
        "smoothLogDepthResidualBound": (
            args.smooth_log_depth_residual_bound
        ),
        "batchNormRunningState": {
            "beforeSha256": before_sha,
            "afterSha256": after_sha,
            "unchanged": buffers_unchanged,
        },
        "evaluationSteps": list(EXPECTED_STEPS),
        "pointCloudCount": len(checksums),
        "pointCloudBytes": sum(int(item["bytes"]) for item in checksums),
        "pointCloudChecksums": str(checksum_path),
        "metricValidation": {
            "requestedTolerance": args.metric_tolerance,
            "maxAbsoluteDifference": {
                metric: max(
                    validation_differences[sample_id][step][metric]
                    for sample_id in validation_differences
                    for step in validation_differences[sample_id]
                )
                for metric in (
                    "point_rel",
                    "depth_rel",
                    "depth_delta_1.01",
                    "depth_delta_1.25",
                    "boundary_f1",
                )
            },
        },
        "samples": [
            {
                "id": sample["id"],
                "split": sample["split"],
                "cropXYXY": sample["cropXYXY"],
                "selection": sample["selection"],
            }
            for sample in samples
        ],
    }
    (report_output / "selected_sample_metrics.json").write_text(
        json.dumps(metric_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_output / "pointcloud_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
