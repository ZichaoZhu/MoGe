from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping

import numpy as np
import torch
from PIL import Image

from moge.model.ssr import capture_batch_norm_running_state
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.export_exp29_viewer import (
    _array_sha256,
    _ground_truth_metrics,
    _locked_crop,
    _public_url,
    _verify_ply,
)
from tools.moge3.export_exp9_pointclouds import (
    EXPECTED_STEPS,
    VOXEL_DEPTH_SCALE,
    point_asset,
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


EXPERIMENT = "exp30_hypersim_100_long_two_stage_overfit"
SPLITS = ("train", "val", "test")
STAGES = ("initial", "stage1", "final")
DEFAULT_RELATIVE_PATHS = {
    "data": "experiment/exp11_hypersim_100_train_staged_joint_overfit/data",
    "initial_checkpoint": (
        "experiment/stage5_scaling_validation/runs/"
        "exp30_hypersim_100_long_two_stage_overfit/artifacts/training/"
        "stage1/attempt_00/initial_checkpoint.pt"
    ),
    "stage1_checkpoint": (
        "experiment/stage5_scaling_validation/runs/"
        "exp30_hypersim_100_long_two_stage_overfit/artifacts/training/"
        "stage1/attempt_09/checkpoint.pt"
    ),
    "final_checkpoint": (
        "experiment/stage5_scaling_validation/runs/"
        "exp30_hypersim_100_long_two_stage_overfit/artifacts/training/"
        "stage2/attempt_03/milestones/step_030000.pt"
    ),
    "initial_metrics": (
        "experiment/stage5_scaling_validation/runs/"
        "exp30_hypersim_100_long_two_stage_overfit/artifacts/evaluation/"
        "initial/per_frame_metrics.csv"
    ),
    "stage1_metrics": (
        "experiment/stage5_scaling_validation/runs/"
        "exp30_hypersim_100_long_two_stage_overfit/artifacts/evaluation/"
        "stage1_best/per_frame_metrics.csv"
    ),
    "final_metrics": (
        "experiment/stage5_scaling_validation/runs/"
        "exp30_hypersim_100_long_two_stage_overfit/artifacts/evaluation/"
        "joint_terminal/per_frame_metrics.csv"
    ),
    "selection_manifest": (
        "experiment/stage5_scaling_validation/runs/"
        "exp30_hypersim_100_long_two_stage_overfit/viewer_selection.json"
    ),
    "viewer": (
        "experiment/stage2_training_strategy/runs/"
        "exp9_thin_structure_single_image_staged_overfit/viewer"
    ),
    "report_output": (
        "experiment/stage5_scaling_validation/runs/"
        "exp30_hypersim_100_long_two_stage_overfit/results/viewer_export"
    ),
}
LEGACY_RELATIVE_PATHS = {
    "viewer": "experiment/exp9_thin_structure_single_image_staged_overfit/viewer",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Exp30 initial, detached-stage best and joint terminal "
            "point clouds for the split-aware browser."
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
    if explicit is not None:
        return explicit
    preferred = project_root / DEFAULT_RELATIVE_PATHS[key]
    legacy = LEGACY_RELATIVE_PATHS.get(key)
    if not preferred.exists() and legacy is not None:
        return project_root / legacy
    return preferred


def _stage_paths(paths: Mapping[str, Path]) -> Dict[str, tuple[Path, Path, int]]:
    return {
        "initial": (
            paths["initial_checkpoint"],
            paths["initial_metrics"],
            0,
        ),
        "stage1": (
            paths["stage1_checkpoint"],
            paths["stage1_metrics"],
            20_000,
        ),
        "final": (
            paths["final_checkpoint"],
            paths["final_metrics"],
            30_000,
        ),
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from moge.scripts.train_hypersim_smallset_v3 import load_raw_sample

    if min(
        args.height,
        args.width,
        args.num_tokens,
        args.crop_size,
    ) <= 0:
        raise ValueError("Image, token and crop parameters must be positive")
    if args.smooth_log_depth_residual_bound <= 0:
        raise ValueError("Exp30 requires the configured positive residual bound")

    project_root = assert_safe_path(
        args.project_root,
        safe_root=args.safe_root,
        must_exist=True,
    )
    paths = {
        key: _resolve(
            getattr(args, key),
            project_root=project_root,
            key=key,
        )
        for key in DEFAULT_RELATIVE_PATHS
    }
    for key in (
        "data",
        "initial_checkpoint",
        "stage1_checkpoint",
        "final_checkpoint",
        "initial_metrics",
        "stage1_metrics",
        "final_metrics",
        "selection_manifest",
    ):
        paths[key] = assert_safe_path(
            paths[key],
            safe_root=args.safe_root,
            must_exist=True,
        )
    paths["viewer"] = assert_safe_path(
        paths["viewer"],
        safe_root=args.safe_root,
        must_exist=True,
        writable=True,
    )
    paths["report_output"] = assert_safe_path(
        paths["report_output"],
        safe_root=args.safe_root,
        writable=True,
    )
    paths["report_output"].mkdir(parents=True, exist_ok=True)

    selection = json.loads(
        paths["selection_manifest"].read_text(encoding="utf-8")
    )
    validate_selection(selection, experiment=EXPERIMENT, splits=SPLITS)
    data_manifest = json.loads(
        (paths["data"] / "manifest.json").read_text(encoding="utf-8")
    )
    metadata_by_id = {
        str(sample["id"]): sample for sample in data_manifest["samples"]
    }
    references = {
        stage: load_reference_metrics(metrics)
        for stage, (_, metrics, _) in _stage_paths(paths).items()
    }

    public_root = paths["viewer"] / "public"
    output_root = public_root / "data" / "exp30"
    output_root.mkdir(parents=True, exist_ok=True)
    prepared: list[Dict[str, Any]] = []
    for split in SPLITS:
        for selected in selection["splits"][split]:
            sample_id = str(selected["id"])
            metadata = metadata_by_id.get(sample_id)
            if metadata is None or metadata.get("split") != split:
                raise ValueError(f"{sample_id} is absent from the {split} split")
            if any(sample_id not in references[stage] for stage in STAGES):
                raise ValueError(
                    f"{sample_id} is absent from at least one stage metric table"
                )
            image, gt_points = load_raw_sample(
                paths["data"],
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
            Image.fromarray(colors).save(
                rgb_output,
                quality=95,
                subsampling=0,
            )
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

    checksums: list[Dict[str, Any]] = []
    ground_truth_assets: Dict[str, Dict[str, Any]] = {}
    structure_pixels: Dict[str, int] = {}
    metric_records: Dict[str, Any] = {}
    for sample in prepared:
        metrics, pixels = _ground_truth_metrics(
            sample["gt_points"],
            crop=sample["crop"],
            boundary_threshold=args.boundary_threshold,
        )
        structure_pixels[sample["id"]] = pixels
        gt_points = sample["gt_points"].float().numpy()
        valid = np.isfinite(gt_points).all(axis=-1) & (gt_points[..., 2] > 0)
        serialized = gt_points.copy()
        serialized[~valid] = 0.0
        output = output_root / sample["id"] / "ground_truth.ply"
        write_binary_ply(
            output,
            serialized,
            sample["colors"],
            width=args.width,
            height=args.height,
        )
        _verify_ply(output, serialized, sample["colors"])
        asset = point_asset(
            output,
            public_root=public_root,
            points=gt_points,
            checkpoint_digest=_array_sha256(gt_points),
            alignment={"scale": 1.0, "zShift": 0.0},
            metrics=metrics,
        )
        asset["validPointCount"] = int(valid.sum())
        ground_truth_assets[sample["id"]] = asset
        checksums.append(
            {
                "sample": sample["id"],
                "split": sample["split"],
                "stage": "ground_truth",
                "k": None,
                "url": asset["url"],
                "bytes": output.stat().st_size,
                "sha256": asset["sha256"],
            }
        )
        metric_records[sample["id"]] = {
            "split": sample["split"],
            "cropXYXY": sample["crop"],
            "groundTruth": metrics,
        }

    device = torch.device(args.device)
    stage_assets: Dict[str, Dict[str, Dict[str, Any]]] = {
        sample["id"]: {} for sample in prepared
    }
    stage_reports: Dict[str, Any] = {}
    validation_differences: Dict[
        str,
        Dict[str, Dict[str, Dict[str, float]]],
    ] = {}
    zero_identity_max_error = 0.0

    for stage, (checkpoint_path, _, expected_step) in _stage_paths(paths).items():
        checkpoint_digest = sha256_file(checkpoint_path)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        actual_step = int(checkpoint.get("step", -1))
        if actual_step != expected_step:
            raise ValueError(
                f"{stage} checkpoint step {actual_step} != {expected_step}"
            )
        model = construct_checkpoint_model(checkpoint, device=device)
        before_state = capture_batch_norm_running_state(model.ssr)
        before_sha = batch_norm_state_sha256(before_state)
        validation_differences[stage] = {}

        for sample in prepared:
            raw_by_k, metrics_by_k, pixels = predict_and_measure(
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
            if pixels != structure_pixels[sample["id"]]:
                raise RuntimeError("GT structure mask changed between stages")
            reference = references[stage][sample["id"]]
            validation_differences[stage][sample["id"]] = {}
            for refinement_step in EXPECTED_STEPS:
                validation_differences[stage][sample["id"]][
                    str(refinement_step)
                ] = validate_full_metrics(
                    sample_id=sample["id"],
                    step=refinement_step,
                    actual=metrics_by_k[refinement_step]["full"],
                    reference=reference,
                    tolerance=args.metric_tolerance,
                )
                metrics_by_k[refinement_step]["full"] = (
                    canonical_full_metrics(
                        reference,
                        step=refinement_step,
                        pixels=int(
                            metrics_by_k[refinement_step]["full"]["pixels"]
                        ),
                    )
                )

            if stage == "initial":
                initial_points = raw_by_k[0]
                for refinement_step in EXPECTED_STEPS[1:]:
                    difference = float(
                        np.max(
                            np.abs(
                                raw_by_k[refinement_step] - initial_points
                            )
                        )
                    )
                    zero_identity_max_error = max(
                        zero_identity_max_error,
                        difference,
                    )
                    if difference != 0.0:
                        raise RuntimeError(
                            f"{sample['id']} initial K={refinement_step} "
                            f"is not identical to K=0: {difference}"
                        )
                exported_steps = (0,)
            else:
                exported_steps = EXPECTED_STEPS

            k0_rel = float(metrics_by_k[0]["full"]["point_rel"])
            assets: Dict[str, Any] = {}
            for refinement_step in exported_steps:
                output = (
                    output_root
                    / sample["id"]
                    / f"{stage}_k{refinement_step}.ply"
                )
                points = raw_by_k[refinement_step]
                write_binary_ply(
                    output,
                    points,
                    sample["colors"],
                    width=args.width,
                    height=args.height,
                )
                _verify_ply(output, points, sample["colors"])
                metrics = metrics_by_k[refinement_step]
                asset = point_asset(
                    output,
                    public_root=public_root,
                    points=points,
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
                    if refinement_step == 0
                    else (k0_rel - current_rel) / max(k0_rel, 1e-12)
                )
                assets[str(refinement_step)] = asset
                checksums.append(
                    {
                        "sample": sample["id"],
                        "split": sample["split"],
                        "stage": stage,
                        "k": refinement_step,
                        "url": asset["url"],
                        "bytes": output.stat().st_size,
                        "sha256": asset["sha256"],
                    }
                )
            if stage == "initial":
                assets.update(
                    {
                        str(step): {"alias": "initial.0"}
                        for step in EXPECTED_STEPS[1:]
                    }
                )
            stage_assets[sample["id"]][stage] = assets
            metric_records[sample["id"]][stage] = {
                str(step): (
                    assets[str(step)]["metrics"]
                    if "metrics" in assets[str(step)]
                    else assets["0"]["metrics"]
                )
                for step in EXPECTED_STEPS
            }

        after_state = capture_batch_norm_running_state(model.ssr)
        after_sha = batch_norm_state_sha256(after_state)
        buffers_unchanged = batch_norm_states_equal(
            before_state,
            after_state,
        )
        if not buffers_unchanged or before_sha != after_sha:
            raise RuntimeError(
                f"{stage} SSR BatchNorm buffers changed during export"
            )
        stage_reports[stage] = {
            "checkpoint": str(checkpoint_path),
            "checkpointStep": expected_step,
            "checkpointSha256": checkpoint_digest,
            "batchNormRunningState": {
                "beforeSha256": before_sha,
                "afterSha256": after_sha,
                "unchanged": buffers_unchanged,
            },
        }
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    sample_records = []
    for sample in prepared:
        selected = sample["selection"]
        actual_improvement = float(
            stage_assets[sample["id"]]["stage1"]["3"][
                "pointRelReductionFromK0"
            ]
        )
        expected_improvement = float(selected["relativeImprovement"])
        if not math.isclose(
            expected_improvement,
            actual_improvement,
            rel_tol=2e-4,
            abs_tol=5e-5,
        ):
            raise ValueError(
                f"{sample['id']} stage1 selection improvement "
                f"{expected_improvement} != {actual_improvement}"
            )
        outcome = "improved" if actual_improvement > 0 else "degraded"
        if selected.get("outcome") != outcome:
            raise ValueError(f"{sample['id']} selection outcome changed")
        selection_record = {
            "metric": str(selection["metric"]),
            "policy": str(selection["policy"][sample["split"]]),
            "rank": int(selected["rank"]),
            "total": int(selected["total"]),
            "relativeImprovement": actual_improvement,
            "outcome": outcome,
            "structureDescription": str(selected["structureDescription"]),
        }
        sample_records.append(
            {
                "id": sample["id"],
                "split": sample["split"],
                "label": f"图片 {int(selected['picture'])}",
                "order": int(selected["picture"]),
                "description": (
                    f"{selection_record['structureDescription']} · "
                    f"阶段一{'改善' if outcome == 'improved' else '退化'}"
                ),
                "websiteEnabled": True,
                "cropXYXY": sample["crop"],
                "structureMask": {
                    "type": "near_quantile",
                    "quantile": 0.5,
                    "pixels": structure_pixels[sample["id"]],
                },
                "rgbUrl": sample["rgb_url"],
                "selection": selection_record,
                "groundTruth": ground_truth_assets[sample["id"]],
                "stages": stage_assets[sample["id"]],
            }
        )
        metric_records[sample["id"]]["selection"] = selection_record

    order_by_split = {
        split: [
            sample["id"]
            for sample in sample_records
            if sample["split"] == split
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
        "availableStages": list(STAGES),
        "defaultStages": {"left": "stage1", "right": "final"},
        "samples": sample_records,
        "provenance": {
            "sourceExperiment": EXPERIMENT,
            "initialization": "官方 MoGe-2 ViT-L 与零初始化 SSR",
            "checkpointStep": 20_000,
            "checkpointSha256": stage_reports["stage1"][
                "checkpointSha256"
            ],
            "inferencePolicy": "stored_batch_norm_running_statistics",
            "selectionMetric": str(selection["metric"]),
            "selectionPolicy": str(selection["selectionBasis"]),
            "smoothLogDepthResidualBound": (
                args.smooth_log_depth_residual_bound
            ),
            "note": (
                "Exp30使用100张Hypersim进行长程两阶段训练。阶段一step "
                "20000是全局最佳；联合训练实际终点为step 30000，但未超过"
                "阶段一并使Base退化。窗口A为GT，B/C默认直接比较阶段一与"
                "联合终点；验证与测试不构成泛化成功证据。"
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

    max_differences = {}
    for stage in STAGES:
        max_differences[stage] = {
            metric: max(
                validation_differences[stage][sample_id][step][metric]
                for sample_id in validation_differences[stage]
                for step in validation_differences[stage][sample_id]
            )
            for metric in (
                "point_rel",
                "depth_rel",
                "depth_delta_1.01",
                "depth_delta_1.25",
                "boundary_f1",
            )
        }
    report = {
        "status": "complete",
        "experiment": EXPERIMENT,
        "manifest": str(manifest_path),
        "selectionManifest": str(paths["selection_manifest"]),
        "splits": order_by_split,
        "stages": stage_reports,
        "inferencePolicy": "stored_batch_norm_running_statistics",
        "smoothLogDepthResidualBound": (
            args.smooth_log_depth_residual_bound
        ),
        "initialZeroIdentityMaxError": zero_identity_max_error,
        "evaluationSteps": list(EXPECTED_STEPS),
        "pointCloudCount": len(checksums),
        "logicalPointCloudCount": len(prepared) * (1 + 3 * 4),
        "pointCloudBytes": sum(int(item["bytes"]) for item in checksums),
        "pointCloudChecksums": str(checksum_path),
        "metricValidation": {
            "requestedTolerance": args.metric_tolerance,
            "maxAbsoluteDifferenceByStage": max_differences,
        },
        "samples": [
            {
                "id": sample["id"],
                "split": sample["split"],
                "cropXYXY": sample["cropXYXY"],
                "selection": sample["selection"],
            }
            for sample in sample_records
        ],
    }
    (paths["report_output"] / "selected_sample_metrics.json").write_text(
        json.dumps(metric_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (paths["report_output"] / "pointcloud_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
