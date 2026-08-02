from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.export_exp9_pointclouds import (
    EXPECTED_STEPS,
    VOXEL_DEPTH_SCALE,
    point_asset,
    read_binary_ply,
    write_binary_ply,
)

if TYPE_CHECKING:
    from moge.model.v3 import MoGeModel


SPLITS = ("train", "val", "test")
SAMPLE_DISPLAY = {
    "train": {
        "label": "训练样本",
        "description": "悬空楼梯与支架（与 Exp9 样本 04 同帧）",
    },
    "val": {
        "label": "验证样本",
        "description": "窗格、横向木条与椅架",
    },
    "test": {
        "label": "测试样本",
        "description": "餐桌、椅架与吊灯细杆",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export Exp12 step-3000/step-3300 point maps and selected-scope "
            "metrics for the multi-experiment browser viewer."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--final-checkpoint", type=Path, required=True)
    parser.add_argument("--viewer", type=Path, required=True)
    parser.add_argument("--report-output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument(
        "--evaluation-steps",
        type=int,
        nargs="+",
        default=list(EXPECTED_STEPS),
    )
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def crop_from_selection(entry: Mapping[str, Any]) -> list[int]:
    return [
        int(entry["crop_x0"]),
        int(entry["crop_y0"]),
        int(entry["crop_x1"]),
        int(entry["crop_y1"]),
    ]


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
) -> tuple[MoGeModel, Mapping[str, Any]]:
    from moge.model.v3 import MoGeModel

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    pretrained = checkpoint.get("pretrained")
    if not pretrained:
        raise ValueError(f"{checkpoint_path} does not identify its base model")
    model = MoGeModel.from_pretrained(str(pretrained)).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    return model, checkpoint


@torch.no_grad()
def predict_and_measure(
    model: MoGeModel,
    image: torch.Tensor,
    gt_points: torch.Tensor,
    *,
    crop: Sequence[int],
    num_tokens: int,
    steps: Sequence[int],
    boundary_threshold: float,
) -> tuple[
    Dict[int, np.ndarray],
    Dict[int, Dict[str, Any]],
    torch.Tensor,
]:
    from moge.scripts.overfit_hypersim_staged_v3 import (
        _scope_metrics,
        build_structure_mask,
    )
    from moge.train.losses_v3 import solve_global_affine_alignment

    output = model(
        image[None],
        num_tokens=num_tokens,
        num_refinement_steps=max(steps),
        return_intermediates=True,
    )
    sequence = output["points_sequence"]
    gt = gt_points.to(image.device)
    full_valid = torch.isfinite(gt).all(dim=-1) & (gt[..., 2] > 0)
    x0, y0, x1, y1 = (int(value) for value in crop)
    crop_valid = full_valid[y0:y1, x0:x1]
    structure_mask = build_structure_mask(
        gt,
        crop,
        {"type": "near_quantile", "quantile": 0.5},
    )

    raw_by_k: Dict[int, np.ndarray] = {}
    metrics_by_k: Dict[int, Dict[str, Any]] = {}
    for step in steps:
        prediction = sequence[step][0].float()
        alignment = solve_global_affine_alignment(prediction[None], gt[None])
        aligned = alignment.apply(prediction[None])[0]
        raw_by_k[step] = prediction.detach().cpu().numpy()
        metrics_by_k[step] = {
            "full": _scope_metrics(
                aligned,
                gt,
                full_valid,
                boundary_threshold=boundary_threshold,
            ),
            "crop": _scope_metrics(
                aligned[y0:y1, x0:x1],
                gt[y0:y1, x0:x1],
                crop_valid,
                boundary_threshold=boundary_threshold,
            ),
            "structure": _scope_metrics(
                aligned[y0:y1, x0:x1],
                gt[y0:y1, x0:x1],
                structure_mask,
                boundary_threshold=boundary_threshold,
            ),
            "alignment": {
                "scale": float(alignment.scale.mean().item()),
                "zShift": float(alignment.shift[..., 2].mean().item()),
            },
        }
    return raw_by_k, metrics_by_k, structure_mask.detach().cpu()


def export_stage(
    *,
    stage: str,
    model: MoGeModel,
    checkpoint_digest: str,
    samples: Sequence[Dict[str, Any]],
    steps: Sequence[int],
    output_root: Path,
    public_root: Path,
    width: int,
    height: int,
    num_tokens: int,
    boundary_threshold: float,
) -> list[Dict[str, Any]]:
    checksum_records = []
    for sample in samples:
        raw_by_k, metrics_by_k, structure_mask = predict_and_measure(
            model,
            sample["image"].to(next(model.parameters()).device),
            sample["gt_points"],
            crop=sample["crop"],
            num_tokens=num_tokens,
            steps=steps,
            boundary_threshold=boundary_threshold,
        )
        sample["structure_pixels"] = int(structure_mask.sum().item())
        stage_assets: Dict[str, Any] = {}
        for step in steps:
            output = output_root / sample["id"] / f"{stage}_k{step}.ply"
            write_binary_ply(
                output,
                raw_by_k[step],
                sample["colors"],
                width=width,
                height=height,
            )
            decoded_points, decoded_colors, comments = read_binary_ply(output)
            np.testing.assert_array_equal(
                decoded_points,
                raw_by_k[step].reshape(-1, 3),
            )
            np.testing.assert_array_equal(
                decoded_colors,
                sample["colors"].reshape(-1, 3),
            )
            if comments.get("vertex_order") != "row_major_one_vertex_per_pixel":
                raise ValueError("PLY raster order was not preserved")
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
            stage_assets[str(step)] = asset
            checksum_records.append(
                {
                    "sample": sample["id"],
                    "split": sample["split"],
                    "stage": stage,
                    "k": step,
                    "url": asset["url"],
                    "bytes": output.stat().st_size,
                    "sha256": asset["sha256"],
                }
            )
        sample["stages"][stage] = stage_assets
    return checksum_records


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from moge.scripts.train_hypersim_smallset_v3 import load_raw_sample

    steps = tuple(sorted(set(int(step) for step in args.evaluation_steps)))
    if steps != EXPECTED_STEPS:
        raise ValueError(f"Expected exactly K={EXPECTED_STEPS}, received {steps}")
    if min(args.height, args.width, args.num_tokens) <= 0:
        raise ValueError("Image dimensions and token count must be positive")

    data = assert_safe_path(
        args.data,
        safe_root=args.safe_root,
        must_exist=True,
    )
    selection_path = assert_safe_path(
        args.selection,
        safe_root=args.safe_root,
        must_exist=True,
    )
    initial_checkpoint = assert_safe_path(
        args.initial_checkpoint,
        safe_root=args.safe_root,
        must_exist=True,
    )
    final_checkpoint = assert_safe_path(
        args.final_checkpoint,
        safe_root=args.safe_root,
        must_exist=True,
    )
    viewer = assert_safe_path(
        args.viewer,
        safe_root=args.safe_root,
        must_exist=True,
        writable=True,
    )
    report_output = assert_safe_path(
        args.report_output,
        safe_root=args.safe_root,
        writable=True,
    )
    report_output.mkdir(parents=True, exist_ok=True)
    public_root = viewer / "public"
    output_root = public_root / "data" / "exp12"
    output_root.mkdir(parents=True, exist_ok=True)

    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    metadata_by_id = {
        str(sample["id"]): sample for sample in manifest["samples"]
    }
    samples: list[Dict[str, Any]] = []
    for order, split in enumerate(SPLITS, start=1):
        selected = selection[split]
        sample_id = str(selected["id"])
        metadata = metadata_by_id.get(sample_id)
        if metadata is None or metadata["split"] != split:
            raise ValueError(f"Locked {split} sample is absent from its split")
        image, gt_points = load_raw_sample(
            data,
            metadata,
            args.height,
            args.width,
            args.safe_root,
        )
        colors = (
            image.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
        ).round().astype(np.uint8)
        sample_output = output_root / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)
        rgb_output = sample_output / "source_rgb.jpg"
        Image.fromarray(colors).save(rgb_output, quality=95, subsampling=0)
        samples.append(
            {
                "id": sample_id,
                "split": split,
                "label": SAMPLE_DISPLAY[split]["label"],
                "description": SAMPLE_DISPLAY[split]["description"],
                "order": order,
                "crop": crop_from_selection(selected),
                "image": image,
                "gt_points": gt_points,
                "colors": colors,
                "rgb_url": "/" + rgb_output.relative_to(public_root).as_posix(),
                "stages": {},
            }
        )

    device = torch.device(args.device)
    checkpoint_paths = {
        "initial": initial_checkpoint,
        "final": final_checkpoint,
    }
    checkpoint_steps: Dict[str, int] = {}
    checkpoint_digests: Dict[str, str] = {}
    checksum_records = []
    for stage, checkpoint_path in checkpoint_paths.items():
        digest = sha256_file(checkpoint_path)
        checkpoint_digests[stage] = digest
        model, checkpoint = load_checkpoint_model(
            checkpoint_path,
            device=device,
        )
        checkpoint_steps[stage] = int(checkpoint["step"])
        checksum_records.extend(
            export_stage(
                stage=stage,
                model=model,
                checkpoint_digest=digest,
                samples=samples,
                steps=steps,
                output_root=output_root,
                public_root=public_root,
                width=args.width,
                height=args.height,
                num_tokens=args.num_tokens,
                boundary_threshold=args.boundary_threshold,
            )
        )
        del model, checkpoint
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if checkpoint_steps != {"initial": 3000, "final": 3300}:
        raise ValueError(f"Unexpected checkpoint steps: {checkpoint_steps}")

    sample_records = [
        {
            "id": sample["id"],
            "split": sample["split"],
            "label": sample["label"],
            "order": sample["order"],
            "description": sample["description"],
            "websiteEnabled": True,
            "cropXYXY": sample["crop"],
            "structureMask": {
                "type": "near_quantile",
                "quantile": 0.5,
                "pixels": sample["structure_pixels"],
            },
            "rgbUrl": sample["rgb_url"],
            "stages": sample["stages"],
        }
        for sample in samples
    ]
    browser_manifest = {
        "version": 1,
        "experiment": "exp12_hypersim_100_immediate_joint_finetuning",
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
        "steps": list(steps),
        "websiteSampleOrder": [sample["id"] for sample in samples],
        "archivedInitialSampleIds": [sample["id"] for sample in samples],
        "samples": sample_records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(browser_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_path = output_root / "pointcloud_sha256.json"
    checksum_path.write_text(
        json.dumps(checksum_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "selectionStatus": selection["status"],
        "manifest": str(manifest_path),
        "samples": [
            {
                "id": sample["id"],
                "split": sample["split"],
                "cropXYXY": sample["crop"],
                "structurePixels": sample["structure_pixels"],
            }
            for sample in samples
        ],
        "checkpointSteps": checkpoint_steps,
        "checkpointSha256": checkpoint_digests,
        "evaluationSteps": list(steps),
        "pointCloudCount": len(checksum_records),
        "pointCloudBytes": sum(item["bytes"] for item in checksum_records),
        "pointCloudChecksums": str(checksum_path),
    }
    (report_output / "selected_sample_metrics.json").write_text(
        json.dumps(
            {
                sample["id"]: {
                    "split": sample["split"],
                    "cropXYXY": sample["cropXYXY"],
                    "stages": {
                        stage: {
                            step: sample["stages"][stage][step]["metrics"]
                            for step in sample["stages"][stage]
                        }
                        for stage in ("initial", "final")
                    },
                }
                for sample in sample_records
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
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
