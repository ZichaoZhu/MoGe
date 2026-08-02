from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import struct
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


EXPECTED_STEPS = (0, 1, 3, 5)
VOXEL_DEPTH_SCALE = 200
PLY_VERTEX_DTYPE = np.dtype(
    [
        ("x", "<f4"),
        ("y", "<f4"),
        ("z", "<f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
    ],
    align=False,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export raw Exp9 point maps and a browser manifest without "
            "modifying any training checkpoint"
        )
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--samples", nargs="+", required=True)
    parser.add_argument("--website-samples", nargs="+", required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=list(EXPECTED_STEPS))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def write_binary_ply(
    path: Path,
    points: np.ndarray,
    colors: np.ndarray,
    *,
    width: int,
    height: int,
) -> None:
    points = np.asarray(points, dtype=np.float32)
    colors = np.asarray(colors, dtype=np.uint8)
    if points.shape != (height, width, 3):
        raise ValueError(f"Unexpected point shape: {points.shape}")
    if colors.shape != (height, width, 3):
        raise ValueError(f"Unexpected color shape: {colors.shape}")
    if not np.isfinite(points).all():
        raise ValueError("Point map contains NaN or infinity")

    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = np.empty(height * width, dtype=PLY_VERTEX_DTYPE)
    flat_points = points.reshape(-1, 3)
    flat_colors = colors.reshape(-1, 3)
    vertices["x"], vertices["y"], vertices["z"] = flat_points.T
    vertices["red"], vertices["green"], vertices["blue"] = flat_colors.T
    header = "\n".join(
        (
            "ply",
            "format binary_little_endian 1.0",
            "comment generated_by MoGe-3 Exp9 exporter",
            f"comment raster_width {width}",
            f"comment raster_height {height}",
            "comment vertex_order row_major_one_vertex_per_pixel",
            f"element vertex {height * width}",
            "property float x",
            "property float y",
            "property float z",
            "property uchar red",
            "property uchar green",
            "property uchar blue",
            "end_header",
            "",
        )
    ).encode("ascii")
    with path.open("wb") as file:
        file.write(header)
        file.write(vertices.tobytes(order="C"))


def read_binary_ply(path: Path) -> Tuple[np.ndarray, np.ndarray, Dict[str, str]]:
    with path.open("rb") as file:
        header_lines = []
        while True:
            line = file.readline()
            if not line:
                raise ValueError("PLY header ended unexpectedly")
            decoded = line.decode("ascii").rstrip("\n")
            header_lines.append(decoded)
            if decoded == "end_header":
                break
        if header_lines[:2] != ["ply", "format binary_little_endian 1.0"]:
            raise ValueError("Only the Exp9 binary little-endian PLY format is supported")
        vertex_line = next(
            (line for line in header_lines if line.startswith("element vertex ")),
            None,
        )
        if vertex_line is None:
            raise ValueError("PLY has no vertex count")
        count = int(vertex_line.split()[-1])
        payload = file.read()
    expected_bytes = count * PLY_VERTEX_DTYPE.itemsize
    if len(payload) != expected_bytes:
        raise ValueError(
            f"PLY payload has {len(payload)} bytes, expected {expected_bytes}"
        )
    vertices = np.frombuffer(payload, dtype=PLY_VERTEX_DTYPE, count=count)
    points = np.stack((vertices["x"], vertices["y"], vertices["z"]), axis=-1)
    colors = np.stack(
        (vertices["red"], vertices["green"], vertices["blue"]), axis=-1
    )
    comments = {}
    for line in header_lines:
        if line.startswith("comment ") and " " in line[8:]:
            key, value = line[8:].split(" ", 1)
            comments[key] = value
    return points, colors, comments


def apply_scale_z_shift(
    points: np.ndarray,
    *,
    scale: float,
    z_shift: float,
) -> np.ndarray:
    aligned = np.asarray(points, dtype=np.float32) * np.float32(scale)
    aligned = aligned.copy()
    aligned[..., 2] += np.float32(z_shift)
    return aligned


def voxel_depth_bins(points: np.ndarray, depth_scale: int = VOXEL_DEPTH_SCALE) -> np.ndarray:
    depth = np.asarray(points, dtype=np.float32)[..., 2]
    if not np.isfinite(depth).all() or np.any(depth <= 0):
        raise ValueError("SSR voxelization requires finite positive Z")
    return np.rint(depth_scale * np.log(depth)).astype(np.int32)


def resolve_asset(
    stage_assets: Mapping[str, Mapping[str, Any]],
    step: int,
) -> Mapping[str, Any]:
    key = str(step)
    asset = stage_assets[key]
    if "alias" not in asset:
        return asset
    stage_name, target_key = str(asset["alias"]).split(".", 1)
    if stage_name != "initial":
        raise ValueError(f"Unsupported asset alias: {asset['alias']}")
    return stage_assets[target_key]


def discover_sample_directory(experiment_root: Path, token: str) -> Path:
    matches = sorted(
        path
        for path in experiment_root.iterdir()
        if path.is_dir() and path.name.startswith(f"{token}_")
    )
    if len(matches) != 1:
        raise ValueError(
            f"Sample token {token!r} matched {len(matches)} experiment directories"
        )
    return matches[0]


def checkpoint_sha_lookup(experiment_root: Path) -> Dict[str, str]:
    records = json.loads(
        (experiment_root / "checkpoint_manifest.json").read_text(encoding="utf-8")
    )
    return {str(record["path"]): str(record["sha256"]) for record in records}


def checkpoint_sha(
    lookup: Mapping[str, str],
    experiment_root: Path,
    checkpoint: Path,
) -> str:
    relative = checkpoint.relative_to(experiment_root).as_posix()
    if relative not in lookup:
        raise ValueError(f"Checkpoint is absent from the SHA-256 manifest: {relative}")
    return lookup[relative]


@torch.no_grad()
def predict_point_maps(
    model: torch.nn.Module,
    image: torch.Tensor,
    gt_points: torch.Tensor,
    *,
    num_tokens: int,
    steps: Sequence[int],
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict[str, float]]]:
    from moge.train.losses_v3 import solve_global_affine_alignment

    requested = tuple(sorted(set(int(step) for step in steps)))
    output = model(
        image[None],
        num_tokens=num_tokens,
        num_refinement_steps=max(requested),
        return_intermediates=True,
    )
    sequence = output["points_sequence"]
    gt = gt_points.to(image.device)
    predictions: Dict[int, np.ndarray] = {}
    alignments: Dict[int, Dict[str, float]] = {}
    for step in requested:
        prediction = sequence[step][0].float()
        alignment = solve_global_affine_alignment(prediction[None], gt[None])
        predictions[step] = prediction.detach().cpu().numpy()
        alignments[step] = {
            "scale": float(alignment.scale.mean().item()),
            "zShift": float(alignment.shift[..., 2].mean().item()),
        }
    return predictions, alignments


def validate_alignment_against_report(
    computed: Mapping[str, float],
    expected: Mapping[str, float],
) -> None:
    pairs = (
        (computed["scale"], expected["scale"], "scale"),
        (computed["zShift"], expected["z_shift"], "z shift"),
    )
    for actual, reference, label in pairs:
        if not math.isclose(
            float(actual),
            float(reference),
            rel_tol=2e-4,
            abs_tol=2e-5,
        ):
            raise ValueError(
                f"Recomputed {label} {actual} does not match report {reference}"
            )


def point_asset(
    path: Path,
    *,
    public_root: Path,
    points: np.ndarray,
    checkpoint_digest: str,
    alignment: Mapping[str, float],
    metrics: Mapping[str, Any],
) -> Dict[str, Any]:
    minimum = points.reshape(-1, 3).min(axis=0)
    maximum = points.reshape(-1, 3).max(axis=0)
    return {
        "url": "/" + path.relative_to(public_root).as_posix(),
        "pointCount": int(points.shape[0] * points.shape[1]),
        "sha256": sha256_file(path),
        "checkpointSha256": checkpoint_digest,
        "alignment": dict(alignment),
        "bounds": {
            "min": [float(value) for value in minimum],
            "max": [float(value) for value in maximum],
        },
        "metrics": metrics,
    }


def export_stage(
    *,
    sample_directory: Path,
    stage: str,
    model: torch.nn.Module,
    image: torch.Tensor,
    gt_points: torch.Tensor,
    colors: np.ndarray,
    steps: Sequence[int],
    output_directory: Path,
    public_root: Path,
    width: int,
    height: int,
    num_tokens: int,
    checkpoint_digest: str,
) -> Tuple[Dict[str, Dict[str, Any]], float]:
    report = json.loads(
        (sample_directory / "metrics" / f"{stage}.json").read_text(encoding="utf-8")
    )
    predictions, alignments = predict_point_maps(
        model,
        image,
        gt_points,
        num_tokens=num_tokens,
        steps=steps,
    )
    identity_error = 0.0
    if stage == "initial":
        base = predictions[0]
        identity_error = max(
            float(np.max(np.abs(predictions[step] - base)))
            for step in steps
            if step
        )
        if identity_error != 0.0:
            raise ValueError(
                f"Initial SSR is not an exact identity: max error {identity_error}"
            )

    assets: Dict[str, Dict[str, Any]] = {}
    exported_steps = (0,) if stage == "initial" else tuple(steps)
    for step in exported_steps:
        expected = report["metrics"][f"k{step}"]
        validate_alignment_against_report(
            alignments[step],
            expected["alignment"],
        )
        output = output_directory / f"{stage}_k{step}.ply"
        write_binary_ply(
            output,
            predictions[step],
            colors,
            width=width,
            height=height,
        )
        decoded_points, decoded_colors, comments = read_binary_ply(output)
        np.testing.assert_array_equal(
            decoded_points,
            predictions[step].reshape(-1, 3),
        )
        np.testing.assert_array_equal(decoded_colors, colors.reshape(-1, 3))
        if comments.get("vertex_order") != "row_major_one_vertex_per_pixel":
            raise ValueError("PLY raster-order metadata was not preserved")
        assets[str(step)] = point_asset(
            output,
            public_root=public_root,
            points=predictions[step],
            checkpoint_digest=checkpoint_digest,
            alignment=alignments[step],
            metrics={
                scope: expected[scope]
                for scope in ("full", "crop", "structure")
            },
        )
    if stage == "initial":
        for step in steps:
            if step:
                assets[str(step)] = {"alias": "initial.0"}
    return assets, identity_error


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from moge.scripts.overfit_hypersim_staged_v3 import (
        build_structure_mask,
        load_sample_and_selection,
    )
    from tools.moge3.visualize_staged_single_overfit import load_model

    steps = tuple(sorted(set(int(step) for step in args.steps)))
    if steps != EXPECTED_STEPS:
        raise ValueError(f"Expected exactly K={EXPECTED_STEPS}, received {steps}")
    if not set(args.website_samples).issubset(args.samples):
        raise ValueError("Website samples must be included in --samples")

    experiment_root = assert_safe_path(
        args.experiment_root,
        safe_root=args.safe_root,
        must_exist=True,
        writable=True,
    )
    config = json.loads((experiment_root / "config.json").read_text(encoding="utf-8"))
    width = int(config["model"]["width"])
    height = int(config["model"]["height"])
    num_tokens = int(config["model"]["num_tokens"])
    data = assert_safe_path(
        experiment_root / str(config["data"]["reuse"]),
        safe_root=args.safe_root,
        must_exist=True,
    )
    selection = assert_safe_path(
        experiment_root / str(config["data"]["selection"]),
        safe_root=args.safe_root,
        must_exist=True,
    )
    viewer = assert_safe_path(
        experiment_root / "viewer",
        safe_root=args.safe_root,
        writable=True,
    )
    public_root = viewer / "public"
    output_root = public_root / "data"
    output_root.mkdir(parents=True, exist_ok=True)

    checkpoint_digests = checkpoint_sha_lookup(experiment_root)
    website_tokens = [str(token) for token in args.website_samples]
    device = torch.device(args.device)
    sample_records = []
    checksum_records = []
    identity_errors: Dict[str, float] = {}

    for token in (str(value) for value in args.samples):
        sample_directory = discover_sample_directory(experiment_root, token)
        sample_config = json.loads(
            (sample_directory / "config.json").read_text(encoding="utf-8")
        )
        sample_id = str(sample_config["sample_id"])
        image, gt_points, _, entry = load_sample_and_selection(
            data,
            selection,
            sample_id,
            height=height,
            width=width,
            safe_root=args.safe_root,
        )
        crop = [int(value) for value in entry["crop_xyxy"]]
        structure = build_structure_mask(
            gt_points,
            crop,
            entry["display_mask"],
        )
        colors = (
            image.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
        ).round().astype(np.uint8)
        sample_output = output_root / sample_directory.name
        sample_output.mkdir(parents=True, exist_ok=True)
        rgb_output = sample_output / "source_rgb.jpg"
        Image.fromarray(colors).save(
            rgb_output,
            quality=95,
            subsampling=0,
        )

        initial_checkpoint = assert_safe_path(
            sample_directory / "checkpoints" / "initial.pt",
            safe_root=args.safe_root,
            must_exist=True,
        )
        model, _ = load_model(
            initial_checkpoint,
            pretrained=args.pretrained,
            device=device,
        )
        initial_assets, identity_error = export_stage(
            sample_directory=sample_directory,
            stage="initial",
            model=model,
            image=image.to(device),
            gt_points=gt_points,
            colors=colors,
            steps=steps,
            output_directory=sample_output,
            public_root=public_root,
            width=width,
            height=height,
            num_tokens=num_tokens,
            checkpoint_digest=checkpoint_sha(
                checkpoint_digests,
                experiment_root,
                initial_checkpoint,
            ),
        )
        identity_errors[sample_id] = identity_error
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

        stages: Dict[str, Any] = {"initial": initial_assets}
        if token in website_tokens:
            final_checkpoint = assert_safe_path(
                sample_directory / "checkpoints" / "final.pt",
                safe_root=args.safe_root,
                must_exist=True,
            )
            model, _ = load_model(
                final_checkpoint,
                pretrained=args.pretrained,
                device=device,
            )
            final_assets, _ = export_stage(
                sample_directory=sample_directory,
                stage="final",
                model=model,
                image=image.to(device),
                gt_points=gt_points,
                colors=colors,
                steps=steps,
                output_directory=sample_output,
                public_root=public_root,
                width=width,
                height=height,
                num_tokens=num_tokens,
                checkpoint_digest=checkpoint_sha(
                    checkpoint_digests,
                    experiment_root,
                    final_checkpoint,
                ),
            )
            stages["final"] = final_assets
            del model
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        record = {
            "id": sample_id,
            "label": f"样本 {token}",
            "order": int(sample_config["order"]),
            "description": str(entry.get("description", "")),
            "websiteEnabled": token in website_tokens,
            "cropXYXY": crop,
            "structureMask": {
                **dict(entry["display_mask"]),
                "pixels": int(structure.sum().item()),
            },
            "rgbUrl": "/" + rgb_output.relative_to(public_root).as_posix(),
            "stages": stages,
        }
        sample_records.append(record)
        for stage_name, stage_assets in stages.items():
            for step_key, asset in stage_assets.items():
                if "url" in asset:
                    checksum_records.append(
                        {
                            "sample": sample_id,
                            "stage": stage_name,
                            "k": int(step_key),
                            "url": asset["url"],
                            "bytes": (
                                public_root / str(asset["url"]).lstrip("/")
                            ).stat().st_size,
                            "sha256": asset["sha256"],
                        }
                    )

    website_ids = [
        discover_sample_directory(experiment_root, token)
        for token in website_tokens
    ]
    website_sample_ids = [
        json.loads((path / "config.json").read_text(encoding="utf-8"))["sample_id"]
        for path in website_ids
    ]
    manifest = {
        "version": 1,
        "experiment": config["experiment_id"],
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
        "resolution": {"width": width, "height": height},
        "steps": list(steps),
        "websiteSampleOrder": website_sample_ids,
        "archivedInitialSampleIds": [record["id"] for record in sample_records],
        "samples": sample_records,
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_path = output_root / "pointcloud_sha256.json"
    checksum_path.write_text(
        json.dumps(checksum_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "manifest": str(manifest_path),
        "sampleCount": len(sample_records),
        "websiteSampleCount": len(website_sample_ids),
        "uniquePointCloudCount": len(checksum_records),
        "pointCloudBytes": sum(int(item["bytes"]) for item in checksum_records),
        "initialIdentityMaxAbsError": identity_errors,
        "checksums": str(checksum_path),
    }
    (experiment_root / "pointcloud_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
