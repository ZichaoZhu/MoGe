from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable

import h5py

from moge.utils.remote_guard import (
    DEFAULT_SAFE_ROOT,
    assert_read_path,
    assert_safe_path,
)


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_row(metadata_path: Path, scene: str) -> Dict[str, str]:
    with metadata_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["scene_name"] == scene:
                return row
    raise KeyError(f"Scene {scene!r} is absent from {metadata_path}")


def checked_source(path: Path, source_root: Path) -> Path:
    if path.is_symlink():
        raise PermissionError(f"Refusing symlinked dataset input: {path}")
    resolved = assert_read_path(
        path,
        allowed_read_roots=(source_root,),
        must_exist=True,
    )
    if not resolved.is_file():
        raise FileNotFoundError(f"Expected a regular file: {resolved}")
    return resolved


def copy_checked(source: Path, destination: Path) -> Dict[str, object]:
    shutil.copy2(source, destination)
    if sha256(source) != sha256(destination):
        raise IOError(f"Checksum mismatch after copying {source}")
    return {
        "source": str(source),
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": sha256(destination),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Copy an explicit, read-only Hypersim mini-batch into the safe project root"
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("/nas1/datasets/hypersim/raw"),
    )
    parser.add_argument("--scene", default="ai_001_001")
    parser.add_argument("--camera", default="cam_00")
    parser.add_argument("--frames", type=int, nargs="+", default=[0])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = assert_read_path(
        args.source_root,
        allowed_read_roots=(args.source_root,),
        must_exist=True,
    )
    if source_root != Path("/nas1/datasets/hypersim/raw"):
        raise PermissionError(
            "This preparation tool is intentionally restricted to "
            "/nas1/datasets/hypersim/raw"
        )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)

    metadata_path = checked_source(
        source_root
        / "ml-hypersim"
        / "contrib"
        / "mikeroberts3000"
        / "metadata_camera_parameters.csv",
        source_root,
    )
    row = camera_row(metadata_path, args.scene)
    height = int(float(row["settings_output_img_height"]))
    width = int(float(row["settings_output_img_width"]))
    matrix = [
        [float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
        for i in range(3)
    ]

    samples = []
    scene_root = source_root / args.scene
    for frame in args.frames:
        stem = f"frame.{frame:04d}"
        rgb_source = checked_source(
            scene_root
            / "images"
            / f"scene_{args.camera}_final_preview"
            / f"{stem}.tonemap.jpg",
            source_root,
        )
        depth_source = checked_source(
            scene_root
            / "images"
            / f"scene_{args.camera}_geometry_hdf5"
            / f"{stem}.depth_meters.hdf5",
            source_root,
        )
        with h5py.File(depth_source, "r") as file:
            if tuple(file["dataset"].shape) != (height, width):
                raise ValueError(
                    f"Unexpected depth shape in {depth_source}: "
                    f"{tuple(file['dataset'].shape)}"
                )

        rgb_destination = output / f"{args.scene}_{args.camera}_{stem}.tonemap.jpg"
        depth_destination = output / f"{args.scene}_{args.camera}_{stem}.depth_meters.hdf5"
        samples.append(
            {
                "frame": frame,
                "rgb": copy_checked(rgb_source, rgb_destination),
                "depth": copy_checked(depth_source, depth_destination),
            }
        )

    manifest = {
        "format": "moge3-hypersim-single-batch-v1",
        "source_root": str(source_root),
        "source_access": "read-only",
        "scene": args.scene,
        "camera": args.camera,
        "height": height,
        "width": width,
        "meters_per_asset_unit": float(row["settings_units_info_meters_scale"]),
        "M_cam_from_uv": matrix,
        "coordinate_conversion": "[x,y,z]_hypersim -> [x,-y,-z]_moge",
        "depth_semantics": "Euclidean distance in meters from camera optical center",
        "samples": samples,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"manifest": str(manifest_path), "samples": len(samples)}))


if __name__ == "__main__":
    main()
