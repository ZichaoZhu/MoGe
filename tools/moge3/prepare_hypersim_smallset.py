from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict

import h5py

from moge.utils.remote_guard import (
    DEFAULT_SAFE_ROOT,
    assert_read_path,
    assert_safe_path,
)


AUTHORIZED_SOURCE = Path("/nas1/datasets/hypersim/raw")


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checked_file(path: Path, source_root: Path) -> Path:
    if path.is_symlink():
        raise PermissionError(f"Refusing symlinked dataset input: {path}")
    resolved = assert_read_path(
        path,
        allowed_read_roots=(source_root,),
        must_exist=True,
    )
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return resolved


def copy_checked(source: Path, destination: Path) -> Dict[str, object]:
    if destination.is_symlink():
        raise PermissionError(f"Refusing symlinked dataset output: {destination}")
    shutil.copy2(source, destination)
    source_hash = sha256(source)
    destination_hash = sha256(destination)
    if source_hash != destination_hash:
        raise IOError(f"Checksum mismatch after copying {source}")
    return {
        "source": str(source),
        "file": destination.name,
        "bytes": destination.stat().st_size,
        "sha256": destination_hash,
    }


def load_camera_rows(metadata_path: Path) -> Dict[str, Dict[str, str]]:
    with metadata_path.open(newline="", encoding="utf-8") as file:
        return {row["scene_name"]: row for row in csv.DictReader(file)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare a fixed multi-scene Hypersim subset under the safe project root"
    )
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=AUTHORIZED_SOURCE)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_root = assert_read_path(
        args.source_root,
        allowed_read_roots=(args.source_root,),
        must_exist=True,
    )
    if source_root != AUTHORIZED_SOURCE:
        raise PermissionError(f"Source is restricted to {AUTHORIZED_SOURCE}")
    spec_path = assert_safe_path(
        args.spec,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)

    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("format") != "moge3-hypersim-smallset-spec-v1":
        raise ValueError(f"Unsupported spec format: {spec.get('format')}")
    if set(spec["splits"]) != {"train", "val"}:
        raise ValueError("The spec must contain exactly train and val splits")

    metadata_path = checked_file(
        source_root
        / "ml-hypersim"
        / "contrib"
        / "mikeroberts3000"
        / "metadata_camera_parameters.csv",
        source_root,
    )
    camera_rows = load_camera_rows(metadata_path)

    samples = []
    sample_ids = set()
    for split in ("train", "val"):
        for group in spec["splits"][split]:
            scene = group["scene"]
            camera = group.get("camera", "cam_00")
            if scene not in camera_rows:
                raise KeyError(f"Missing camera metadata for {scene}")
            row = camera_rows[scene]
            height = int(float(row["settings_output_img_height"]))
            width = int(float(row["settings_output_img_width"]))
            matrix = [
                [float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
                for i in range(3)
            ]
            for frame in group["frames"]:
                sample_id = f"{scene}_{camera}_frame.{int(frame):04d}"
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample: {sample_id}")
                sample_ids.add(sample_id)
                rgb_source = checked_file(
                    source_root
                    / scene
                    / "images"
                    / f"scene_{camera}_final_preview"
                    / f"frame.{int(frame):04d}.tonemap.jpg",
                    source_root,
                )
                depth_source = checked_file(
                    source_root
                    / scene
                    / "images"
                    / f"scene_{camera}_geometry_hdf5"
                    / f"frame.{int(frame):04d}.depth_meters.hdf5",
                    source_root,
                )
                with h5py.File(depth_source, "r") as file:
                    if tuple(file["dataset"].shape) != (height, width):
                        raise ValueError(
                            f"Unexpected depth shape in {depth_source}: "
                            f"{tuple(file['dataset'].shape)}"
                        )
                rgb_destination = output / f"{sample_id}.tonemap.jpg"
                depth_destination = output / f"{sample_id}.depth_meters.hdf5"
                samples.append(
                    {
                        "id": sample_id,
                        "split": split,
                        "scene": scene,
                        "camera": camera,
                        "frame": int(frame),
                        "height": height,
                        "width": width,
                        "M_cam_from_uv": matrix,
                        "rgb": copy_checked(rgb_source, rgb_destination),
                        "depth": copy_checked(depth_source, depth_destination),
                    }
                )

    manifest = {
        "format": "moge3-hypersim-smallset-v1",
        "source_root": str(source_root),
        "source_access": "read-only",
        "depth_semantics": "Euclidean distance in meters from camera optical center",
        "coordinate_conversion": "[x,y,z]_hypersim -> [x,-y,-z]_moge",
        "counts": {
            split: sum(sample["split"] == split for sample in samples)
            for split in ("train", "val")
        },
        "samples": samples,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "counts": manifest["counts"],
                "total_bytes": sum(
                    int(sample[key]["bytes"])
                    for sample in samples
                    for key in ("rgb", "depth")
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
