from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import h5py

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_read_path, assert_safe_path
from tools.moge3.prepare_hypersim_smallset import (
    AUTHORIZED_SOURCE,
    checked_file,
    copy_checked,
    load_camera_rows,
)


SPLITS = ("train", "val", "test")
FORMAT = "moge3-hypersim-generalization-spec-v1"
MANIFEST_FORMAT = "moge3-hypersim-generalization-v1"


def scene_group(scene: str) -> str:
    fields = scene.split("_")
    if len(fields) != 3 or fields[0] != "ai":
        raise ValueError(f"Unexpected Hypersim scene name: {scene}")
    return "_".join(fields[:2])


def validate_disjoint_scene_groups(spec: Dict[str, object]) -> None:
    owner: Dict[str, str] = {}
    exact_scenes: Dict[str, str] = {}
    for split in SPLITS:
        for group in spec["splits"][split]:
            scene = str(group["scene"])
            base = scene_group(scene)
            if scene in exact_scenes:
                raise ValueError(
                    f"Scene {scene} occurs in both {exact_scenes[scene]} and {split}"
                )
            if base in owner and owner[base] != split:
                raise ValueError(
                    f"Scene group {base} leaks across {owner[base]} and {split}"
                )
            exact_scenes[scene] = split
            owner[base] = split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare disjoint Hypersim train/val/test splits under the server safe root"
        )
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
    if spec.get("format") != FORMAT:
        raise ValueError(f"Unsupported spec format: {spec.get('format')}")
    if tuple(spec.get("splits", {}).keys()) != SPLITS:
        raise ValueError(f"Spec splits must be ordered as {SPLITS}")
    validate_disjoint_scene_groups(spec)

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
    scene_groups: Dict[str, list[str]] = {split: [] for split in SPLITS}
    for split in SPLITS:
        for group in spec["splits"][split]:
            scene = str(group["scene"])
            camera = str(group.get("camera", "cam_00"))
            base = scene_group(scene)
            if base not in scene_groups[split]:
                scene_groups[split].append(base)
            if scene not in camera_rows:
                raise KeyError(f"Missing camera metadata for {scene}")
            row = camera_rows[scene]
            height = int(float(row["settings_output_img_height"]))
            width = int(float(row["settings_output_img_width"]))
            matrix = [
                [float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
                for i in range(3)
            ]
            frames = [int(frame) for frame in group["frames"]]
            if not frames or frames != sorted(set(frames)):
                raise ValueError(f"Frames for {scene} must be a non-empty sorted set")
            for frame in frames:
                sample_id = f"{scene}_{camera}_frame.{frame:04d}"
                if sample_id in sample_ids:
                    raise ValueError(f"Duplicate sample: {sample_id}")
                sample_ids.add(sample_id)
                rgb_source = checked_file(
                    source_root
                    / scene
                    / "images"
                    / f"scene_{camera}_final_preview"
                    / f"frame.{frame:04d}.tonemap.jpg",
                    source_root,
                )
                depth_source = checked_file(
                    source_root
                    / scene
                    / "images"
                    / f"scene_{camera}_geometry_hdf5"
                    / f"frame.{frame:04d}.depth_meters.hdf5",
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
                        "scene_group": scene_group(scene),
                        "camera": camera,
                        "frame": frame,
                        "height": height,
                        "width": width,
                        "M_cam_from_uv": matrix,
                        "rgb": copy_checked(rgb_source, rgb_destination),
                        "depth": copy_checked(depth_source, depth_destination),
                    }
                )

    manifest = {
        "format": MANIFEST_FORMAT,
        "source_root": str(source_root),
        "source_access": "read-only",
        "depth_semantics": "Euclidean distance in meters from camera optical center",
        "coordinate_conversion": "[x,y,z]_hypersim -> [x,-y,-z]_moge",
        "split_policy": (
            "train, validation, and test use disjoint ai_XXX source scene groups"
        ),
        "scene_groups": scene_groups,
        "counts": {
            split: sum(sample["split"] == split for sample in samples)
            for split in SPLITS
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
                "scene_counts": {
                    split: len(spec["splits"][split]) for split in SPLITS
                },
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
