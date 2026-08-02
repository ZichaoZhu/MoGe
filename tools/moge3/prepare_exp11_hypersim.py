from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_read_path, assert_safe_path


MANIFEST_FORMAT = "moge3-hypersim-generalization-v1"
SPLITS = ("train", "val", "test")
AUTHORIZED_SOURCE = Path("/nas1/datasets/hypersim/raw")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_group(scene: str) -> str:
    fields = scene.split("_")
    if len(fields) != 3 or fields[0] != "ai":
        raise ValueError(f"Unexpected Hypersim scene name: {scene}")
    return "_".join(fields[:2])


def approved_candidates(
    candidate_report: dict[str, Any],
    approval: dict[str, Any],
    *,
    candidate_sha256: str,
) -> list[dict[str, Any]]:
    if approval.get("status") != "approved":
        raise ValueError("Candidate review has not been approved")
    if approval.get("candidate_sha256") != candidate_sha256:
        raise ValueError("Approval does not match the reviewed candidate manifest")
    candidates = list(candidate_report.get("candidates", []))
    if len(candidates) != 52 or approval.get("candidate_count") != 52:
        raise ValueError("Exp11 requires exactly 52 approved candidates")
    identifiers = [str(candidate["id"]) for candidate in candidates]
    if len(set(identifiers)) != 52:
        raise ValueError("Approved candidates contain duplicate IDs")
    return candidates


def assemble_descriptors(
    exp10_manifest: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, list[str]]]:
    if exp10_manifest.get("format") != MANIFEST_FORMAT:
        raise ValueError("Exp10 manifest has an unsupported format")
    if exp10_manifest.get("counts") != {"train": 48, "val": 16, "test": 16}:
        raise ValueError("Exp10 split counts are not 48/16/16")

    descriptors: list[dict[str, Any]] = []
    for sample in exp10_manifest["samples"]:
        descriptors.append(
            {
                "id": str(sample["id"]),
                "split": str(sample["split"]),
                "scene": str(sample["scene"]),
                "camera": str(sample["camera"]),
                "frame": int(sample["frame"]),
                "training_source": str(
                    sample.get("training_source", "exp10_existing")
                ),
            }
        )
    for candidate in candidates:
        scene = str(candidate["scene"])
        camera = str(candidate["camera"])
        frame = int(candidate["frame"])
        expected_id = f"{scene}_{camera}_frame.{frame:04d}"
        if candidate["id"] != expected_id:
            raise ValueError(f"Candidate ID does not match its fields: {candidate['id']}")
        descriptors.append(
            {
                "id": expected_id,
                "split": "train",
                "scene": scene,
                "camera": camera,
                "frame": frame,
                "training_source": "exp11_reviewed_fine_structure_52",
                "candidate_display_id": str(candidate["display_id"]),
                "candidate_source_rank": int(candidate["source_rank"]),
                "candidate_crop_xyxy": [
                    int(candidate[f"crop_{axis}"])
                    for axis in ("x0", "y0", "x1", "y1")
                ],
            }
        )

    identifiers = [descriptor["id"] for descriptor in descriptors]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Exp11 contains duplicate sample IDs")
    counts = {
        split: sum(descriptor["split"] == split for descriptor in descriptors)
        for split in SPLITS
    }
    if counts != {"train": 100, "val": 16, "test": 16}:
        raise ValueError(f"Unexpected Exp11 counts: {counts}")

    group_owners: dict[str, str] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for descriptor in descriptors:
        group = scene_group(descriptor["scene"])
        split = descriptor["split"]
        previous = group_owners.setdefault(group, split)
        if previous != split:
            raise ValueError(f"Scene group {group} leaks across {previous} and {split}")
        if group not in groups[split]:
            groups[split].append(group)
    return descriptors, {split: groups[split] for split in SPLITS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare the approved 100-frame Exp11 Hypersim training set"
    )
    parser.add_argument("--exp10-manifest", type=Path, required=True)
    parser.add_argument("--candidate-report", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=AUTHORIZED_SOURCE)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    return parser.parse_args()


def main() -> None:
    import h5py
    from PIL import Image

    from tools.moge3.prepare_hypersim_smallset import (
        checked_file,
        copy_checked,
        load_camera_rows,
    )

    args = parse_args()
    source_root = assert_read_path(
        args.source_root,
        allowed_read_roots=(args.source_root,),
        must_exist=True,
    )
    if source_root != AUTHORIZED_SOURCE:
        raise PermissionError(f"Source is restricted to {AUTHORIZED_SOURCE}")
    exp10_path = assert_safe_path(
        args.exp10_manifest, safe_root=args.safe_root, must_exist=True
    )
    candidate_path = assert_safe_path(
        args.candidate_report, safe_root=args.safe_root, must_exist=True
    )
    approval_path = assert_safe_path(
        args.approval, safe_root=args.safe_root, must_exist=True
    )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    if output.is_symlink():
        raise PermissionError(f"Refusing symlinked output: {output}")

    exp10_manifest = json.loads(exp10_path.read_text(encoding="utf-8"))
    candidate_report = json.loads(candidate_path.read_text(encoding="utf-8"))
    approval = json.loads(approval_path.read_text(encoding="utf-8"))
    candidates = approved_candidates(
        candidate_report,
        approval,
        candidate_sha256=sha256(candidate_path),
    )
    descriptors, groups = assemble_descriptors(exp10_manifest, candidates)
    candidate_by_id = {str(candidate["id"]): candidate for candidate in candidates}

    metadata_path = checked_file(
        source_root
        / "ml-hypersim"
        / "contrib"
        / "mikeroberts3000"
        / "metadata_camera_parameters.csv",
        source_root,
    )
    camera_rows = load_camera_rows(metadata_path)
    samples: list[dict[str, Any]] = []
    for descriptor in descriptors:
        scene = descriptor["scene"]
        camera = descriptor["camera"]
        frame = descriptor["frame"]
        if scene not in camera_rows:
            raise KeyError(f"Missing camera metadata for {scene}")
        row = camera_rows[scene]
        height = int(float(row["settings_output_img_height"]))
        width = int(float(row["settings_output_img_width"]))
        matrix = [
            [float(row[f"M_cam_from_uv_{i}{j}"]) for j in range(3)]
            for i in range(3)
        ]
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
        if descriptor["id"] in candidate_by_id:
            candidate = candidate_by_id[descriptor["id"]]
            if Path(candidate["rgb_source"]) != rgb_source:
                raise ValueError(f"Reviewed RGB source changed for {descriptor['id']}")
            if Path(candidate["depth_source"]) != depth_source:
                raise ValueError(f"Reviewed depth source changed for {descriptor['id']}")
        with Image.open(rgb_source) as image:
            if image.size != (width, height):
                raise ValueError(
                    f"Unexpected RGB shape in {rgb_source}: {image.size}"
                )
        with h5py.File(depth_source, "r") as handle:
            if tuple(handle["dataset"].shape) != (height, width):
                raise ValueError(
                    f"Unexpected depth shape in {depth_source}: "
                    f"{tuple(handle['dataset'].shape)}"
                )

        rgb_destination = output / f"{descriptor['id']}.tonemap.jpg"
        depth_destination = output / f"{descriptor['id']}.depth_meters.hdf5"
        samples.append(
            {
                **descriptor,
                "scene_group": scene_group(scene),
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
        "source_access": "read-only source copied into Exp11 data",
        "depth_semantics": "Euclidean distance in meters from camera optical center",
        "coordinate_conversion": "[x,y,z]_hypersim -> [x,-y,-z]_moge",
        "split_policy": (
            "train extends Exp10's 48 frames with 52 user-reviewed distinct "
            "Hypersim scenes; validation and test preserve Exp10 held-out groups"
        ),
        "provenance": {
            "exp10_training_frames": 48,
            "exp11_reviewed_training_frames": 52,
            "validation_frames": 16,
            "test_frames": 16,
            "candidate_manifest_sha256": sha256(candidate_path),
            "approval_manifest_sha256": sha256(approval_path),
        },
        "scene_groups": groups,
        "counts": {
            split: sum(sample["split"] == split for sample in samples)
            for split in SPLITS
        },
        "samples": samples,
    }
    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksums = [
        f"{sample[key]['sha256']}  {sample[key]['file']}"
        for sample in samples
        for key in ("rgb", "depth")
    ]
    (output / "assets.sha256").write_text(
        "\n".join(checksums) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "manifest": str(manifest_path),
                "counts": manifest["counts"],
                "scene_group_counts": {
                    split: len(groups[split]) for split in SPLITS
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
