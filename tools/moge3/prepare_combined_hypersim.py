from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Dict, Iterable, List

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


MANIFEST_FORMAT = "moge3-hypersim-generalization-v1"
SPLITS = ("train", "val", "test")


def _sample_scene_group(sample: Dict[str, object]) -> str:
    scene_group = sample.get("scene_group")
    if scene_group is not None:
        return str(scene_group)
    scene = str(sample["scene"])
    fields = scene.split("_")
    if len(fields) < 2:
        raise ValueError(f"Cannot derive scene group from scene: {scene}")
    return "_".join(fields[:2])


def _scene_groups(samples: Iterable[Dict[str, object]], split: str) -> List[str]:
    return list(
        dict.fromkeys(
            _sample_scene_group(sample)
            for sample in samples
            if sample["split"] == split
        )
    )


def merge_manifests(
    exp2: Dict[str, object],
    exp7: Dict[str, object],
) -> Dict[str, object]:
    if exp2.get("format") not in {
        "moge3-hypersim-smallset-v1",
        MANIFEST_FORMAT,
    }:
        raise ValueError(f"Unsupported exp2 manifest: {exp2.get('format')}")
    if exp7.get("format") != MANIFEST_FORMAT:
        raise ValueError(f"Unsupported exp7 manifest: {exp7.get('format')}")

    exp2_train = [
        copy.deepcopy(sample)
        for sample in exp2["samples"]
        if sample["split"] == "train"
    ]
    exp7_train = [
        copy.deepcopy(sample)
        for sample in exp7["samples"]
        if sample["split"] == "train"
    ]
    held_out = [
        copy.deepcopy(sample)
        for sample in exp7["samples"]
        if sample["split"] in {"val", "test"}
    ]
    samples = exp2_train + exp7_train + held_out

    for sample in samples:
        sample["scene_group"] = _sample_scene_group(sample)

    identifiers = [str(sample["id"]) for sample in samples]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Combined manifest contains duplicate sample IDs")
    counts = {
        split: sum(sample["split"] == split for sample in samples)
        for split in SPLITS
    }
    if counts != {"train": 48, "val": 16, "test": 16}:
        raise ValueError(f"Unexpected combined counts: {counts}")

    owners: Dict[str, str] = {}
    for sample in samples:
        group = _sample_scene_group(sample)
        split = str(sample["split"])
        previous = owners.setdefault(group, split)
        if previous != split:
            raise ValueError(
                f"Scene group {group} leaks across {previous} and {split}"
            )

    for sample in exp2_train:
        sample["training_source"] = "exp2_trajectory_24"
    for sample in exp7_train:
        sample["training_source"] = "exp7_curated_24"
    for sample in held_out:
        sample["training_source"] = "exp7_held_out"

    return {
        "format": MANIFEST_FORMAT,
        "source_root": "combined prepared Hypersim assets from exp2 and exp7",
        "source_access": "read-only inputs copied into exp10 data",
        "depth_semantics": "Euclidean distance in meters from camera optical center",
        "coordinate_conversion": "[x,y,z]_hypersim -> [x,-y,-z]_moge",
        "split_policy": (
            "training is exp2 train 24 plus exp7 train 24; validation and "
            "test reuse exp7 held-out scene groups"
        ),
        "provenance": {
            "exp2_train": 24,
            "exp7_train": 24,
            "exp7_validation": 16,
            "exp7_test": 16,
            "tartanair_used": False,
        },
        "scene_groups": {
            split: _scene_groups(samples, split) for split in SPLITS
        },
        "counts": counts,
        "samples": samples,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _copy_asset(
    source_data: Path,
    destination_data: Path,
    asset: Dict[str, object],
    *,
    safe_root: Path,
) -> None:
    source = assert_safe_path(
        source_data / str(asset["file"]),
        safe_root=safe_root,
        must_exist=True,
    )
    if source.is_symlink():
        raise PermissionError(f"Refusing symlinked input: {source}")
    destination = assert_safe_path(
        destination_data / str(asset["file"]),
        safe_root=safe_root,
        writable=True,
    )
    if destination.exists():
        if destination.is_symlink():
            raise PermissionError(f"Refusing symlinked output: {destination}")
    else:
        shutil.copy2(source, destination)
    expected = str(asset["sha256"])
    actual = _sha256(destination)
    if actual != expected:
        raise ValueError(f"Checksum mismatch for {destination}: {actual}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine the two approved 24-frame Hypersim training sets"
    )
    parser.add_argument("--exp2-data", type=Path, required=True)
    parser.add_argument("--exp7-data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exp2_data = assert_safe_path(
        args.exp2_data, safe_root=args.safe_root, must_exist=True
    )
    exp7_data = assert_safe_path(
        args.exp7_data, safe_root=args.safe_root, must_exist=True
    )
    output = assert_safe_path(
        args.output, safe_root=args.safe_root, writable=True
    )
    output.mkdir(parents=True, exist_ok=True)
    exp2 = json.loads((exp2_data / "manifest.json").read_text(encoding="utf-8"))
    exp7 = json.loads((exp7_data / "manifest.json").read_text(encoding="utf-8"))
    manifest = merge_manifests(exp2, exp7)

    exp2_ids = {
        str(sample["id"])
        for sample in exp2["samples"]
        if sample["split"] == "train"
    }
    for sample in manifest["samples"]:
        source_data = exp2_data if str(sample["id"]) in exp2_ids else exp7_data
        for key in ("rgb", "depth"):
            _copy_asset(
                source_data,
                output,
                sample[key],
                safe_root=args.safe_root,
            )

    manifest_path = output / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "counts": manifest["counts"],
                "scene_groups": {
                    split: len(manifest["scene_groups"][split])
                    for split in SPLITS
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
