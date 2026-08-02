from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the locked RGB/GT-only fine-structure ROI manifest for Exp13."
    )
    parser.add_argument("--data-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--held-out-selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    return parser.parse_args()


def scaled_crop(
    entry: dict[str, Any],
    *,
    source_width: int,
    source_height: int,
    width: int,
    height: int,
) -> list[int]:
    values = [
        round(int(entry["crop_x0"]) * width / source_width),
        round(int(entry["crop_y0"]) * height / source_height),
        round(int(entry["crop_x1"]) * width / source_width),
        round(int(entry["crop_y1"]) * height / source_height),
    ]
    x0, y0, x1, y1 = values
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError(f"Scaled crop is invalid: {values}")
    return values


def build_manifest(
    *,
    data_manifest: dict[str, Any],
    candidates: dict[str, Any],
    approval: dict[str, Any],
    held_out: dict[str, Any],
    width: int,
    height: int,
) -> dict[str, Any]:
    if approval.get("status") != "approved":
        raise ValueError("Exp11 candidates have not been approved")
    selected = list(candidates.get("candidates", []))
    if len(selected) != int(approval.get("candidate_count", -1)):
        raise ValueError("Approved candidate count does not match candidate manifest")
    samples = {str(sample["id"]): sample for sample in data_manifest["samples"]}
    entries: list[dict[str, Any]] = []
    seen: set[str] = set()

    for candidate in selected:
        sample_id = str(candidate["id"])
        metadata = samples.get(sample_id)
        if metadata is None or metadata["split"] != "train":
            raise ValueError(f"Candidate is absent from the training split: {sample_id}")
        crop = scaled_crop(
            candidate,
            source_width=256,
            source_height=192,
            width=width,
            height=height,
        )
        entries.append(
            {
                "id": sample_id,
                "split": "train",
                "crop_xyxy": crop,
                "structure_mask": {
                    "type": "near_quantile",
                    "quantile": 0.5,
                },
                "source": "exp11_approved_gt_fine_structure_candidate",
            }
        )
        seen.add(sample_id)

    for split in ("train", "val", "test"):
        selected_entry = held_out[split]
        sample_id = str(selected_entry["id"])
        metadata = samples.get(sample_id)
        if metadata is None or metadata["split"] != split:
            raise ValueError(f"Locked {split} sample is absent from its split")
        if sample_id in seen:
            continue
        crop = [
            int(selected_entry["crop_x0"]),
            int(selected_entry["crop_y0"]),
            int(selected_entry["crop_x1"]),
            int(selected_entry["crop_y1"]),
        ]
        entries.append(
            {
                "id": sample_id,
                "split": split,
                "crop_xyxy": crop,
                "structure_mask": {
                    "type": "near_quantile",
                    "quantile": 0.5,
                },
                "source": "exp12_locked_rgb_gt_selection",
            }
        )
        seen.add(sample_id)

    counts = {
        split: sum(entry["split"] == split for entry in entries)
        for split in ("train", "val", "test")
    }
    return {
        "format": "moge3-fine-structure-rois-v1",
        "selection_inputs": "RGB and ground truth only; no predictions",
        "selection_rule": (
            "Reuse the 52 user-approved Exp11 GT-ranked crops and the three "
            "Exp12 crops locked before prediction rendering."
        ),
        "shape": [height, width],
        "counts": counts,
        "entries": entries,
    }


def main() -> None:
    args = parse_args()
    if min(args.height, args.width) <= 0:
        raise ValueError("Image dimensions must be positive")
    data_manifest_path = assert_safe_path(
        args.data_manifest,
        safe_root=args.safe_root,
        must_exist=True,
    )
    candidates_path = assert_safe_path(
        args.candidates,
        safe_root=args.safe_root,
        must_exist=True,
    )
    approval_path = assert_safe_path(
        args.approval,
        safe_root=args.safe_root,
        must_exist=True,
    )
    held_out_path = assert_safe_path(
        args.held_out_selection,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(
        args.output,
        safe_root=args.safe_root,
        writable=True,
    )
    manifest = build_manifest(
        data_manifest=json.loads(data_manifest_path.read_text(encoding="utf-8")),
        candidates=json.loads(candidates_path.read_text(encoding="utf-8")),
        approval=json.loads(approval_path.read_text(encoding="utf-8")),
        held_out=json.loads(held_out_path.read_text(encoding="utf-8")),
        width=args.width,
        height=args.height,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output)
    print(json.dumps({"output": str(output), **manifest["counts"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
