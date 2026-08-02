from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


HYPERSIM_SOURCE_ROOT = Path("/nas1/datasets/hypersim/raw")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select and render 52 new Hypersim training candidates for Exp11 "
            "without reading model predictions."
        )
    )
    parser.add_argument("--ranking", type=Path, required=True)
    parser.add_argument("--existing-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--count", type=int, default=52)
    parser.add_argument("--items-per-sheet", type=int, default=13)
    parser.add_argument("--max-per-scene-group", type=int, default=2)
    parser.add_argument(
        "--curation",
        type=Path,
        help="Optional reviewed replacement mapping under the safe root.",
    )
    return parser.parse_args()


def scene_group(scene: str) -> str:
    components = scene.split("_")
    if len(components) < 2 or components[0] != "ai":
        raise ValueError(f"Unexpected Hypersim scene name: {scene}")
    return "_".join(components[:2])


def read_ranking(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Ranking is empty")
    required = {
        "id",
        "scene",
        "camera",
        "frame",
        "crop_x0",
        "crop_y0",
        "crop_x1",
        "crop_y1",
        "crop_thin_pixels",
        "crop_thin_near_pixels",
        "crop_thin_far_pixels",
        "rgb_source",
        "depth_source",
    }
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"Ranking is missing columns: {sorted(missing)}")
    return rows


def validate_read_only_source(path_value: str) -> Path:
    source_root = HYPERSIM_SOURCE_ROOT.resolve(strict=True)
    path = Path(path_value).resolve(strict=True)
    if source_root not in path.parents:
        raise ValueError(f"Source escapes the approved Hypersim root: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def eligible_candidates(
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    existing_ids = {str(sample["id"]) for sample in manifest["samples"]}
    existing_scenes = {str(sample["scene"]) for sample in manifest["samples"]}
    held_out_groups = {
        str(group)
        for split in ("val", "test")
        for group in manifest["scene_groups"][split]
    }

    eligible: list[dict[str, Any]] = []
    for source_rank, row in enumerate(rows, start=1):
        group = scene_group(row["scene"])
        if (
            row["id"] in existing_ids
            or row["scene"] in existing_scenes
            or group in held_out_groups
        ):
            continue
        eligible.append(
            {
                **row,
                "source_rank": source_rank,
                "scene_group": group,
            }
        )

    exclusions = {
        "existing_sample_ids": len(existing_ids),
        "existing_scenes": sorted(existing_scenes),
        "held_out_scene_groups": sorted(held_out_groups),
        "eligible_candidates": len(eligible),
    }
    return eligible, exclusions


def select_candidates(
    rows: list[dict[str, str]],
    manifest: dict[str, Any],
    *,
    count: int,
    max_per_scene_group: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    eligible, exclusions = eligible_candidates(rows, manifest)
    selected: list[dict[str, Any]] = []
    selected_scenes: set[str] = set()
    group_counts: Counter[str] = Counter()

    # First maximize scene-group diversity, then fill remaining slots while
    # retaining at most max_per_scene_group distinct scenes per group.
    for group_limit in range(1, max_per_scene_group + 1):
        for candidate in eligible:
            if len(selected) == count:
                break
            scene = str(candidate["scene"])
            group = str(candidate["scene_group"])
            if scene in selected_scenes or group_counts[group] >= group_limit:
                continue
            selected.append(candidate)
            selected_scenes.add(scene)
            group_counts[group] += 1
        if len(selected) == count:
            break

    if len(selected) != count:
        raise RuntimeError(
            f"Could select only {len(selected)} of {count} requested candidates"
        )

    exclusions["maximum_per_scene_group"] = max_per_scene_group
    return selected, exclusions


def apply_curation(
    selected: list[dict[str, Any]],
    eligible: list[dict[str, Any]],
    curation: dict[str, Any],
    *,
    max_per_scene_group: int,
) -> list[dict[str, Any]]:
    replacements = list(curation.get("replacements", []))
    if not replacements:
        raise ValueError("Curation must contain at least one replacement")
    eligible_by_id = {str(candidate["id"]): candidate for candidate in eligible}
    selected_ids = {str(candidate["id"]) for candidate in selected}
    result = list(selected)
    replaced_from: set[str] = set()
    replaced_to: set[str] = set()
    for replacement in replacements:
        source_id = str(replacement["remove"])
        destination_id = str(replacement["add"])
        if source_id in replaced_from or destination_id in replaced_to:
            raise ValueError("Curation replacement ids must be unique")
        if source_id not in selected_ids:
            raise ValueError(f"Curated removal is not selected: {source_id}")
        if destination_id not in eligible_by_id:
            raise ValueError(f"Curated addition is not eligible: {destination_id}")
        if destination_id in selected_ids:
            raise ValueError(f"Curated addition is already selected: {destination_id}")
        index = next(
            index
            for index, candidate in enumerate(result)
            if candidate["id"] == source_id
        )
        result[index] = {
            **eligible_by_id[destination_id],
            "curation_replaced": source_id,
            "curation_reason": str(replacement["reason"]),
        }
        selected_ids.remove(source_id)
        selected_ids.add(destination_id)
        replaced_from.add(source_id)
        replaced_to.add(destination_id)

    scenes = [str(candidate["scene"]) for candidate in result]
    if len(scenes) != len(set(scenes)):
        raise ValueError("Curated selection contains duplicate scenes")
    group_counts = Counter(str(candidate["scene_group"]) for candidate in result)
    if max(group_counts.values()) > max_per_scene_group:
        raise ValueError("Curated selection exceeds the scene-group limit")
    return result


def review_tile(
    candidate: dict[str, Any],
    *,
    display_id: str,
    width: int = 600,
    height: int = 360,
) -> Image.Image:
    rgb_path = validate_read_only_source(str(candidate["rgb_source"]))
    with Image.open(rgb_path) as loaded:
        image = loaded.convert("RGB")

    label_height = 54
    full_width, full_height = 400, 300
    inset_size = 190
    full = image.resize((full_width, full_height), Image.Resampling.LANCZOS)
    full_draw = ImageDraw.Draw(full)
    x0 = int(candidate["crop_x0"])
    y0 = int(candidate["crop_y0"])
    x1 = int(candidate["crop_x1"])
    y1 = int(candidate["crop_y1"])
    source_shape = (256, 192)
    full_draw.rectangle(
        (
            round(x0 * full_width / source_shape[0]),
            round(y0 * full_height / source_shape[1]),
            round(x1 * full_width / source_shape[0]),
            round(y1 * full_height / source_shape[1]),
        ),
        outline=(255, 45, 45),
        width=4,
    )

    crop = image.crop(
        (
            round(x0 * image.width / source_shape[0]),
            round(y0 * image.height / source_shape[1]),
            round(x1 * image.width / source_shape[0]),
            round(y1 * image.height / source_shape[1]),
        )
    ).resize((inset_size, inset_size), Image.Resampling.LANCZOS)

    tile = Image.new("RGB", (width, height), "white")
    tile.paste(full, (0, label_height))
    tile.paste(crop, (405, label_height + 45))
    draw = ImageDraw.Draw(tile)
    draw.rectangle(
        (404, label_height + 44, 405 + inset_size, label_height + 45 + inset_size),
        outline=(255, 45, 45),
        width=3,
    )
    draw.text(
        (8, 6),
        f"{display_id} | {candidate['id']} | source rank {candidate['source_rank']}",
        fill="black",
    )
    draw.text(
        (8, 28),
        (
            f"group={candidate['scene_group']} | "
            f"thin={candidate['crop_thin_pixels']} | red box enlarged at right"
        ),
        fill="black",
    )
    return tile


def save_contact_sheets(
    selected: list[dict[str, Any]],
    output: Path,
    *,
    items_per_sheet: int,
) -> list[str]:
    columns = 2
    tile_width, tile_height = 600, 360
    artifacts: list[str] = []
    for page_start in range(0, len(selected), items_per_sheet):
        page = selected[page_start : page_start + items_per_sheet]
        rows = (len(page) + columns - 1) // columns
        canvas = Image.new(
            "RGB",
            (columns * tile_width, rows * tile_height),
            (235, 235, 235),
        )
        for local_index, candidate in enumerate(page):
            global_index = page_start + local_index + 1
            display_id = f"N{global_index:02d}"
            candidate["display_id"] = display_id
            tile = review_tile(candidate, display_id=display_id)
            x = (local_index % columns) * tile_width
            y = (local_index // columns) * tile_height
            canvas.paste(tile, (x, y))
        page_index = page_start // items_per_sheet + 1
        filename = f"candidate_52_contact_sheet_{page_index:02d}.jpg"
        canvas.save(output / filename, quality=92, optimize=True)
        artifacts.append(filename)
    return artifacts


def main() -> None:
    args = parse_args()
    if args.count <= 0 or args.items_per_sheet <= 0:
        raise ValueError("Candidate and sheet counts must be positive")
    if args.max_per_scene_group <= 0:
        raise ValueError("--max-per-scene-group must be positive")

    ranking_path = assert_safe_path(
        args.ranking,
        safe_root=args.safe_root,
        must_exist=True,
    )
    manifest_path = assert_safe_path(
        args.existing_manifest,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(
        args.output,
        safe_root=args.safe_root,
        writable=True,
    )
    output.mkdir(parents=True, exist_ok=True)

    rows = read_ranking(ranking_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected, exclusions = select_candidates(
        rows,
        manifest,
        count=args.count,
        max_per_scene_group=args.max_per_scene_group,
    )
    curation = None
    if args.curation is not None:
        curation_path = assert_safe_path(
            args.curation,
            safe_root=args.safe_root,
            must_exist=True,
        )
        curation = json.loads(curation_path.read_text(encoding="utf-8"))
        eligible, _ = eligible_candidates(rows, manifest)
        selected = apply_curation(
            selected,
            eligible,
            curation,
            max_per_scene_group=args.max_per_scene_group,
        )
    sheets = save_contact_sheets(
        selected,
        output,
        items_per_sheet=args.items_per_sheet,
    )
    report = {
        "status": "awaiting-user-review",
        "selection_inputs": (
            "Hypersim ground-truth fine-structure ranking and RGB only; "
            "no model predictions or held-out metrics"
        ),
        "selection_rule": (
            "exclude every existing Exp10 scene and all validation/test scene "
            "groups; rank by the existing GT-only fine-structure scan; maximize "
            "scene-group diversity; use distinct scenes"
        ),
        "requested_total_training_images": 100,
        "existing_training_images": 48,
        "new_candidate_images": len(selected),
        "source_ranking": str(ranking_path),
        "existing_manifest": str(manifest_path),
        "exclusions": exclusions,
        "manual_curation": curation,
        "candidates": selected,
        "artifacts": {"contact_sheets": sheets},
    }
    (output / "candidate_52.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
