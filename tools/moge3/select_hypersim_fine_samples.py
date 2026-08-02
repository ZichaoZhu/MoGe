from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
from PIL import Image, ImageDraw

from moge.scripts.train_hypersim_smallset_v3 import load_raw_sample
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.visualize_fine_structure import (
    paper_coarse_fine_mask,
    select_crop,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select fixed Hypersim fine-structure visualization samples using "
            "ground-truth geometry only"
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--crop-height", type=int, default=192)
    parser.add_argument("--crop-width", type=int, default=192)
    parser.add_argument("--crop-stride", type=int, default=32)
    parser.add_argument("--minimum-fine-pixels", type=int, default=96)
    parser.add_argument("--contact-sheet-items", type=int, default=6)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["val", "test"],
    )
    parser.add_argument("--train-sample-id")
    parser.add_argument("--val-sample-id")
    parser.add_argument("--test-sample-id")
    parser.add_argument(
        "--selection-status",
        choices=("locked-before-training", "locked-before-rendering"),
        default="locked-before-training",
    )
    return parser.parse_args()


def save_csv(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    rows = list(rows)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def contact_sheet(
    output: Path,
    split: str,
    rows: List[Dict[str, object]],
    images: Dict[str, np.ndarray],
    *,
    maximum_items: int,
) -> None:
    selected = rows[:maximum_items]
    columns = 3
    tile_width, tile_height, label_height = 320, 240, 34
    row_count = (len(selected) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_width, row_count * (tile_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(selected):
        image = images[str(row["id"])]
        if image.dtype == np.uint8:
            panel_array = image
        else:
            panel_array = np.round(image.clip(0, 1) * 255).astype(np.uint8)
        panel = Image.fromarray(panel_array).resize(
            (tile_width, tile_height),
            Image.Resampling.LANCZOS,
        )
        panel_draw = ImageDraw.Draw(panel)
        scale_x = tile_width / image.shape[1]
        scale_y = tile_height / image.shape[0]
        panel_draw.rectangle(
            (
                int(row["crop_x0"] * scale_x),
                int(row["crop_y0"] * scale_y),
                int(row["crop_x1"] * scale_x),
                int(row["crop_y1"] * scale_y),
            ),
            outline="red",
            width=3,
        )
        column = index % columns
        grid_row = index // columns
        x = column * tile_width
        y = grid_row * (tile_height + label_height)
        canvas.paste(panel, (x, y + label_height))
        draw.text(
            (x + 5, y + 5),
            f"{row['id']} | fine={row['fine_pixels']}",
            fill="black",
        )
    canvas.save(output / f"{split}_fine_sample_candidates.png")


def main() -> None:
    args = parse_args()
    if args.contact_sheet_items <= 0:
        raise ValueError("--contact-sheet-items must be positive")
    data_dir = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = assert_safe_path(
        data_dir / "manifest.json",
        safe_root=args.safe_root,
        must_exist=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows: List[Dict[str, object]] = []
    images: Dict[str, np.ndarray] = {}
    for sample in manifest["samples"]:
        split = str(sample["split"])
        if split not in args.splits:
            continue
        image, gt_points = load_raw_sample(
            data_dir,
            sample,
            args.height,
            args.width,
            args.safe_root,
        )
        gt_numpy = gt_points.numpy()
        valid = np.isfinite(gt_numpy).all(axis=-1)
        fine_mask = paper_coarse_fine_mask(gt_numpy[..., 2], valid)
        crop, score, count = select_crop(
            fine_mask,
            np.zeros_like(fine_mask, dtype=np.float32),
            np.zeros_like(fine_mask, dtype=np.float32),
            crop_height=args.crop_height,
            crop_width=args.crop_width,
            stride=args.crop_stride,
            min_fine_pixels=args.minimum_fine_pixels,
            mode="structure",
        )
        x0, y0, x1, y1 = crop
        rows.append(
            {
                "id": str(sample["id"]),
                "split": split,
                "scene": str(sample["scene"]),
                "frame": int(sample["frame"]),
                "fine_pixels": count,
                "fine_fraction": count / float((y1 - y0) * (x1 - x0)),
                "selection_score": score,
                "crop_x0": x0,
                "crop_y0": y0,
                "crop_x1": x1,
                "crop_y1": y1,
            }
        )
        images[str(sample["id"])] = (
            np.round(
                image.permute(1, 2, 0).numpy().clip(0, 1) * 255
            )
            .astype(np.uint8)
        )

    rows.sort(
        key=lambda row: (
            str(row["split"]),
            -int(row["fine_pixels"]),
            str(row["id"]),
        )
    )
    by_split = {
        split: [row for row in rows if row["split"] == split]
        for split in args.splits
    }
    if any(not by_split[split] for split in by_split):
        raise ValueError("Every requested split must contain candidates")
    save_csv(output / "fine_sample_candidates.csv", rows)
    for split in args.splits:
        contact_sheet(
            output,
            split,
            by_split[split],
            images,
            maximum_items=args.contact_sheet_items,
        )

    fixed_ids = {
        "train": args.train_sample_id,
        "val": args.val_sample_id,
        "test": args.test_sample_id,
    }
    locked = {}
    for split in args.splits:
        if fixed_ids[split] is None:
            locked[split] = by_split[split][0]
            continue
        matches = [
            row for row in by_split[split] if row["id"] == fixed_ids[split]
        ]
        if len(matches) != 1:
            raise ValueError(
                f"Fixed {split} sample was not found exactly once: {fixed_ids[split]}"
            )
        locked[split] = matches[0]

    selection = {
        "status": args.selection_status,
        "selection_inputs": "ground-truth depth and RGB only; no model prediction",
        "selection_rule": (
            "fixed ids were chosen from GT/RGB contact sheets before rendering "
            "model predictions"
        ),
        "mask_scope": (
            "Appendix B.1 coarse proposal without SAM2 expansion; diagnostic only"
        ),
        "shape": [args.height, args.width],
        "crop_shape": [args.crop_height, args.crop_width],
        **locked,
        "artifacts": {
            "ranking": "fine_sample_candidates.csv",
            **{
                f"{split}_contact_sheet": f"{split}_fine_sample_candidates.png"
                for split in args.splits
            },
        },
    }
    (output / "fine_sample_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, ensure_ascii=False))


if __name__ == "__main__":
    main()
