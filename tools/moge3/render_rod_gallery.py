from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from PIL import Image, ImageDraw

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.visualize_split_rods import validate_crop


SELECTION_INPUTS = "RGB and ground truth only; no predictions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and aggregate a thin-structure GIF gallery"
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=192)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num-tokens", type=int, default=1200)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--mask-dilation", type=int, default=0)
    parser.add_argument("--yaw-min", type=float, default=-90.0)
    parser.add_argument("--yaw-max", type=float, default=90.0)
    parser.add_argument("--orbit-frames", type=int, default=46)
    parser.add_argument("--frame-duration-ms", type=int, default=110)
    parser.add_argument(
        "--experiment-label",
        default="exp2_hypersim_smallset_overfit",
    )
    parser.add_argument("--aggregate-only", action="store_true")
    return parser.parse_args()


def validate_gallery_selection(
    selection: Mapping[str, object],
    manifest: Mapping[str, object],
    *,
    height: int,
    width: int,
) -> List[Dict[str, object]]:
    if selection.get("status") != "locked-before-rendering":
        raise ValueError("gallery selection must be locked-before-rendering")
    if selection.get("selection_inputs") != SELECTION_INPUTS:
        raise ValueError("gallery selection provenance must exclude predictions")
    if list(selection.get("shape", [])) != [height, width]:
        raise ValueError("gallery selection shape does not match render shape")
    raw_entries = selection.get("entries")
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("gallery selection must contain a non-empty entries list")
    expected_count = int(selection.get("expected_count", len(raw_entries)))
    if len(raw_entries) != expected_count:
        raise ValueError(
            f"expected {expected_count} entries, found {len(raw_entries)}"
        )

    manifest_by_id = {
        str(sample["id"]): sample for sample in manifest.get("samples", [])
    }
    entries: List[Dict[str, object]] = []
    seen_ids = set()
    seen_orders = set()
    for raw_entry in raw_entries:
        if not isinstance(raw_entry, dict):
            raise ValueError("every gallery entry must be an object")
        sample_id = str(raw_entry["id"])
        order = int(raw_entry["order"])
        if sample_id in seen_ids or order in seen_orders:
            raise ValueError("gallery sample ids and orders must be unique")
        if sample_id not in manifest_by_id:
            raise ValueError(f"gallery sample is absent from manifest: {sample_id}")
        if str(manifest_by_id[sample_id]["split"]) != "train":
            raise ValueError(f"gallery sample is not a training sample: {sample_id}")
        display_mask = raw_entry.get("display_mask")
        if not isinstance(display_mask, dict):
            raise ValueError(f"gallery entry has no display mask: {sample_id}")
        crop = validate_crop(raw_entry["crop_xyxy"], height, width)
        entries.append(
            {
                **raw_entry,
                "id": sample_id,
                "order": order,
                "crop_xyxy": list(crop),
            }
        )
        seen_ids.add(sample_id)
        seen_orders.add(order)
    entries.sort(key=lambda entry: int(entry["order"]))
    if [int(entry["order"]) for entry in entries] != list(
        range(1, len(entries) + 1)
    ):
        raise ValueError("gallery orders must be contiguous and one-based")
    return entries


def shard_entries(
    entries: Sequence[Mapping[str, object]],
    *,
    shard_index: int,
    num_shards: int,
) -> List[Mapping[str, object]]:
    if num_shards <= 0 or not 0 <= shard_index < num_shards:
        raise ValueError("invalid gallery shard")
    return [
        entry
        for entry in entries
        if (int(entry["order"]) - 1) % num_shards == shard_index
    ]


def item_slug(entry: Mapping[str, object]) -> str:
    return f"{int(entry['order']):02d}_{entry['id']}"


def one_item_selection(
    selection: Mapping[str, object],
    entry: Mapping[str, object],
) -> Dict[str, object]:
    return {
        "status": selection["status"],
        "selection_inputs": selection["selection_inputs"],
        "selection_rule": selection["selection_rule"],
        "shape": selection["shape"],
        "splits": {
            "train": {
                "id": entry["id"],
                "crop_xyxy": entry["crop_xyxy"],
                "display_mask": entry["display_mask"],
                "description": entry["description"],
            }
        },
    }


def run_item(
    *,
    entry: Mapping[str, object],
    selection: Mapping[str, object],
    data: Path,
    output: Path,
    checkpoint: Path,
    safe_root: Path,
    device: str,
    height: int,
    width: int,
    num_tokens: int,
    mask_dilation: int,
    yaw_min: float,
    yaw_max: float,
    orbit_frames: int,
    frame_duration_ms: int,
    experiment_label: str,
) -> None:
    slug = item_slug(entry)
    item_output = output / "items" / slug
    report_path = item_output / "report.json"
    if report_path.is_file():
        print(f"[skip] {slug}: report already exists", flush=True)
        return
    if item_output.exists() and any(item_output.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite incomplete non-empty item: {item_output}"
        )

    generated_selection = output / "generated_selections" / f"{slug}.json"
    generated_selection.parent.mkdir(parents=True, exist_ok=True)
    generated_selection.write_text(
        json.dumps(
            one_item_selection(selection, entry),
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    log_path = output / "logs" / f"{slug}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    visualizer = Path(__file__).resolve().with_name("visualize_split_rods.py")
    command = [
        sys.executable,
        str(visualizer),
        "--data",
        str(data),
        "--selection",
        str(generated_selection),
        "--output",
        str(item_output),
        "--safe-root",
        str(safe_root),
        "--series",
        "K=3",
        str(checkpoint),
        "3",
        "--experiment-label",
        f"{experiment_label} / {int(entry['order']):02d}",
        "--device",
        device,
        "--height",
        str(height),
        "--width",
        str(width),
        "--num-tokens",
        str(num_tokens),
        "--mask-dilation",
        str(mask_dilation),
        "--yaw-min",
        str(yaw_min),
        "--yaw-max",
        str(yaw_max),
        "--orbit-frames",
        str(orbit_frames),
        "--motion",
        "one-way",
        "--frame-duration-ms",
        str(frame_duration_ms),
    ]
    print(f"[render] {slug}", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(f"render failed for {slug}:\n{tail}")
    if not report_path.is_file():
        raise RuntimeError(f"render produced no report for {slug}")
    print(f"[done] {slug}", flush=True)


def save_contact_sheet(
    output: Path,
    entries: Sequence[Mapping[str, object]],
) -> None:
    columns = 3
    tile_width = 540
    label_height = 50
    panels = []
    for entry in entries:
        slug = item_slug(entry)
        preview_path = output / "items" / slug / "train_rod_preview.png"
        preview = Image.open(preview_path).convert("RGB")
        frame_height = preview.height // 3
        middle = preview.crop(
            (0, frame_height, preview.width, 2 * frame_height)
        )
        resized_height = round(middle.height * tile_width / middle.width)
        panels.append(
            middle.resize(
                (tile_width, resized_height),
                Image.Resampling.LANCZOS,
            )
        )
    tile_height = max(panel.height for panel in panels)
    rows = (len(panels) + columns - 1) // columns
    canvas = Image.new(
        "RGB",
        (columns * tile_width, rows * (tile_height + label_height)),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    for index, (entry, panel) in enumerate(zip(entries, panels)):
        column = index % columns
        row = index // columns
        x = column * tile_width
        y = row * (tile_height + label_height)
        draw.text(
            (x + 8, y + 5),
            f"{int(entry['order']):02d} {entry['id']}",
            fill="black",
        )
        draw.text(
            (x + 8, y + 25),
            f"crop xyxy: {' '.join(str(value) for value in entry['crop_xyxy'])}",
            fill="black",
        )
        canvas.paste(panel, (x, y + label_height))
    canvas.save(output / "gallery_contact_sheet.jpg", quality=92, optimize=True)


def aggregate(
    *,
    output: Path,
    selection_path: Path,
    selection: Mapping[str, object],
    entries: Sequence[Mapping[str, object]],
    experiment_label: str,
) -> None:
    rows = []
    item_reports = []
    for entry in entries:
        slug = item_slug(entry)
        report_path = output / "items" / slug / "report.json"
        if not report_path.is_file():
            raise FileNotFoundError(f"missing gallery report: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        split = report["splits"]["train"]
        metrics = split["metrics"]
        row = {
            "order": int(entry["order"]),
            "id": entry["id"],
            "scene": split["sample"]["scene"],
            "frame": int(split["sample"]["frame"]),
            "description": entry["description"],
            "crop_xyxy": " ".join(str(value) for value in entry["crop_xyxy"]),
            "display_mask_pixels": int(split["display_mask_pixels"]),
            "k0_point_rel": float(metrics["K=0"]["point_rel"]),
            "k3_point_rel": float(metrics["K=3"]["point_rel"]),
            "point_rel_delta": (
                float(metrics["K=3"]["point_rel"])
                - float(metrics["K=0"]["point_rel"])
            ),
            "k0_depth_rel": float(metrics["K=0"]["depth_rel"]),
            "k3_depth_rel": float(metrics["K=3"]["depth_rel"]),
            "depth_rel_delta": (
                float(metrics["K=3"]["depth_rel"])
                - float(metrics["K=0"]["depth_rel"])
            ),
            "gif": f"items/{slug}/train_rod_orbit.gif",
            "preview": f"items/{slug}/train_rod_preview.png",
        }
        rows.append(row)
        item_reports.append(report)

    metrics_path = output / "gallery_metrics.csv"
    with metrics_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    save_contact_sheet(output, entries)

    count = len(rows)
    summary = {
        "count": count,
        "point_rel_improved": sum(row["point_rel_delta"] < 0 for row in rows),
        "depth_rel_improved": sum(row["depth_rel_delta"] < 0 for row in rows),
        "mean_k0_point_rel": sum(row["k0_point_rel"] for row in rows) / count,
        "mean_k3_point_rel": sum(row["k3_point_rel"] for row in rows) / count,
        "mean_k0_depth_rel": sum(row["k0_depth_rel"] for row in rows) / count,
        "mean_k3_depth_rel": sum(row["k3_depth_rel"] for row in rows) / count,
    }
    gallery_report = {
        "status": "complete",
        "experiment": experiment_label,
        "selection": {
            "path": str(selection_path),
            "status": selection["status"],
            "inputs": selection["selection_inputs"],
            "rule": selection["selection_rule"],
        },
        "summary": summary,
        "checkpoint": item_reports[0]["checkpoints"][0],
        "render": item_reports[0]["render"],
        "artifacts": {
            "metrics": "gallery_metrics.csv",
            "contact_sheet": "gallery_contact_sheet.jpg",
            "index": "INDEX.md",
        },
        "items": rows,
    }
    (output / "gallery_report.json").write_text(
        json.dumps(gallery_report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    lines = [
        f"# {experiment_label} 训练图细结构 GIF",
        "",
        "选区只依据 RGB 与真值深度，并在渲染预测前锁定。"
        "所有动画均为单向 yaw −90°→+90°，共 46 帧。",
        "",
        "| # | 样本 | 结构 | K=0 点图 Rel | K=3 点图 Rel | GIF |",
        "|---:|---|---|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['order']:02d} | `{row['id']}` | {row['description']} | "
            f"{100 * row['k0_point_rel']:.3f}% | "
            f"{100 * row['k3_point_rel']:.3f}% | "
            f"[查看]({row['gif']}) |"
        )
    (output / "INDEX.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(gallery_report, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    data = assert_safe_path(
        args.data, safe_root=args.safe_root, must_exist=True
    )
    selection_path = assert_safe_path(
        args.selection, safe_root=args.safe_root, must_exist=True
    )
    checkpoint = assert_safe_path(
        args.checkpoint, safe_root=args.safe_root, must_exist=True
    )
    output = assert_safe_path(
        args.output, safe_root=args.safe_root, writable=True
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = assert_safe_path(
        data / "manifest.json",
        safe_root=args.safe_root,
        must_exist=True,
    )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = validate_gallery_selection(
        selection,
        manifest,
        height=args.height,
        width=args.width,
    )
    if args.aggregate_only:
        aggregate(
            output=output,
            selection_path=selection_path,
            selection=selection,
            entries=entries,
            experiment_label=args.experiment_label,
        )
        return

    selected = shard_entries(
        entries,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
    )
    for entry in selected:
        run_item(
            entry=entry,
            selection=selection,
            data=data,
            output=output,
            checkpoint=checkpoint,
            safe_root=args.safe_root,
            device=args.device,
            height=args.height,
            width=args.width,
            num_tokens=args.num_tokens,
            mask_dilation=args.mask_dilation,
            yaw_min=args.yaw_min,
            yaw_max=args.yaw_max,
            orbit_frames=args.orbit_frames,
            frame_duration_ms=args.frame_duration_ms,
            experiment_label=args.experiment_label,
        )


if __name__ == "__main__":
    main()
