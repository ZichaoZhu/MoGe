from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from moge.model.ssr import (
    capture_batch_norm_running_state,
    stateless_batch_statistics,
)
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.export_exp9_pointclouds import (
    EXPECTED_STEPS,
    VOXEL_DEPTH_SCALE,
    point_asset,
    read_binary_ply,
    write_binary_ply,
)

if TYPE_CHECKING:
    from moge.model.v3 import MoGeModel


SUPPORTED_EXPERIMENT = "exp20_stateless_ssr_batch_statistics"
SPLITS = ("train", "val", "test")
DEFAULT_RELATIVE_PATHS = {
    "data": (
        "experiment/stage2_training_strategy/runs/"
        "exp11_hypersim_100_train_staged_joint_overfit/data"
    ),
    "checkpoint": (
        "experiment/stage2_training_strategy/runs/"
        "exp15_detached_head_ssr_coadaptation/artifacts/formal/checkpoint.pt"
    ),
    "viewer": (
        "experiment/stage2_training_strategy/runs/"
        "exp9_thin_structure_single_image_staged_overfit/viewer"
    ),
    "report_output": (
        "experiment/stage3_stability_normalization/runs/"
        "exp20_stateless_ssr_batch_statistics/results/viewer_export"
    ),
    "reference_metrics": (
        "experiment/stage3_stability_normalization/runs/"
        "exp20_stateless_ssr_batch_statistics/results/remote/metrics/"
        "per_frame_metrics.csv"
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export one checkpoint/inference policy as a split-aware v2 point-"
            "cloud manifest for the browser viewer."
        )
    )
    parser.add_argument("--experiment", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=SPLITS,
        default=list(SPLITS),
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=list(EXPECTED_STEPS),
    )
    parser.add_argument("--selection-manifest", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--data", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--viewer", type=Path)
    parser.add_argument("--report-output", type=Path)
    parser.add_argument("--reference-metrics", type=Path)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--crop-stride", type=int, default=16)
    parser.add_argument("--edge-relative-threshold", type=float, default=0.05)
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    parser.add_argument("--metric-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--initial-seed",
        type=int,
        default=151,
        help=(
            "Torch seed used before constructing the MoGe-2 + zero-initialized "
            "SSR training-start model."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash an in-memory model state without materializing a checkpoint file."""
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
        byte_view = tensor.reshape(-1).view(torch.uint8).numpy()
        digest.update(memoryview(byte_view))
    return digest.hexdigest()


def resolve_default_path(
    explicit: Path | None,
    *,
    project_root: Path,
    key: str,
) -> Path:
    return explicit if explicit is not None else project_root / DEFAULT_RELATIVE_PATHS[key]


def validate_selection(
    selection: Mapping[str, Any],
    *,
    experiment: str,
    splits: Sequence[str],
    expected_counts: Mapping[str, int] | None = None,
) -> None:
    if selection.get("version") not in {1, 2}:
        raise ValueError("Selection manifest version must be 1 or 2")
    if selection.get("experiment") != experiment:
        raise ValueError("Selection manifest experiment does not match")
    seen: set[str] = set()
    for split in splits:
        entries = selection.get("splits", {}).get(split)
        expected_count = (
            int(expected_counts.get(split, 5))
            if expected_counts is not None
            else 5
        )
        if not isinstance(entries, list) or len(entries) != expected_count:
            raise ValueError(
                f"{split} must contain exactly {expected_count} samples"
            )
        pictures = [int(entry["picture"]) for entry in entries]
        expected_pictures = list(range(1, expected_count + 1))
        if pictures != expected_pictures:
            raise ValueError(
                f"{split} picture numbers must be 1..{expected_count} in order"
            )
        for entry in entries:
            sample_id = str(entry["id"])
            if sample_id in seen:
                raise ValueError(f"Duplicate selected sample: {sample_id}")
            seen.add(sample_id)
            if int(entry["rank"]) < 1 or int(entry["rank"]) > int(entry["total"]):
                raise ValueError(f"Invalid rank for {sample_id}")


def _window_starts(length: int, size: int, stride: int) -> list[int]:
    if not 0 < size <= length or stride <= 0:
        raise ValueError("Crop size/stride is incompatible with image dimensions")
    starts = list(range(0, length - size + 1, stride))
    final = length - size
    if starts[-1] != final:
        starts.append(final)
    return starts


def _integral_sum(integral: np.ndarray, x: int, y: int, size: int) -> float:
    x1, y1 = x + size, y + size
    return float(
        integral[y1, x1]
        - integral[y, x1]
        - integral[y1, x]
        + integral[y, x]
    )


def depth_edge_crop(
    gt_points: torch.Tensor | np.ndarray,
    *,
    size: int = 192,
    stride: int = 16,
    relative_threshold: float = 0.05,
    minimum_valid_ratio: float = 0.9,
) -> list[int]:
    points = (
        gt_points.detach().cpu().numpy()
        if isinstance(gt_points, torch.Tensor)
        else np.asarray(gt_points)
    )
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError("Ground-truth point map must have shape HxWx3")
    height, width = points.shape[:2]
    depth = points[..., 2].astype(np.float64, copy=False)
    valid = np.isfinite(points).all(axis=-1) & np.isfinite(depth) & (depth > 0)
    safe_depth = np.where(valid, depth, 1.0)
    log_depth = np.log(safe_depth)
    threshold = math.log1p(relative_threshold)
    edges = np.zeros((height, width), dtype=np.float64)
    horizontal_valid = valid[:, 1:] & valid[:, :-1]
    vertical_valid = valid[1:, :] & valid[:-1, :]
    horizontal = horizontal_valid & (
        np.abs(log_depth[:, 1:] - log_depth[:, :-1]) > threshold
    )
    vertical = vertical_valid & (
        np.abs(log_depth[1:, :] - log_depth[:-1, :]) > threshold
    )
    edges[:, 1:] += horizontal
    edges[:, :-1] += horizontal
    edges[1:, :] += vertical
    edges[:-1, :] += vertical
    edge_integral = np.pad(edges, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    valid_integral = (
        np.pad(valid.astype(np.float64), ((1, 0), (1, 0)))
        .cumsum(0)
        .cumsum(1)
    )
    candidates: list[tuple[float, int, int, float]] = []
    for y in _window_starts(height, size, stride):
        for x in _window_starts(width, size, stride):
            valid_ratio = _integral_sum(valid_integral, x, y, size) / (size * size)
            edge_score = _integral_sum(edge_integral, x, y, size)
            candidates.append((edge_score, y, x, valid_ratio))
    for required_ratio in (minimum_valid_ratio, 0.75, 0.0):
        eligible = [
            item for item in candidates if item[3] >= required_ratio
        ]
        if eligible:
            score, y, x, _ = min(
                eligible,
                key=lambda item: (-item[0], item[1], item[2]),
            )
            if score < 0:
                raise AssertionError("Depth-edge score cannot be negative")
            return [x, y, x + size, y + size]
    raise ValueError("No crop candidate was generated")


def batch_norm_state_sha256(
    state: Mapping[str, Mapping[str, torch.Tensor]],
) -> str:
    digest = hashlib.sha256()
    for module_name in sorted(state):
        digest.update(module_name.encode("utf-8"))
        for buffer_name in (
            "running_mean",
            "running_var",
            "num_batches_tracked",
        ):
            tensor = state[module_name][buffer_name].detach().cpu().contiguous()
            digest.update(buffer_name.encode("utf-8"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(np.asarray(tensor.shape, dtype=np.int64).tobytes())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def batch_norm_states_equal(
    first: Mapping[str, Mapping[str, torch.Tensor]],
    second: Mapping[str, Mapping[str, torch.Tensor]],
) -> bool:
    return set(first) == set(second) and all(
        torch.equal(first[name][buffer], second[name][buffer])
        for name in first
        for buffer in first[name]
    )


def load_reference_metrics(path: Path) -> Dict[str, Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {str(row["id"]): row for row in rows}


def validate_full_metrics(
    *,
    sample_id: str,
    step: int,
    actual: Mapping[str, float],
    reference: Mapping[str, str],
    tolerance: float,
) -> Dict[str, float]:
    columns = {
        "point_rel": f"k{step}_point_rel",
        "depth_rel": f"k{step}_depth_rel",
        "depth_delta_1.01": f"k{step}_depth_delta_1.01",
        "depth_delta_1.25": f"k{step}_depth_delta_1.25",
        "boundary_f1": f"k{step}_boundary_f1",
    }
    differences: Dict[str, float] = {}
    for metric, column in columns.items():
        expected = float(reference[column])
        difference = abs(float(actual[metric]) - expected)
        differences[metric] = difference
        allowed = (
            max(tolerance, 5e-4)
            if metric in {"point_rel", "depth_rel"}
            else max(tolerance, 1e-2)
        )
        if not math.isclose(
            float(actual[metric]),
            expected,
            rel_tol=allowed,
            abs_tol=allowed,
        ):
            raise ValueError(
                f"{sample_id} K={step} {metric}={actual[metric]} "
                f"does not match reference {expected}"
            )
    return differences


def canonical_full_metrics(
    reference: Mapping[str, str],
    *,
    step: int,
    pixels: int,
) -> Dict[str, float | int]:
    return {
        "pixels": pixels,
        "point_rel": float(reference[f"k{step}_point_rel"]),
        "depth_rel": float(reference[f"k{step}_depth_rel"]),
        "depth_delta_1.01": float(
            reference[f"k{step}_depth_delta_1.01"]
        ),
        "depth_delta_1.25": float(
            reference[f"k{step}_depth_delta_1.25"]
        ),
        "boundary_f1": float(reference[f"k{step}_boundary_f1"]),
    }


def checkpoint_model_metadata(
    checkpoint: Mapping[str, Any],
) -> tuple[str, str]:
    pretrained = checkpoint.get("pretrained")
    if not pretrained:
        raise ValueError("Checkpoint does not identify its pretrained base model")
    normalization = checkpoint.get("args", {}).get(
        "ssr_normalization",
        "batch_norm",
    )
    if normalization != "batch_norm":
        raise ValueError("Exp20 requires the BatchNorm SSR checkpoint")
    return str(pretrained), str(normalization)


def construct_initial_model(
    checkpoint: Mapping[str, Any],
    *,
    seed: int,
    device: torch.device,
) -> tuple[MoGeModel, str]:
    from moge.model.v3 import MoGeModel

    pretrained, normalization = checkpoint_model_metadata(checkpoint)
    torch.manual_seed(seed)
    model = MoGeModel.from_pretrained(
        pretrained,
        model_kwargs={"ssr": {"normalization": normalization}},
    ).eval()
    state_digest = state_dict_sha256(model.state_dict())
    return model.to(device), state_digest


def construct_checkpoint_model(
    checkpoint: Mapping[str, Any],
    *,
    device: torch.device,
) -> MoGeModel:
    from moge.model.v3 import MoGeModel

    pretrained, normalization = checkpoint_model_metadata(checkpoint)
    model = MoGeModel.from_pretrained(
        pretrained,
        model_kwargs={"ssr": {"normalization": normalization}},
    )
    model.load_state_dict(checkpoint["model"], strict=True)
    return model.to(device).eval()


@torch.no_grad()
def predict_and_measure(
    model: MoGeModel,
    image: torch.Tensor,
    gt_points: torch.Tensor,
    *,
    crop: Sequence[int],
    num_tokens: int,
    steps: Sequence[int],
    boundary_threshold: float,
    smooth_log_depth_residual_bound: float = 0.0,
) -> tuple[Dict[int, np.ndarray], Dict[int, Dict[str, Any]], int]:
    from moge.scripts.overfit_hypersim_staged_v3 import (
        _scope_metrics,
        build_structure_mask,
    )
    from moge.train.losses_v3 import solve_global_affine_alignment

    output = model(
        image[None],
        num_tokens=num_tokens,
        num_refinement_steps=max(steps),
        return_intermediates=True,
        smooth_log_depth_residual_bound=smooth_log_depth_residual_bound,
    )
    sequence = output["points_sequence"]
    gt = gt_points.to(image.device)
    full_valid = torch.isfinite(gt).all(dim=-1) & (gt[..., 2] > 0)
    x0, y0, x1, y1 = (int(value) for value in crop)
    crop_valid = full_valid[y0:y1, x0:x1]
    structure_mask = build_structure_mask(
        gt,
        crop,
        {"type": "near_quantile", "quantile": 0.5},
    )
    raw_by_k: Dict[int, np.ndarray] = {}
    metrics_by_k: Dict[int, Dict[str, Any]] = {}
    for step in steps:
        prediction = sequence[step][0].float()
        alignment = solve_global_affine_alignment(prediction[None], gt[None])
        aligned = alignment.apply(prediction[None])[0]
        raw_by_k[step] = prediction.detach().cpu().numpy()
        metrics_by_k[step] = {
            "full": _scope_metrics(
                aligned,
                gt,
                full_valid,
                boundary_threshold=boundary_threshold,
            ),
            "crop": _scope_metrics(
                aligned[y0:y1, x0:x1],
                gt[y0:y1, x0:x1],
                crop_valid,
                boundary_threshold=boundary_threshold,
            ),
            "structure": _scope_metrics(
                aligned[y0:y1, x0:x1],
                gt[y0:y1, x0:x1],
                structure_mask,
                boundary_threshold=boundary_threshold,
            ),
            "alignment": {
                "scale": float(alignment.scale.mean().item()),
                "zShift": float(alignment.shift[..., 2].mean().item()),
            },
        }
    return raw_by_k, metrics_by_k, int(structure_mask.sum().item())


def _public_url(path: Path, public_root: Path) -> str:
    return "/" + path.relative_to(public_root).as_posix()


def _selected_entries(
    selection: Mapping[str, Any],
    splits: Sequence[str],
) -> Iterable[tuple[str, Mapping[str, Any]]]:
    for split in splits:
        for entry in selection["splits"][split]:
            yield split, entry


def run(args: argparse.Namespace) -> Dict[str, Any]:
    from moge.scripts.train_hypersim_smallset_v3 import load_raw_sample

    if args.experiment != SUPPORTED_EXPERIMENT:
        raise ValueError(f"Unsupported experiment: {args.experiment}")
    splits = tuple(dict.fromkeys(str(split) for split in args.splits))
    if not splits or any(split not in SPLITS for split in splits):
        raise ValueError("Splits must be a non-empty subset of train/val/test")
    steps = tuple(sorted(set(int(step) for step in args.steps)))
    if steps != EXPECTED_STEPS:
        raise ValueError(f"Expected exactly K={EXPECTED_STEPS}, received {steps}")
    if min(
        args.height,
        args.width,
        args.num_tokens,
        args.crop_size,
        args.crop_stride,
    ) <= 0:
        raise ValueError("Image, token and crop parameters must be positive")

    project_root = assert_safe_path(
        args.project_root,
        safe_root=args.safe_root,
        must_exist=True,
    )
    paths = {
        key: resolve_default_path(
            getattr(args, key),
            project_root=project_root,
            key=key,
        )
        for key in DEFAULT_RELATIVE_PATHS
    }
    data = assert_safe_path(paths["data"], safe_root=args.safe_root, must_exist=True)
    checkpoint_path = assert_safe_path(
        paths["checkpoint"],
        safe_root=args.safe_root,
        must_exist=True,
    )
    viewer = assert_safe_path(
        paths["viewer"],
        safe_root=args.safe_root,
        must_exist=True,
        writable=True,
    )
    report_output = assert_safe_path(
        paths["report_output"],
        safe_root=args.safe_root,
        writable=True,
    )
    reference_metrics_path = assert_safe_path(
        paths["reference_metrics"],
        safe_root=args.safe_root,
        must_exist=True,
    )
    selection_path = assert_safe_path(
        args.selection_manifest,
        safe_root=args.safe_root,
        must_exist=True,
    )
    report_output.mkdir(parents=True, exist_ok=True)

    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    validate_selection(selection, experiment=args.experiment, splits=splits)
    data_manifest = json.loads(
        (data / "manifest.json").read_text(encoding="utf-8")
    )
    metadata_by_id = {
        str(sample["id"]): sample for sample in data_manifest["samples"]
    }
    reference_by_id = load_reference_metrics(reference_metrics_path)
    public_root = viewer / "public"
    output_root = public_root / "data" / "exp20"
    output_root.mkdir(parents=True, exist_ok=True)

    prepared: list[Dict[str, Any]] = []
    for split, selected in _selected_entries(selection, splits):
        sample_id = str(selected["id"])
        metadata = metadata_by_id.get(sample_id)
        if metadata is None or metadata.get("split") != split:
            raise ValueError(f"{sample_id} is absent from the {split} split")
        if sample_id not in reference_by_id:
            raise ValueError(f"{sample_id} is absent from reference metrics")
        image, gt_points = load_raw_sample(
            data,
            metadata,
            args.height,
            args.width,
            args.safe_root,
        )
        crop = depth_edge_crop(
            gt_points,
            size=args.crop_size,
            stride=args.crop_stride,
            relative_threshold=args.edge_relative_threshold,
        )
        colors = (
            image.permute(1, 2, 0).numpy().clip(0.0, 1.0) * 255.0
        ).round().astype(np.uint8)
        sample_output = output_root / sample_id
        sample_output.mkdir(parents=True, exist_ok=True)
        rgb_output = sample_output / "source_rgb.jpg"
        Image.fromarray(colors).save(rgb_output, quality=95, subsampling=0)
        prepared.append(
            {
                "id": sample_id,
                "split": split,
                "selection": dict(selected),
                "image": image,
                "gt_points": gt_points,
                "crop": crop,
                "colors": colors,
                "rgb_url": _public_url(rgb_output, public_root),
            }
        )

    device = torch.device(args.device)
    checkpoint_digest = sha256_file(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("step", -1)) != 800:
        raise ValueError(f"Exp20 requires checkpoint step 800, got {checkpoint.get('step')}")
    pretrained_name = str(checkpoint["pretrained"])
    checksums: list[Dict[str, Any]] = []
    sample_records: list[Dict[str, Any]] = []
    metric_records: Dict[str, Any] = {}
    validation_differences: Dict[str, Dict[str, Dict[str, float]]] = {}
    initial_assets_by_id: Dict[str, Dict[str, Any]] = {}
    initial_metrics_by_id: Dict[str, Dict[str, Any]] = {}
    structure_pixels_by_id: Dict[str, int] = {}

    initial_model, initialization_digest = construct_initial_model(
        checkpoint,
        seed=args.initial_seed,
        device=device,
    )
    zero_identity_max_error = 0.0
    for sample in prepared:
        raw_by_k, metrics_by_k, structure_pixels = predict_and_measure(
            initial_model,
            sample["image"].to(device),
            sample["gt_points"],
            crop=sample["crop"],
            num_tokens=args.num_tokens,
            steps=steps,
            boundary_threshold=args.boundary_threshold,
        )
        initial_points = raw_by_k[0]
        for step in steps[1:]:
            difference = float(
                np.max(np.abs(raw_by_k[step] - initial_points))
            )
            zero_identity_max_error = max(zero_identity_max_error, difference)
            if difference != 0.0:
                raise RuntimeError(
                    f"{sample['id']} zero-initialized SSR is not an exact "
                    f"identity at K={step}: max error {difference}"
                )
        output = output_root / sample["id"] / "initial_k0.ply"
        write_binary_ply(
            output,
            initial_points,
            sample["colors"],
            width=args.width,
            height=args.height,
        )
        decoded_points, decoded_colors, comments = read_binary_ply(output)
        np.testing.assert_array_equal(
            decoded_points,
            initial_points.reshape(-1, 3),
        )
        np.testing.assert_array_equal(
            decoded_colors,
            sample["colors"].reshape(-1, 3),
        )
        if comments.get("vertex_order") != "row_major_one_vertex_per_pixel":
            raise ValueError("PLY raster order was not preserved")
        initial_metrics = metrics_by_k[0]
        initial_asset = point_asset(
            output,
            public_root=public_root,
            points=initial_points,
            checkpoint_digest=initialization_digest,
            alignment=initial_metrics["alignment"],
            metrics={
                scope: initial_metrics[scope]
                for scope in ("full", "crop", "structure")
            },
        )
        initial_asset["pointRelReductionFromK0"] = 0.0
        initial_assets_by_id[sample["id"]] = {
            "0": initial_asset,
            "1": {"alias": "initial.0"},
            "3": {"alias": "initial.0"},
            "5": {"alias": "initial.0"},
        }
        initial_metrics_by_id[sample["id"]] = {
            "0": initial_asset["metrics"],
        }
        structure_pixels_by_id[sample["id"]] = structure_pixels
        checksums.append(
            {
                "sample": sample["id"],
                "split": sample["split"],
                "stage": "initial",
                "k": 0,
                "aliases": [1, 3, 5],
                "url": initial_asset["url"],
                "bytes": output.stat().st_size,
                "sha256": initial_asset["sha256"],
            }
        )
    del initial_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    model = construct_checkpoint_model(checkpoint, device=device)
    before_state = capture_batch_norm_running_state(model.ssr)
    before_sha = batch_norm_state_sha256(before_state)
    with stateless_batch_statistics(model.ssr):
        for sample in prepared:
            raw_by_k, metrics_by_k, structure_pixels = predict_and_measure(
                model,
                sample["image"].to(device),
                sample["gt_points"],
                crop=sample["crop"],
                num_tokens=args.num_tokens,
                steps=steps,
                boundary_threshold=args.boundary_threshold,
            )
            if structure_pixels != structure_pixels_by_id[sample["id"]]:
                raise RuntimeError("GT structure mask changed between stages")
            reference = reference_by_id[sample["id"]]
            validation_differences[sample["id"]] = {}
            for step in steps:
                validation_differences[sample["id"]][str(step)] = (
                    validate_full_metrics(
                        sample_id=sample["id"],
                        step=step,
                        actual=metrics_by_k[step]["full"],
                        reference=reference,
                        tolerance=args.metric_tolerance,
                    )
                )
                metrics_by_k[step]["full"] = canonical_full_metrics(
                    reference,
                    step=step,
                    pixels=int(metrics_by_k[step]["full"]["pixels"]),
                )
            k0_rel = float(metrics_by_k[0]["full"]["point_rel"])
            stage_assets: Dict[str, Any] = {}
            for step in steps:
                output = output_root / sample["id"] / f"final_k{step}.ply"
                write_binary_ply(
                    output,
                    raw_by_k[step],
                    sample["colors"],
                    width=args.width,
                    height=args.height,
                )
                decoded_points, decoded_colors, comments = read_binary_ply(output)
                np.testing.assert_array_equal(
                    decoded_points,
                    raw_by_k[step].reshape(-1, 3),
                )
                np.testing.assert_array_equal(
                    decoded_colors,
                    sample["colors"].reshape(-1, 3),
                )
                if comments.get("vertex_order") != "row_major_one_vertex_per_pixel":
                    raise ValueError("PLY raster order was not preserved")
                metrics = metrics_by_k[step]
                asset = point_asset(
                    output,
                    public_root=public_root,
                    points=raw_by_k[step],
                    checkpoint_digest=checkpoint_digest,
                    alignment=metrics["alignment"],
                    metrics={
                        scope: metrics[scope]
                        for scope in ("full", "crop", "structure")
                    },
                )
                current_rel = float(metrics["full"]["point_rel"])
                asset["pointRelReductionFromK0"] = (
                    0.0 if step == 0 else (k0_rel - current_rel) / max(k0_rel, 1e-12)
                )
                stage_assets[str(step)] = asset
                checksums.append(
                    {
                        "sample": sample["id"],
                        "split": sample["split"],
                        "stage": "final",
                        "k": step,
                        "url": asset["url"],
                        "bytes": output.stat().st_size,
                        "sha256": asset["sha256"],
                    }
                )
            expected_improvement = float(
                sample["selection"]["relativeImprovement"]
            )
            actual_improvement = float(
                stage_assets["3"]["pointRelReductionFromK0"]
            )
            if not math.isclose(
                expected_improvement,
                actual_improvement,
                rel_tol=2e-4,
                abs_tol=5e-5,
            ):
                raise ValueError(
                    f"{sample['id']} locked K=3 improvement "
                    f"{expected_improvement} != {actual_improvement}"
                )
            selection_record = {
                "metric": str(selection["metric"]),
                "policy": str(selection["policy"][sample["split"]]),
                "rank": int(sample["selection"]["rank"]),
                "total": int(sample["selection"]["total"]),
                "relativeImprovement": actual_improvement,
            }
            sample_record = {
                "id": sample["id"],
                "split": sample["split"],
                "label": f"图片 {int(sample['selection']['picture'])}",
                "order": int(sample["selection"]["picture"]),
                "description": (
                    f"{sample['split']} 第 {selection_record['rank']}/"
                    f"{selection_record['total']} 名"
                ),
                "websiteEnabled": True,
                "cropXYXY": sample["crop"],
                "structureMask": {
                    "type": "near_quantile",
                    "quantile": 0.5,
                    "pixels": structure_pixels,
                },
                "rgbUrl": sample["rgb_url"],
                "selection": selection_record,
                "stages": {
                    "initial": initial_assets_by_id[sample["id"]],
                    "final": stage_assets,
                },
            }
            sample_records.append(sample_record)
            metric_records[sample["id"]] = {
                "split": sample["split"],
                "cropXYXY": sample["crop"],
                "selection": selection_record,
                "initial": initial_metrics_by_id[sample["id"]],
                "final": {
                    step: stage_assets[step]["metrics"]
                    for step in stage_assets
                },
            }

    after_state = capture_batch_norm_running_state(model.ssr)
    after_sha = batch_norm_state_sha256(after_state)
    buffers_unchanged = batch_norm_states_equal(before_state, after_state)
    if not buffers_unchanged or before_sha != after_sha:
        raise RuntimeError("SSR BatchNorm running buffers changed during export")
    del model, checkpoint
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    order_by_split = {
        split: [
            record["id"]
            for record in sample_records
            if record["split"] == split
        ]
        for split in splits
    }
    browser_manifest = {
        "version": 2,
        "experiment": args.experiment,
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
        "resolution": {"width": args.width, "height": args.height},
        "steps": list(steps),
        "availableSplits": list(splits),
        "defaultSplit": "train" if "train" in splits else splits[0],
        "websiteSampleOrderBySplit": order_by_split,
        "availableStages": ["initial", "final"],
        "defaultStages": {"left": "initial", "right": "final"},
        "samples": sample_records,
        "provenance": {
            "sourceExperiment": (
                "exp15_detached_head_ssr_coadaptation"
            ),
            "initialization": (
                "官方 MoGe-2 ViT-L 与零初始化 SSR"
            ),
            "initializationSeed": args.initial_seed,
            "initializationStateSha256": initialization_digest,
            "checkpointStep": 800,
            "checkpointSha256": checkpoint_digest,
            "inferencePolicy": (
                "current_batch_statistics_without_running_buffers"
            ),
            "selectionMetric": str(selection["metric"]),
        },
    }
    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(browser_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    checksum_path = output_root / "pointcloud_sha256.json"
    checksum_path.write_text(
        json.dumps(checksums, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    report = {
        "status": "complete",
        "experiment": args.experiment,
        "manifest": str(manifest_path),
        "selectionManifest": str(selection_path),
        "splits": order_by_split,
        "checkpointStep": 800,
        "checkpointSha256": checkpoint_digest,
        "initialization": {
            "pretrained": pretrained_name,
            "ssr": "zero-initialized",
            "seed": args.initial_seed,
            "stateSha256": initialization_digest,
            "zeroIdentityMaxError": zero_identity_max_error,
        },
        "inferencePolicy": (
            "current_batch_statistics_without_running_buffers"
        ),
        "batchNormRunningState": {
            "beforeSha256": before_sha,
            "afterSha256": after_sha,
            "unchanged": buffers_unchanged,
        },
        "evaluationSteps": list(steps),
        "pointCloudCount": len(checksums),
        "logicalPointCloudCount": len(prepared) * 8,
        "pointCloudBytes": sum(int(item["bytes"]) for item in checksums),
        "pointCloudChecksums": str(checksum_path),
        "metricValidation": {
            "requestedTolerance": args.metric_tolerance,
            "continuousEffectiveTolerance": max(
                args.metric_tolerance,
                5e-4,
            ),
            "thresholdMetricEffectiveTolerance": max(
                args.metric_tolerance,
                1e-2,
            ),
            "maxAbsoluteDifference": {
                metric: max(
                    validation_differences[sample_id][step][metric]
                    for sample_id in validation_differences
                    for step in validation_differences[sample_id]
                )
                for metric in (
                    "point_rel",
                    "depth_rel",
                    "depth_delta_1.01",
                    "depth_delta_1.25",
                    "boundary_f1",
                )
            },
        },
        "samples": [
            {
                "id": sample["id"],
                "split": sample["split"],
                "cropXYXY": sample["cropXYXY"],
                "selection": sample["selection"],
            }
            for sample in sample_records
        ],
    }
    (report_output / "selected_sample_metrics.json").write_text(
        json.dumps(metric_records, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (report_output / "pointcloud_export_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
