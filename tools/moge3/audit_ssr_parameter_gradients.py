from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from moge.model.ssr import (
    capture_batch_norm_running_state,
    load_batch_norm_running_state,
)
from moge.model.v3 import MoGeModel
from moge.scripts.train_hypersim_joint_v3 import RawSample, load_raw_samples
from moge.train.losses_v3 import (
    affine_invariant_global_loss_v3,
    edge_angle_loss_v3,
    radial_partition_local_loss,
)
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


OBJECTIVES = (
    "global",
    "local",
    "edge_paper",
    "combined_checkpoint",
    "combined_paper",
)
BASE_OBJECTIVES = ("global", "local", "edge_paper")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure global/local/edge gradients in the actual SSR parameter "
            "space on deterministic two-image microbatches."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--pairs-per-split", type=int, default=4)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=["train", "val", "test"],
    )
    parser.add_argument("--local-scales", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--seed", type=int, default=261)
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def atomic_write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty audit table: {path}")
    temporary = path.with_suffix(path.suffix + ".incomplete")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def select_microbatches(
    samples: Sequence[RawSample],
    *,
    splits: Sequence[str],
    pairs_per_split: int,
) -> list[tuple[str, int, tuple[RawSample, RawSample]]]:
    if pairs_per_split <= 0:
        raise ValueError("pairs_per_split must be positive")
    selected = []
    for split in splits:
        candidates = [sample for sample in samples if sample.split == split]
        needed = 2 * pairs_per_split
        if len(candidates) < needed:
            raise ValueError(
                f"Split {split} has {len(candidates)} samples; {needed} are required"
            )
        for pair_index in range(pairs_per_split):
            start = pair_index * 2
            selected.append(
                (
                    split,
                    pair_index,
                    (candidates[start], candidates[start + 1]),
                )
            )
    return selected


def parameter_group(name: str) -> str:
    if name.startswith("unet."):
        name = name.removeprefix("unet.")
    root = name.split(".", 1)[0]
    if root in {"input_projection", "visual_projection"}:
        return "input_fusion"
    if root == "encoder_blocks":
        level = name.split(".", 2)[1]
        return "bottleneck" if level == "4" else "encoder"
    if root in {"downsample"}:
        return "encoder"
    if root in {"bottleneck_fusion"}:
        return "bottleneck"
    if root in {"upsample", "decoder_fusion", "decoder_blocks"}:
        return "decoder"
    if root == "output_layer":
        return "output"
    return "other"


def combine_gradients(
    gradients: Sequence[Sequence[torch.Tensor | None]],
    coefficients: Sequence[float],
) -> tuple[torch.Tensor | None, ...]:
    if len(gradients) != len(coefficients):
        raise ValueError("Gradient lists and coefficients must have equal length")
    if not gradients:
        raise ValueError("At least one gradient list is required")
    parameter_count = len(gradients[0])
    if any(len(values) != parameter_count for values in gradients):
        raise ValueError("All gradient lists must cover the same parameters")
    combined = []
    for parameter_index in range(parameter_count):
        value = None
        for values, coefficient in zip(gradients, coefficients, strict=True):
            gradient = values[parameter_index]
            if gradient is None or coefficient == 0:
                continue
            contribution = gradient * coefficient
            value = contribution if value is None else value + contribution
        combined.append(value)
    return tuple(combined)


def gradient_inner_product(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> float:
    if len(left) != len(right):
        raise ValueError("Gradient lists must have the same length")
    total = 0.0
    for lhs, rhs in zip(left, right, strict=True):
        if lhs is None or rhs is None:
            continue
        total += float(torch.sum(lhs.double() * rhs.double()).item())
    return total


def gradient_cosine(
    left: Sequence[torch.Tensor | None],
    right: Sequence[torch.Tensor | None],
) -> float:
    left_squared = gradient_inner_product(left, left)
    right_squared = gradient_inner_product(right, right)
    denominator = math.sqrt(left_squared * right_squared)
    if denominator == 0:
        return math.nan
    return gradient_inner_product(left, right) / denominator


def restrict_gradients(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    gradients: Sequence[torch.Tensor | None],
    *,
    scope: str,
) -> tuple[torch.Tensor | None, ...]:
    if len(named_parameters) != len(gradients):
        raise ValueError("Parameters and gradients must have equal length")
    if scope not in {"all", "non_output", "output"}:
        raise ValueError(f"Unknown parameter scope: {scope}")
    selected = []
    for (name, _), gradient in zip(
        named_parameters,
        gradients,
        strict=True,
    ):
        is_output = parameter_group(name) == "output"
        keep = (
            scope == "all"
            or (scope == "output" and is_output)
            or (scope == "non_output" and not is_output)
        )
        selected.append(gradient if keep else None)
    return tuple(selected)


def gradient_statistics(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    gradients: Sequence[torch.Tensor | None],
) -> dict[str, float | int]:
    if len(named_parameters) != len(gradients):
        raise ValueError("Parameters and gradients must have equal length")
    parameter_numel = sum(parameter.numel() for _, parameter in named_parameters)
    active_numel = 0
    active_tensors = 0
    squared = 0.0
    absolute = 0.0
    maximum = 0.0
    nonzero = 0
    for (_, parameter), gradient in zip(
        named_parameters,
        gradients,
        strict=True,
    ):
        if gradient is None:
            continue
        if not torch.isfinite(gradient).all():
            raise RuntimeError("SSR parameter gradient contains NaN or Inf")
        active_tensors += 1
        active_numel += parameter.numel()
        squared += float(gradient.double().square().sum().item())
        absolute += float(gradient.double().abs().sum().item())
        maximum = max(maximum, float(gradient.detach().abs().max().item()))
        nonzero += int(torch.count_nonzero(gradient).item())
    l2 = math.sqrt(squared)
    return {
        "parameter_numel": parameter_numel,
        "active_parameter_numel": active_numel,
        "active_parameter_tensors": active_tensors,
        "gradient_l2": l2,
        "gradient_rms": l2 / math.sqrt(parameter_numel),
        "gradient_l1": absolute,
        "gradient_abs_max": maximum,
        "gradient_nonzero_fraction": nonzero / parameter_numel,
    }


def grouped_gradient_rows(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    gradients: Sequence[torch.Tensor | None],
    *,
    common: Mapping[str, Any],
) -> list[dict[str, Any]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, (name, _) in enumerate(named_parameters):
        groups[parameter_group(name)].append(index)
    rows = []
    for group, indices in sorted(groups.items()):
        selected_parameters = [named_parameters[index] for index in indices]
        selected_gradients = [gradients[index] for index in indices]
        rows.append(
            {
                **common,
                "parameter_group": group,
                **gradient_statistics(selected_parameters, selected_gradients),
            }
        )
    return rows


def build_objectives(
    points_sequence: Sequence[torch.Tensor],
    gt_points: torch.Tensor,
    *,
    local_scales: Sequence[int],
    seed: int,
    edge_checkpoint_scale: float,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=gt_points.device).manual_seed(seed)
    totals = {
        name: gt_points.new_zeros(())
        for name in BASE_OBJECTIVES
    }
    for points in points_sequence:
        global_per_image, alignment = affine_invariant_global_loss_v3(
            points,
            gt_points,
        )
        local_per_image = radial_partition_local_loss(
            points,
            gt_points,
            alignment,
            scales=local_scales,
            generator=generator,
        )
        edge_per_image, _ = edge_angle_loss_v3(points, gt_points)
        totals["global"] = totals["global"] + global_per_image.mean()
        totals["local"] = totals["local"] + local_per_image.mean()
        totals["edge_paper"] = totals["edge_paper"] + edge_per_image.mean()
    totals["combined_checkpoint"] = (
        totals["global"]
        + totals["local"]
        + edge_checkpoint_scale * totals["edge_paper"]
    )
    totals["combined_paper"] = (
        totals["global"] + totals["local"] + totals["edge_paper"]
    )
    return totals


def parameter_gradient_records(
    named_parameters: Sequence[tuple[str, torch.nn.Parameter]],
    gradient_sets: Mapping[str, Sequence[torch.Tensor | None]],
    losses: Mapping[str, float],
    *,
    common: Mapping[str, Any],
    objective_metadata: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    paper_combined_squared = gradient_inner_product(
        gradient_sets["combined_paper"],
        gradient_sets["combined_paper"],
    )
    checkpoint_combined_squared = gradient_inner_product(
        gradient_sets["combined_checkpoint"],
        gradient_sets["combined_checkpoint"],
    )
    objective_rows = []
    group_rows = []
    for name in OBJECTIVES:
        gradients = gradient_sets[name]
        loss = float(losses[name])
        statistics = gradient_statistics(named_parameters, gradients)
        objective_rows.append(
            {
                **common,
                "objective": name,
                "loss": loss,
                **statistics,
                "loss_normalized_gradient_rms": statistics["gradient_rms"]
                / max(abs(loss), 1e-30),
                "cosine_to_paper_combined": gradient_cosine(
                    gradients,
                    gradient_sets["combined_paper"],
                ),
                "projection_fraction_to_paper_combined": (
                    gradient_inner_product(
                        gradients,
                        gradient_sets["combined_paper"],
                    )
                    / paper_combined_squared
                ),
                "cosine_to_checkpoint_combined": gradient_cosine(
                    gradients,
                    gradient_sets["combined_checkpoint"],
                ),
                "projection_fraction_to_checkpoint_combined": (
                    gradient_inner_product(
                        gradients,
                        gradient_sets["combined_checkpoint"],
                    )
                    / checkpoint_combined_squared
                ),
                **(objective_metadata or {}),
            }
        )
        group_rows.extend(
            grouped_gradient_rows(
                named_parameters,
                gradients,
                common={**common, "objective": name},
            )
        )
    cosine_rows = []
    for name, left, right in (
        ("global_local", "global", "local"),
        ("global_edge", "global", "edge_paper"),
        ("local_edge", "local", "edge_paper"),
    ):
        cosine_rows.append(
            {
                **common,
                "gradient_pair": name,
                "left_objective": left,
                "right_objective": right,
                "cosine": gradient_cosine(
                    gradient_sets[left],
                    gradient_sets[right],
                ),
            }
        )
    return objective_rows, cosine_rows, group_rows


def aggregate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    keys: Sequence[str],
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in keys)].append(row)
    ignored = {
        "batch_id",
        "pair_index",
        "sample_ids",
        *keys,
    }
    aggregates = []
    for group, selected in sorted(groups.items()):
        record = dict(zip(keys, group, strict=True))
        record["batch_count"] = len(selected)
        for key in selected[0]:
            if key in ignored:
                continue
            try:
                values = np.asarray(
                    [float(row[key]) for row in selected],
                    dtype=np.float64,
                )
            except (TypeError, ValueError):
                continue
            finite = values[np.isfinite(values)]
            if finite.size:
                record[f"{key}_mean"] = float(finite.mean())
                record[f"{key}_median"] = float(np.median(finite))
                record[f"{key}_min"] = float(finite.min())
                record[f"{key}_max"] = float(finite.max())
        aggregates.append(record)
    return aggregates


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not 1 <= args.refinement_steps <= 7:
        raise ValueError("refinement_steps must be in [1, 7]")
    if len(set(args.splits)) != len(args.splits):
        raise ValueError("splits must be unique")
    data = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    checkpoint_path = assert_safe_path(
        args.checkpoint,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(
        args.output,
        safe_root=args.safe_root,
        writable=True,
    )
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    samples = load_raw_samples(
        data,
        manifest,
        height=args.height,
        width=args.width,
        safe_root=args.safe_root,
        splits=tuple(args.splits),
    )
    microbatches = select_microbatches(
        samples,
        splits=args.splits,
        pairs_per_split=args.pairs_per_split,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained = checkpoint.get("pretrained")
    if not pretrained:
        raise ValueError("Checkpoint does not identify its pretrained base model")
    normalization = checkpoint.get("args", {}).get(
        "ssr_normalization",
        "batch_norm",
    )
    if normalization != "batch_norm":
        raise ValueError(
            "Exp26 requires the Exp24 BatchNorm checkpoint and true microbatch=2"
        )
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats()
    model = MoGeModel.from_pretrained(
        str(pretrained),
        model_kwargs={"ssr": {"normalization": normalization}},
    ).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)

    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.ssr.train()
    for parameter in model.ssr.parameters():
        parameter.requires_grad_(True)
    named_parameters = tuple(model.ssr.named_parameters())
    parameters = tuple(parameter for _, parameter in named_parameters)
    batch_norm_state = capture_batch_norm_running_state(model.ssr)
    edge_checkpoint_scale = min(args.height, args.width) / max(
        args.height,
        args.width,
    )

    objective_rows: list[dict[str, Any]] = []
    cosine_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    effective_objective_rows: list[dict[str, Any]] = []
    effective_cosine_rows: list[dict[str, Any]] = []
    effective_group_rows: list[dict[str, Any]] = []
    effective_gradients: dict[
        str,
        dict[str, tuple[torch.Tensor | None, ...]],
    ] = {}
    effective_losses: dict[str, dict[str, float]] = {}
    effective_sample_ids: dict[str, list[str]] = {}
    effective_maximum_residual: dict[str, float] = {}
    cross_split_gradients: dict[
        str,
        dict[str, tuple[torch.Tensor | None, ...]],
    ] = {}
    started = time.perf_counter()
    for batch_number, (split, pair_index, pair) in enumerate(microbatches):
        load_batch_norm_running_state(model.ssr, batch_norm_state)
        images = torch.stack([sample.image for sample in pair]).to(device)
        gt_points = torch.stack([sample.gt_points for sample in pair]).to(device)
        output_dict = model(
            images,
            num_tokens=args.num_tokens,
            num_refinement_steps=args.refinement_steps,
            return_intermediates=True,
            detach_base_from_refiner=True,
        )
        sequence = output_dict["points_sequence"][1:]
        objectives = build_objectives(
            sequence,
            gt_points,
            local_scales=tuple(args.local_scales),
            seed=args.seed + 1009 * batch_number,
            edge_checkpoint_scale=edge_checkpoint_scale,
        )
        base_gradients: dict[str, tuple[torch.Tensor | None, ...]] = {}
        for objective_index, name in enumerate(BASE_OBJECTIVES):
            gradients = torch.autograd.grad(
                objectives[name],
                parameters,
                retain_graph=objective_index < len(BASE_OBJECTIVES) - 1,
                allow_unused=True,
            )
            base_gradients[name] = tuple(
                gradient.detach() if gradient is not None else None
                for gradient in gradients
            )
        gradient_sets = {
            **base_gradients,
            "combined_checkpoint": combine_gradients(
                [base_gradients[name] for name in BASE_OBJECTIVES],
                [1.0, 1.0, edge_checkpoint_scale],
            ),
            "combined_paper": combine_gradients(
                [base_gradients[name] for name in BASE_OBJECTIVES],
                [1.0, 1.0, 1.0],
            ),
        }
        maximum_residual = max(
            float(residual.detach().abs().max().item())
            for residual in output_dict["log_depth_residuals"]
        )
        common = {
            "batch_id": f"{split}_{pair_index:02d}",
            "split": split,
            "pair_index": pair_index,
            "sample_ids": "|".join(sample.sample_id for sample in pair),
        }
        batch_records = parameter_gradient_records(
            named_parameters,
            gradient_sets,
            {
                name: float(objectives[name].detach().item())
                for name in OBJECTIVES
            },
            common=common,
            objective_metadata={
                "max_abs_log_depth_residual": maximum_residual,
            },
        )
        objective_rows.extend(batch_records[0])
        cosine_rows.extend(batch_records[1])
        group_rows.extend(batch_records[2])

        if split not in effective_gradients:
            effective_gradients[split] = {}
            effective_losses[split] = {name: 0.0 for name in OBJECTIVES}
            effective_sample_ids[split] = []
            effective_maximum_residual[split] = 0.0
        for name in BASE_OBJECTIVES:
            contribution = combine_gradients(
                [gradient_sets[name]],
                [1.0 / args.pairs_per_split],
            )
            if name in effective_gradients[split]:
                contribution = combine_gradients(
                    [effective_gradients[split][name], contribution],
                    [1.0, 1.0],
                )
            effective_gradients[split][name] = contribution
        for name in OBJECTIVES:
            effective_losses[split][name] += (
                float(objectives[name].detach().item())
                / args.pairs_per_split
            )
        effective_sample_ids[split].extend(
            sample.sample_id for sample in pair
        )
        effective_maximum_residual[split] = max(
            effective_maximum_residual[split],
            maximum_residual,
        )

        if pair_index == args.pairs_per_split - 1:
            split_base = effective_gradients[split]
            split_gradients = {
                **split_base,
                "combined_checkpoint": combine_gradients(
                    [split_base[name] for name in BASE_OBJECTIVES],
                    [1.0, 1.0, edge_checkpoint_scale],
                ),
                "combined_paper": combine_gradients(
                    [split_base[name] for name in BASE_OBJECTIVES],
                    [1.0, 1.0, 1.0],
                ),
            }
            effective_common = {
                "batch_id": (
                    f"{split}_effective_{2 * args.pairs_per_split:02d}"
                ),
                "split": split,
                "pair_index": -1,
                "sample_ids": "|".join(effective_sample_ids[split]),
            }
            effective_records = parameter_gradient_records(
                named_parameters,
                split_gradients,
                effective_losses[split],
                common=effective_common,
                objective_metadata={
                    "max_abs_log_depth_residual": (
                        effective_maximum_residual[split]
                    ),
                },
            )
            effective_objective_rows.extend(effective_records[0])
            effective_cosine_rows.extend(effective_records[1])
            effective_group_rows.extend(effective_records[2])
            cross_split_gradients[split] = {
                name: split_gradients[name]
                for name in (*BASE_OBJECTIVES, "combined_paper")
            }
            del (
                effective_gradients[split],
                effective_losses[split],
                effective_sample_ids[split],
                effective_maximum_residual[split],
                split_gradients,
            )
        del (
            images,
            gt_points,
            output_dict,
            sequence,
            objectives,
            base_gradients,
            gradient_sets,
        )

    load_batch_norm_running_state(model.ssr, batch_norm_state)
    objective_aggregates = aggregate_rows(
        objective_rows,
        keys=("split", "objective"),
    )
    cosine_aggregates = aggregate_rows(
        cosine_rows,
        keys=("split", "gradient_pair"),
    )
    group_aggregates = aggregate_rows(
        group_rows,
        keys=("split", "objective", "parameter_group"),
    )
    effective_scope_cosine_rows = []
    for split in args.splits:
        for pair_name, left, right in (
            ("global_local", "global", "local"),
            ("global_edge", "global", "edge_paper"),
            ("local_edge", "local", "edge_paper"),
        ):
            for scope in ("non_output", "output"):
                effective_scope_cosine_rows.append(
                    {
                        "split": split,
                        "gradient_pair": pair_name,
                        "parameter_scope": scope,
                        "cosine": gradient_cosine(
                            restrict_gradients(
                                named_parameters,
                                cross_split_gradients[split][left],
                                scope=scope,
                            ),
                            restrict_gradients(
                                named_parameters,
                                cross_split_gradients[split][right],
                                scope=scope,
                            ),
                        ),
                    }
                )
    cross_split_rows = []
    for left_index, left_split in enumerate(args.splits):
        for right_split in args.splits[left_index + 1 :]:
            for objective in (*BASE_OBJECTIVES, "combined_paper"):
                for scope in ("all", "non_output", "output"):
                    cross_split_rows.append(
                        {
                            "split_pair": f"{left_split}_{right_split}",
                            "left_split": left_split,
                            "right_split": right_split,
                            "objective": objective,
                            "parameter_scope": scope,
                            "cosine": gradient_cosine(
                                restrict_gradients(
                                    named_parameters,
                                    cross_split_gradients[left_split][objective],
                                    scope=scope,
                                ),
                                restrict_gradients(
                                    named_parameters,
                                    cross_split_gradients[right_split][objective],
                                    scope=scope,
                                ),
                            ),
                        }
                    )
    atomic_write_csv(output / "per_batch_objectives.csv", objective_rows)
    atomic_write_csv(output / "per_batch_cosines.csv", cosine_rows)
    atomic_write_csv(output / "per_group_gradients.csv", group_rows)
    atomic_write_csv(output / "aggregate_objectives.csv", objective_aggregates)
    atomic_write_csv(output / "aggregate_cosines.csv", cosine_aggregates)
    atomic_write_csv(output / "aggregate_groups.csv", group_aggregates)
    atomic_write_csv(
        output / "effective_batch_objectives.csv",
        effective_objective_rows,
    )
    atomic_write_csv(
        output / "effective_batch_cosines.csv",
        effective_cosine_rows,
    )
    atomic_write_csv(
        output / "effective_group_gradients.csv",
        effective_group_rows,
    )
    if cross_split_rows:
        atomic_write_csv(
            output / "effective_scope_cosines.csv",
            effective_scope_cosine_rows,
        )
        atomic_write_csv(
            output / "cross_split_cosines.csv",
            cross_split_rows,
        )

    report: dict[str, Any] = {
        "status": "complete",
        "checkpoint": str(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "ssr_normalization": normalization,
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "refinement_steps": args.refinement_steps,
        "microbatch_size": 2,
        "effective_batch_size": 2 * args.pairs_per_split,
        "pairs_per_split": args.pairs_per_split,
        "splits": args.splits,
        "local_scales": args.local_scales,
        "selection": {
            "policy": (
                "first 2N samples per split in the pre-prediction data manifest, "
                "paired consecutively without shuffling"
            ),
            "batches": [
                {
                    "batch_id": f"{split}_{pair_index:02d}",
                    "split": split,
                    "sample_ids": [sample.sample_id for sample in pair],
                }
                for split, pair_index, pair in microbatches
            ],
        },
        "gradient_target": (
            "All 50.7M trainable SSR parameters, with Base/2D Head frozen and "
            "detached; objectives are summed over K=1..3 exactly as the "
            "refined training term."
        ),
        "module_modes": {
            "base": "eval and requires_grad=False",
            "ssr": "train with BatchNorm using current two-image batch statistics",
            "batch_norm_buffers": (
                "restored to checkpoint state before every batch and after audit"
            ),
        },
        "optimizer_updates": 0,
        "gradient_accumulation": (
            "Effective-batch tables are the exact vector average of all "
            "two-image microbatch gradients in each split."
        ),
        "edge_scaling": {
            "checkpoint_normalizer": "max(H,W)",
            "paper_normalizer": "min(H,W)",
            "checkpoint_edge_to_paper_edge": edge_checkpoint_scale,
        },
        "parameter_count": sum(
            parameter.numel() for _, parameter in named_parameters
        ),
        "parameter_tensor_count": len(named_parameters),
        "objective_aggregates": objective_aggregates,
        "cosine_aggregates": cosine_aggregates,
        "group_aggregates": group_aggregates,
        "effective_objectives": effective_objective_rows,
        "effective_cosines": effective_cosine_rows,
        "effective_groups": effective_group_rows,
        "cross_split_cosines": cross_split_rows,
        "effective_scope_cosines": effective_scope_cosine_rows,
        "peak_cuda_memory_bytes": (
            {
                "allocated": int(torch.cuda.max_memory_allocated()),
                "reserved": int(torch.cuda.max_memory_reserved()),
            }
            if device.type == "cuda"
            else None
        ),
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_write_json(output / "report.json", report)
    return report


def main() -> None:
    print(json.dumps(run(parse_args()), ensure_ascii=False))


if __name__ == "__main__":
    main()
