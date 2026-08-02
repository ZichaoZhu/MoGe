from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import nn

from moge.model.ssr import factorize_points
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.scan_checkpoint_residuals import sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Recalibrate only SSR BatchNorm running statistics without "
            "updating any learned MoGe-3 parameter."
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 8],
    )
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=191)
    return parser.parse_args()


def batch_slices(length: int, batch_size: int) -> list[tuple[int, int]]:
    if length <= 0 or batch_size <= 0:
        raise ValueError("Length and batch size must be positive")
    return [
        (start, min(start + batch_size, length))
        for start in range(0, length, batch_size)
    ]


def ssr_batch_norms(model: Any) -> dict[str, nn.BatchNorm1d]:
    result = {
        name: module
        for name, module in model.ssr.named_modules()
        if isinstance(module, nn.BatchNorm1d)
    }
    if not result:
        raise RuntimeError("SSR contains no BatchNorm1d modules to recalibrate")
    return result


def bn_buffer_keys(model: Any) -> set[str]:
    keys: set[str] = set()
    for name in ssr_batch_norms(model):
        prefix = f"ssr.{name}."
        keys.update(
            {
                prefix + "running_mean",
                prefix + "running_var",
                prefix + "num_batches_tracked",
            }
        )
    return keys


def reset_for_cumulative_calibration(
    modules: Iterable[nn.BatchNorm1d],
) -> dict[int, float | None]:
    original_momenta: dict[int, float | None] = {}
    for module in modules:
        original_momenta[id(module)] = module.momentum
        module.reset_running_stats()
        module.momentum = None
    return original_momenta


def restore_momenta(
    modules: Iterable[nn.BatchNorm1d],
    original_momenta: dict[int, float | None],
) -> None:
    for module in modules:
        module.momentum = original_momenta[id(module)]


def batch_norm_summary(model: Any) -> dict[str, dict[str, float | int]]:
    return {
        name: {
            "running_mean_l2": float(module.running_mean.float().norm().item()),
            "running_variance_mean": float(module.running_var.float().mean().item()),
            "running_variance_minimum": float(module.running_var.float().min().item()),
            "running_variance_maximum": float(module.running_var.float().max().item()),
            "num_batches_tracked": int(module.num_batches_tracked.item()),
        }
        for name, module in ssr_batch_norms(model).items()
    }


def verify_only_allowed_state_changed(
    reference: dict[str, torch.Tensor],
    candidate: dict[str, torch.Tensor],
    allowed_keys: set[str],
) -> dict[str, int]:
    if set(reference) != set(candidate):
        missing = sorted(set(reference) - set(candidate))
        added = sorted(set(candidate) - set(reference))
        raise RuntimeError(f"State keys changed; missing={missing}, added={added}")
    changed_allowed = 0
    compared_unchanged = 0
    for key, source in reference.items():
        target = candidate[key].detach().cpu()
        if key in allowed_keys:
            changed_allowed += int(not torch.equal(source, target))
            continue
        if not torch.equal(source, target):
            raise RuntimeError(f"Calibration modified forbidden state: {key}")
        compared_unchanged += 1
    return {
        "allowed_buffer_count": len(allowed_keys),
        "changed_allowed_buffer_count": changed_allowed,
        "bitwise_unchanged_state_count": compared_unchanged,
    }


@torch.no_grad()
def collect_base_batch(
    model: Any,
    samples: Sequence[Any],
    *,
    device: torch.device,
    num_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    points = []
    visual_features = []
    for sample in samples:
        output = model._forward_base(
            sample.image.unsqueeze(0).to(device),
            num_tokens,
        )
        points.append(output["points"])
        visual_features.append(output["_visual_features"])
    return torch.cat(points, dim=0), torch.cat(visual_features, dim=0)


@torch.no_grad()
def update_ssr_statistics(
    model: Any,
    base_points: torch.Tensor,
    visual_features: torch.Tensor,
    *,
    refinement_steps: int,
) -> float:
    factorized = factorize_points(base_points.float())
    maximum = 0.0
    for _ in range(refinement_steps):
        residual, _ = model.ssr(factorized, visual_features.float())
        maximum = max(maximum, float(residual.detach().abs().amax().item()))
        factorized = torch.cat(
            (
                factorized[..., :2],
                factorized[..., 2:3] + residual[..., None],
            ),
            dim=-1,
        )
    return maximum


def calibrate_arm(
    model: Any,
    samples: Sequence[Any],
    *,
    batch_size: int,
    device: torch.device,
    num_tokens: int,
    refinement_steps: int,
) -> dict[str, Any]:
    model.eval()
    modules = list(ssr_batch_norms(model).values())
    original_momenta = reset_for_cumulative_calibration(modules)
    model.ssr.train()
    maxima = []
    started = time.perf_counter()
    try:
        for batch_index, (start, stop) in enumerate(
            batch_slices(len(samples), batch_size),
            start=1,
        ):
            chunk = samples[start:stop]
            base_points, visual_features = collect_base_batch(
                model,
                chunk,
                device=device,
                num_tokens=num_tokens,
            )
            maximum = update_ssr_statistics(
                model,
                base_points,
                visual_features,
                refinement_steps=refinement_steps,
            )
            maxima.append(maximum)
            print(
                json.dumps(
                    {
                        "event": "calibration_batch",
                        "batch_size": batch_size,
                        "batch_index": batch_index,
                        "batch_count": len(batch_slices(len(samples), batch_size)),
                        "sample_count": len(chunk),
                        "maximum_residual": maximum,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            del base_points, visual_features
    finally:
        restore_momenta(modules, original_momenta)
        model.eval()
    return {
        "batch_size": batch_size,
        "sample_count": len(samples),
        "forward_batch_count": len(batch_slices(len(samples), batch_size)),
        "ssr_call_count": len(batch_slices(len(samples), batch_size))
        * refinement_steps,
        "maximum_train_mode_residual": max(maxima),
        "elapsed_seconds": time.perf_counter() - started,
        "batch_norm": batch_norm_summary(model),
    }


def save_calibrated_checkpoint(
    source_checkpoint: dict[str, Any],
    model: Any,
    path: Path,
    metadata: dict[str, Any],
) -> None:
    payload = copy.copy(source_checkpoint)
    payload["model"] = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }
    payload["batch_norm_calibration"] = metadata
    temporary = path.with_suffix(path.suffix + ".incomplete")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    from moge.model.v3 import MoGeModel
    from moge.scripts.train_hypersim_joint_v3 import load_raw_samples

    if not 1 <= args.refinement_steps <= 7:
        raise ValueError("--refinement-steps must be in [1, 7]")
    batch_sizes = list(dict.fromkeys(args.batch_sizes))
    if not batch_sizes or any(size <= 0 for size in batch_sizes):
        raise ValueError("--batch-sizes must contain positive unique values")
    data = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    checkpoint_path = assert_safe_path(
        args.checkpoint,
        safe_root=args.safe_root,
        must_exist=True,
    )
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
    samples = load_raw_samples(
        data,
        manifest,
        height=args.height,
        width=args.width,
        safe_root=args.safe_root,
        splits=("train",),
    )
    if args.max_train_samples is not None:
        if args.max_train_samples <= 0:
            raise ValueError("--max-train-samples must be positive")
        samples = samples[: args.max_train_samples]
    if not samples:
        raise RuntimeError("No training samples selected for calibration")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    pretrained = args.pretrained or checkpoint.get("pretrained")
    if not pretrained:
        raise ValueError("Checkpoint does not identify its pretrained base model")
    reference_state = {
        key: value.detach().cpu()
        for key, value in checkpoint["model"].items()
    }
    device = torch.device(args.device)
    model = MoGeModel.from_pretrained(str(pretrained)).to(device)
    allowed_keys = bn_buffer_keys(model)
    original_summary = None
    arms: dict[str, Any] = {}
    started = time.perf_counter()
    for batch_size in batch_sizes:
        model.load_state_dict(reference_state, strict=True)
        if original_summary is None:
            original_summary = batch_norm_summary(model)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        metadata = calibrate_arm(
            model,
            samples,
            batch_size=batch_size,
            device=device,
            num_tokens=args.num_tokens,
            refinement_steps=args.refinement_steps,
        )
        candidate_state = model.state_dict()
        metadata["state_audit"] = verify_only_allowed_state_changed(
            reference_state,
            candidate_state,
            allowed_keys,
        )
        metadata["peak_memory_mib"] = (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else 0.0
        )
        checkpoint_output = output / f"bn_recal_b{batch_size}.pt"
        metadata["source_checkpoint_sha256"] = sha256_file(checkpoint_path)
        save_calibrated_checkpoint(
            checkpoint,
            model,
            checkpoint_output,
            metadata,
        )
        metadata["checkpoint"] = str(checkpoint_output)
        metadata["checkpoint_sha256"] = sha256_file(checkpoint_output)
        metadata["checkpoint_bytes"] = checkpoint_output.stat().st_size
        arms[f"batch_{batch_size}"] = metadata
        write_json_atomic(output / f"batch_{batch_size}.json", metadata)

    report = {
        "status": "complete",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "pretrained": str(pretrained),
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "refinement_steps": args.refinement_steps,
        "calibration_sample_ids": [sample.sample_id for sample in samples],
        "calibration_sample_count": len(samples),
        "batch_sizes": batch_sizes,
        "original_batch_norm": original_summary,
        "arms": arms,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output / "report.json", report)
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
