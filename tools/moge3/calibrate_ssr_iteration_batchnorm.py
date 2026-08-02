from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Sequence

import torch

from moge.model.ssr import (
    capture_batch_norm_running_state,
    factorize_points,
    load_batch_norm_running_state,
)
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.calibrate_ssr_batchnorm import (
    batch_norm_summary,
    batch_slices,
    bn_buffer_keys,
    collect_base_batch,
    reset_for_cumulative_calibration,
    restore_momenta,
    ssr_batch_norms,
    verify_only_allowed_state_changed,
)
from tools.moge3.scan_checkpoint_residuals import sha256_file, write_json_atomic


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calibrate an independent SSR BatchNorm population state for each "
            "refinement iteration while keeping all learned weights fixed."
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
    parser.add_argument("--refinement-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--seed", type=int, default=211)
    return parser.parse_args()


def update_factorized(
    factorized: torch.Tensor,
    residual: torch.Tensor,
) -> torch.Tensor:
    return torch.cat(
        (
            factorized[..., :2],
            factorized[..., 2:3] + residual[..., None],
        ),
        dim=-1,
    )


@torch.no_grad()
def calibrate_iteration_states(
    model: Any,
    samples: Sequence[Any],
    *,
    batch_size: int,
    device: torch.device,
    num_tokens: int,
    refinement_steps: int,
) -> tuple[
    list[dict[str, dict[str, torch.Tensor]]],
    list[dict[str, Any]],
]:
    if not samples:
        raise ValueError("Iteration calibration requires training samples")
    states: list[dict[str, dict[str, torch.Tensor]]] = []
    reports: list[dict[str, Any]] = []
    modules = list(ssr_batch_norms(model).values())
    ranges = batch_slices(len(samples), batch_size)
    model.eval()
    for target_iteration in range(refinement_steps):
        original_momenta = reset_for_cumulative_calibration(modules)
        accumulator = capture_batch_norm_running_state(model.ssr)
        maxima = []
        started = time.perf_counter()
        try:
            for batch_index, (start, stop) in enumerate(ranges, start=1):
                base_points, visual_features = collect_base_batch(
                    model,
                    samples[start:stop],
                    device=device,
                    num_tokens=num_tokens,
                )
                factorized = factorize_points(base_points.float())
                for prior_state in states:
                    model.ssr.eval()
                    load_batch_norm_running_state(model.ssr, prior_state)
                    residual, _ = model.ssr(
                        factorized,
                        visual_features.float(),
                    )
                    factorized = update_factorized(factorized, residual)

                load_batch_norm_running_state(model.ssr, accumulator)
                model.ssr.train()
                residual, _ = model.ssr(
                    factorized,
                    visual_features.float(),
                )
                maxima.append(float(residual.detach().abs().amax().item()))
                accumulator = capture_batch_norm_running_state(model.ssr)
                print(
                    json.dumps(
                        {
                            "event": "iteration_calibration_batch",
                            "target_iteration": target_iteration + 1,
                            "batch_index": batch_index,
                            "batch_count": len(ranges),
                            "sample_count": stop - start,
                            "maximum_residual": maxima[-1],
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                del base_points, visual_features, factorized, residual
        finally:
            restore_momenta(modules, original_momenta)
            model.ssr.eval()
        load_batch_norm_running_state(model.ssr, accumulator)
        states.append(accumulator)
        reports.append(
            {
                "iteration": target_iteration + 1,
                "sample_count": len(samples),
                "batch_size": batch_size,
                "forward_batch_count": len(ranges),
                "maximum_calibration_residual": max(maxima),
                "elapsed_seconds": time.perf_counter() - started,
                "batch_norm": batch_norm_summary(model),
            }
        )
    return states, reports


def run(args: argparse.Namespace) -> dict[str, Any]:
    from moge.model.v3 import MoGeModel
    from moge.scripts.train_hypersim_joint_v3 import load_raw_samples

    if not 1 <= args.refinement_steps <= 7:
        raise ValueError("--refinement-steps must be in [1, 7]")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
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
    model.load_state_dict(reference_state, strict=True)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    iteration_states, iteration_reports = calibrate_iteration_states(
        model,
        samples,
        batch_size=args.batch_size,
        device=device,
        num_tokens=args.num_tokens,
        refinement_steps=args.refinement_steps,
    )
    state_audit = verify_only_allowed_state_changed(
        reference_state,
        model.state_dict(),
        bn_buffer_keys(model),
    )
    statistics_path = output / "iteration_statistics.pt"
    temporary = statistics_path.with_suffix(".pt.incomplete")
    torch.save(
        {
            "format_version": 1,
            "source_checkpoint_sha256": sha256_file(checkpoint_path),
            "source_checkpoint_step": int(checkpoint["step"]),
            "batch_size": args.batch_size,
            "sample_ids": [sample.sample_id for sample in samples],
            "iteration_states": iteration_states,
        },
        temporary,
    )
    os.replace(temporary, statistics_path)
    report = {
        "status": "complete",
        "source_checkpoint": str(checkpoint_path),
        "source_checkpoint_sha256": sha256_file(checkpoint_path),
        "checkpoint_step": int(checkpoint["step"]),
        "pretrained": str(pretrained),
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "refinement_steps": args.refinement_steps,
        "calibration_batch_size": args.batch_size,
        "calibration_sample_count": len(samples),
        "calibration_sample_ids": [sample.sample_id for sample in samples],
        "iterations": iteration_reports,
        "state_audit": state_audit,
        "peak_memory_mib": (
            float(torch.cuda.max_memory_allocated(device) / 2**20)
            if device.type == "cuda"
            else 0.0
        ),
        "iteration_statistics": str(statistics_path),
        "iteration_statistics_sha256": sha256_file(statistics_path),
        "iteration_statistics_bytes": statistics_path.stat().st_size,
        "elapsed_seconds": time.perf_counter() - started,
    }
    write_json_atomic(output / "report.json", report)
    return report


def main() -> None:
    report = run(parse_args())
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
