from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from moge.model.v3 import MoGeModel
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load MoGe-2 into MoGe-3 and verify zero SSR")
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = assert_safe_path(args.report, safe_root=args.safe_root, writable=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model = MoGeModel.from_pretrained(args.pretrained).to(device).eval()
    torch.manual_seed(9)
    image = torch.rand(1, 3, args.height, args.width, device=device)
    torch.cuda.reset_peak_memory_stats(device)
    outputs = {}
    timings = {}
    for steps in (0, 1, 3, 5):
        torch.cuda.synchronize(device)
        start = time.perf_counter()
        with torch.inference_mode():
            outputs[steps] = model(
                image,
                num_tokens=args.num_tokens,
                num_refinement_steps=steps,
                return_intermediates=True,
            )
        torch.cuda.synchronize(device)
        timings[str(steps)] = time.perf_counter() - start

    errors = {
        str(steps): float(
            (
                outputs[steps]["points_sequence"][-1]
                - outputs[steps]["points_sequence"][0]
            )
            .abs()
            .max()
            .item()
        )
        for steps in (1, 3, 5)
    }
    residual_errors = {
        str(steps): max(
            (
                float(residual.abs().max().item())
                for residual in outputs[steps]["log_depth_residuals"]
            ),
            default=0.0,
        )
        for steps in (1, 3, 5)
    }
    inferred = model.infer(
        image[0],
        num_tokens=args.num_tokens,
        use_fp16=False,
        num_refinement_steps=0,
        apply_mask=False,
    )
    report = {
        "pretrained": args.pretrained,
        "device": str(device),
        "input_shape": list(image.shape),
        "num_tokens": args.num_tokens,
        "normal_prediction_present": "normal" in outputs[0],
        "k0_infer_keys": sorted(inferred),
        "k0_depth_finite": bool(torch.isfinite(inferred["depth"]).all()),
        "point_max_error_vs_k0": errors,
        "residual_max_abs": residual_errors,
        "strict_identity": all(value == 0.0 for value in errors.values()),
        "timings_seconds": timings,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "model_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "ssr_parameter_count": sum(parameter.numel() for parameter in model.ssr.parameters()),
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
