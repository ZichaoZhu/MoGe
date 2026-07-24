from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Tuple

import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs

from moge.model.ssr import SelfGuidedSparseRefiner
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


class IterativeSSRSmokeModel(torch.nn.Module):
    """Keep all shared-weight iterations inside one DDP forward."""

    def __init__(self, refiner: SelfGuidedSparseRefiner):
        super().__init__()
        self.refiner = refiner

    def forward(
        self,
        initial_q: torch.Tensor,
        visual: torch.Tensor,
        refinement_steps: int,
    ):
        current = initial_q
        log_depths = []
        stats = None
        for _ in range(refinement_steps):
            residual, stats = self.refiner(current, visual)
            current = torch.cat(
                (current[..., :2], current[..., 2:3] + residual[..., None]), dim=-1
            )
            log_depths.append(current[..., 2])
        return torch.stack(log_depths), stats


def synthetic_geometry(
    batch_size: int,
    height: int,
    width: int,
    visual_dim: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = torch.meshgrid(
        torch.linspace(-1.0, 1.0, height, device=device),
        torch.linspace(-1.0, 1.0, width, device=device),
        indexing="ij",
    )
    target_depth = 2.0 + 0.15 * rows + 0.1 * cols
    target_depth = target_depth + 1.2 * (cols > 0.15)
    thin_structure = (cols > -0.48) & (cols < -0.40) & (rows.abs() < 0.8)
    target_depth = torch.where(thin_structure, target_depth * 0.55, target_depth)

    distortion = 0.18 * torch.exp(-((cols - 0.15) / 0.14).square())
    distortion = distortion - 0.12 * torch.exp(-((cols + 0.44) / 0.08).square())
    initial_depth = target_depth * distortion.exp()

    uv = torch.stack((cols, rows), dim=-1)
    target_q = torch.cat((uv, target_depth.log()[..., None]), dim=-1)
    initial_q = torch.cat((uv, initial_depth.log()[..., None]), dim=-1)
    initial_q = initial_q.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()
    target_q = target_q.unsqueeze(0).expand(batch_size, -1, -1, -1).contiguous()

    visual = torch.stack(
        (
            target_depth.log(),
            initial_depth.log(),
            cols,
            rows,
            (cols > 0.15).float(),
            thin_structure.float(),
            torch.sin(torch.pi * cols),
            torch.cos(torch.pi * rows),
        ),
        dim=0,
    )
    if visual_dim < visual.shape[0]:
        visual = visual[:visual_dim]
    elif visual_dim > visual.shape[0]:
        padding = visual.new_zeros((visual_dim - visual.shape[0], height, width))
        visual = torch.cat((visual, padding), dim=0)
    visual = F.interpolate(
        visual[None],
        size=(max(1, height // 8), max(1, width // 8)),
        mode="bilinear",
        align_corners=False,
    )
    visual = visual.expand(batch_size, -1, -1, -1).contiguous()
    return initial_q, target_q, visual


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MoGe-3 sparse refiner synthetic smoke test")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--backend", choices=("spconv", "reference"), default="spconv")
    parser.add_argument("--height", type=int, default=64)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--visual-dim", type=int, default=8)
    parser.add_argument("--voxel-resolution", type=float, default=200.0)
    parser.add_argument("--channels", type=int, nargs="+", default=[8, 16, 32, 64, 128])
    parser.add_argument("--visual-channels", type=int, default=32)
    parser.add_argument("--blocks-per-level", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--ddp-backend", choices=("nccl", "gloo"), default="nccl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    accelerator = Accelerator(
        mixed_precision="no",
        kwargs_handlers=[InitProcessGroupKwargs(backend=args.ddp_backend)],
    )
    torch.manual_seed(17 + accelerator.process_index)

    model = IterativeSSRSmokeModel(
        SelfGuidedSparseRefiner(
            visual_dim=args.visual_dim,
            voxel_resolution=args.voxel_resolution,
            channels=args.channels,
            visual_channels=args.visual_channels,
            blocks_per_level=args.blocks_per_level,
            backend=args.backend,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-2)
    model, optimizer = accelerator.prepare(model, optimizer)

    checkpoint = output / "checkpoint"
    if args.resume and checkpoint.exists():
        accelerator.load_state(str(checkpoint))

    initial_q, target_q, visual = synthetic_geometry(
        args.batch_size,
        args.height,
        args.width,
        args.visual_dim,
        accelerator.device,
    )
    if accelerator.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(accelerator.device)

    with torch.no_grad():
        zero_log_depths, zero_stats = model(initial_q, visual, 1)
        identity_error = (zero_log_depths[0] - initial_q[..., 2]).abs().max()
    accelerator.wait_for_everyone()

    losses = []
    start = time.perf_counter()
    last_stats = zero_stats
    for _ in range(args.steps):
        log_depths, last_stats = model(initial_q, visual, args.refinement_steps)
        loss = F.mse_loss(
            log_depths,
            target_q[..., 2].unsqueeze(0).expand_as(log_depths),
        )
        accelerator.backward(loss)
        grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError("Non-finite SSR gradient in smoke test")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        losses.append(float(accelerator.gather(loss.detach()[None]).mean().item()))
    accelerator.wait_for_everyone()
    elapsed = time.perf_counter() - start
    accelerator.save_state(str(checkpoint))

    peak_memory = (
        torch.cuda.max_memory_allocated(accelerator.device)
        if accelerator.device.type == "cuda"
        else 0
    )
    gathered_memory = accelerator.gather(
        torch.tensor([peak_memory], device=accelerator.device, dtype=torch.long)
    )
    if accelerator.is_main_process:
        report = {
            "backend": args.backend,
            "ddp_backend": args.ddp_backend,
            "world_size": accelerator.num_processes,
            "shape": [args.batch_size, args.height, args.width],
            "channels": args.channels,
            "refinement_steps": args.refinement_steps,
            "optimization_steps": args.steps,
            "identity_max_error": float(identity_error.item()),
            "initial_loss": losses[0] if losses else None,
            "final_loss": losses[-1] if losses else None,
            "loss_decreased": bool(not losses or losses[-1] < losses[0]),
            "elapsed_seconds": elapsed,
            "milliseconds_per_step": 1_000.0 * elapsed / max(1, args.steps),
            "peak_memory_bytes_per_rank": gathered_memory.cpu().tolist(),
            "active_voxels": int(last_stats["active_voxels"].item()),
            "active_voxels_per_level": last_stats["active_voxels_per_level"],
            "depth_span": last_stats["depth_span"].detach().cpu().tolist(),
        }
        (output / "report.json").write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
