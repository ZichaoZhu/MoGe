from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import torch

from moge.model.v3 import MoGeModel
from moge.scripts.overfit_hypersim_v3 import geometry_loss, moving_average
from moge.scripts.train_hypersim_smallset_v3 import (
    CachedSample,
    aggregate_metrics,
    cache_base_predictions,
    evaluate,
    refine_cached,
    save_qualitative,
    zero_initialization_error,
)
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


SPLITS = ("train", "val", "test")
METRIC_KEYS = (
    "point_rel",
    "depth_rel",
    "depth_delta_1.01",
    "depth_delta_1.25",
    "boundary_f1",
)


def configure_ssr_only(model: torch.nn.Module) -> Dict[str, int]:
    """Freeze the complete base model and leave only SSR trainable."""
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.ssr.parameters():
        parameter.requires_grad_(True)

    ssr_ids = {id(parameter) for parameter in model.ssr.parameters()}
    trainable = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError("SSR-only guard found no trainable parameters")
    if {id(parameter) for _, parameter in trainable} != ssr_ids:
        unexpected = [
            name for name, parameter in trainable if id(parameter) not in ssr_ids
        ]
        raise RuntimeError(
            f"Non-SSR parameters remain trainable: {unexpected[:8]}"
        )
    return {
        "model_parameter_tensors": sum(1 for _ in model.parameters()),
        "base_parameter_tensors": sum(
            id(parameter) not in ssr_ids for parameter in model.parameters()
        ),
        "ssr_parameter_tensors": len(ssr_ids),
        "trainable_parameter_tensors": len(trainable),
        "base_trainable_parameter_tensors": 0,
    }


def base_state_digest(model: torch.nn.Module) -> str:
    """Hash every base-model parameter and buffer, excluding SSR state."""
    digest = hashlib.sha256()
    base_tensors = [
        (name, tensor)
        for name, tensor in model.state_dict().items()
        if not name.startswith("ssr.")
    ]
    if not base_tensors:
        raise RuntimeError("Base-state digest found no non-SSR tensors")
    for name, tensor in base_tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def verify_cached_base_detached(samples: Sequence[CachedSample]) -> None:
    if not samples:
        raise RuntimeError("SSR-only guard received an empty feature cache")
    for sample in samples:
        for field in ("base_points", "visual_features"):
            tensor = getattr(sample, field)
            if tensor.requires_grad or tensor.grad_fn is not None:
                raise RuntimeError(
                    f"Cached {field} retains autograd state for {sample.sample_id}"
                )


def verify_optimizer_ssr_only(
    optimizer: torch.optim.Optimizer,
    refiner: torch.nn.Module,
) -> int:
    optimizer_parameters = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
    ]
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    ssr_ids = {id(parameter) for parameter in refiner.parameters()}
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("SSR optimizer contains duplicate parameters")
    if set(optimizer_ids) != ssr_ids:
        raise RuntimeError("Optimizer parameter set is not exactly the SSR set")
    if any(not parameter.requires_grad for parameter in optimizer_parameters):
        raise RuntimeError("SSR optimizer contains a frozen parameter")
    return len(optimizer_parameters)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the frozen-base MoGe-3 SSR on disjoint Hypersim "
            "train/validation/test scene groups"
        )
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--reset-optimizer-on-resume",
        action="store_true",
        help="Load SSR weights and RNG state but start a fresh AdamW optimizer",
    )
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--pretrained", default="Ruicheng/moge-2-vitl-normal")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--num-tokens", type=int, default=2500)
    parser.add_argument("--refinement-steps", type=int, default=3)
    parser.add_argument("--evaluation-steps", type=int, nargs="+", default=[0, 1, 3, 5])
    parser.add_argument("--steps", type=int, default=8000)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument(
        "--microbatch-size",
        type=int,
        help="Samples per autograd graph; defaults to --batch-size",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--global-weight", type=float, default=1.0)
    parser.add_argument("--local-weight", type=float, default=1.0)
    parser.add_argument("--edge-weight", type=float, default=1.0)
    parser.add_argument("--local-scales", type=int, nargs="*", default=[4, 16, 64])
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--periodic-train-samples", type=int, default=64)
    parser.add_argument("--loss-smoothing-window", type=int, default=100)
    parser.add_argument("--boundary-threshold", type=float, default=0.03)
    parser.add_argument(
        "--max-samples-per-split",
        type=int,
        help="Smoke-test limit applied before base-feature caching",
    )
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument(
        "--debug-step",
        type=int,
        help="Enable autograd anomaly tracing and print the sampled IDs at one step",
    )
    parser.add_argument(
        "--debug-sample-indices",
        type=int,
        nargs=2,
        metavar=("INDEX_0", "INDEX_1"),
        help="Override the sampled batch only at --debug-step (diagnostics only)",
    )
    parser.add_argument(
        "--detect-anomaly",
        action="store_true",
        help="Run every backward pass with PyTorch NaN anomaly detection",
    )
    parser.add_argument(
        "--diagnose-loss-gradients",
        action="store_true",
        help="Check point-output gradients and isolate the first non-finite loss term",
    )
    return parser.parse_args()


def stratified_subset(
    samples: Sequence[CachedSample],
    maximum: int,
) -> List[CachedSample]:
    if maximum >= len(samples):
        return list(samples)
    by_scene: Dict[str, List[CachedSample]] = defaultdict(list)
    for sample in samples:
        by_scene[sample.scene].append(sample)
    for values in by_scene.values():
        values.sort(key=lambda sample: sample.frame)
    scenes = sorted(by_scene)
    quota, remainder = divmod(maximum, len(scenes))
    selected: List[CachedSample] = []
    for scene_index, scene in enumerate(scenes):
        values = by_scene[scene]
        count = min(len(values), quota + int(scene_index < remainder))
        indices = np.linspace(0, len(values) - 1, count, dtype=np.int64)
        selected.extend(values[int(index)] for index in indices)
    return selected


def save_csv(path: Path, records: Iterable[Dict[str, object]]) -> None:
    rows = list(records)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def save_checkpoint(
    path: Path,
    *,
    refiner,
    optimizer: torch.optim.Optimizer,
    step: int,
    args: argparse.Namespace,
    selection_score: float,
    cpu_generator: torch.Generator,
    loss_generator: torch.Generator,
) -> None:
    torch.save(
        {
            "ssr": refiner.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "pretrained": args.pretrained,
            "args": vars(args),
            "cpu_generator_state": cpu_generator.get_state(),
            "loss_generator_state": loss_generator.get_state(),
            "selection": {
                "metric": "validation point_rel",
                "mode": "min",
                "score": selection_score,
            },
        },
        path,
    )


def load_numeric_csv(path: Path) -> List[Dict[str, object]]:
    if not path.is_file():
        return []
    rows: List[Dict[str, object]] = []
    with path.open(newline="", encoding="utf-8") as file:
        for source in csv.DictReader(file):
            rows.append(
                {
                    key: int(value) if key == "step" else float(value)
                    for key, value in source.items()
                }
            )
    return rows


def advance_generators(
    *,
    steps: int,
    num_train_samples: int,
    batch_size: int,
    refinement_steps: int,
    local_scales: Sequence[int],
    device: torch.device,
    cpu_generator: torch.Generator,
    loss_generator: torch.Generator,
) -> None:
    """Reconstruct explicit generator states for legacy checkpoints."""
    for _ in range(steps):
        torch.randint(
            num_train_samples,
            (batch_size,),
            generator=cpu_generator,
        )
        for _ in range(refinement_steps):
            torch.rand(batch_size, device=device, generator=loss_generator)
            torch.randn(
                batch_size,
                3,
                3,
                device=device,
                dtype=torch.float32,
                generator=loss_generator,
            )
            for _ in local_scales:
                torch.rand(
                    batch_size,
                    device=device,
                    dtype=torch.float32,
                    generator=loss_generator,
                )


def save_training_plot(
    output: Path,
    losses: Sequence[float],
    eval_history: Sequence[Dict[str, float]],
    baseline: Dict[str, Dict[str, float]],
    smoothing_window: int,
    training_refinement_steps: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    values = np.asarray(losses, dtype=np.float64)
    steps = np.arange(1, len(values) + 1)
    window = min(smoothing_window, max(1, len(values)))
    smooth = moving_average(values, window)
    eval_steps = [record["step"] for record in eval_history]

    plt.style.use("seaborn-v0_8-whitegrid")
    figure, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes[0, 0].plot(
        steps,
        values,
        color="#5B8FF9",
        alpha=0.16,
        linewidth=0.6,
        label="Per-step loss",
    )
    axes[0, 0].plot(
        steps,
        smooth,
        color="#1D4ED8",
        linewidth=2.0,
        label=f"{window}-step mean",
    )
    axes[0, 0].set_title("SSR geometry loss")
    axes[0, 0].set_xlabel("Optimization step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()

    panels = (
        ("point_rel", "Aligned point Rel", True),
        ("depth_rel", "Aligned depth Rel", True),
        ("boundary_f1", "Depth boundary F1", False),
    )
    for axis, (metric, title, lower_is_better) in zip(
        (axes[0, 1], axes[1, 0], axes[1, 1]),
        panels,
    ):
        for split, color in (("train", "#1D4ED8"), ("val", "#D94841")):
            axis.plot(
                eval_steps,
                [record[f"{split}/{metric}"] for record in eval_history],
                color=color,
                marker="o",
                linewidth=1.8,
                markersize=3.5,
                label=f"{split} K={training_refinement_steps}",
            )
            axis.axhline(
                baseline[split][metric],
                color=color,
                linestyle="--",
                alpha=0.55,
                label=f"{split} K=0",
            )
        direction = "lower is better" if lower_is_better else "higher is better"
        axis.set_title(f"{title} ({direction})")
        axis.set_xlabel("Optimization step")
        axis.legend(fontsize=8)

    figure.suptitle("MoGe-3 Hypersim multi-scene SSR generalization")
    figure.tight_layout()
    figure.savefig(output / "training_curves.png", dpi=180)
    figure.savefig(output / "training_curves.pdf")
    plt.close(figure)


def validate_refinement_configuration(
    refinement_steps: int,
    evaluation_steps: Sequence[int],
) -> None:
    if not 1 <= refinement_steps <= 7:
        raise ValueError("Training refinement steps must be in [1, 7]")
    if sorted(set(evaluation_steps)) != list(evaluation_steps):
        raise ValueError("Evaluation steps must be sorted and unique")
    if not evaluation_steps or evaluation_steps[0] != 0 or max(evaluation_steps) > 7:
        raise ValueError("Evaluation steps must start at 0 and end no later than 7")
    if refinement_steps not in evaluation_steps:
        raise ValueError("Evaluation steps must include the training refinement step")
    if 3 not in evaluation_steps:
        raise ValueError("Evaluation steps must include K=3 for common diagnostics")


def changes_from_k0(
    metrics_by_k: Dict[str, Dict[str, Dict[str, float]]],
    step: int,
) -> Dict[str, Dict[str, float]]:
    changes: Dict[str, Dict[str, float]] = {}
    for split in SPLITS:
        before = metrics_by_k["0"][split]
        after = metrics_by_k[str(step)][split]
        changes[split] = {
            "point_rel_reduction": 1.0 - after["point_rel"] / before["point_rel"],
            "depth_rel_reduction": 1.0 - after["depth_rel"] / before["depth_rel"],
            "depth_delta_1.01_change": (
                after["depth_delta_1.01"] - before["depth_delta_1.01"]
            ),
            "depth_delta_1.25_change": (
                after["depth_delta_1.25"] - before["depth_delta_1.25"]
            ),
            "boundary_f1_change": after["boundary_f1"] - before["boundary_f1"],
        }
    return changes


def aggregate_by_split(records: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    return {
        split: aggregate_metrics(
            [record for record in records if record["split"] == split]
        )
        for split in SPLITS
    }


def main() -> None:
    args = parse_args()
    if args.steps <= 0 or args.batch_size <= 0:
        raise ValueError("Steps and batch size must be positive")
    microbatch_size = args.microbatch_size or args.batch_size
    if microbatch_size <= 0 or microbatch_size > args.batch_size:
        raise ValueError("Microbatch size must be in [1, batch size]")
    if args.eval_every <= 0 or args.periodic_train_samples <= 0:
        raise ValueError("Evaluation intervals and sample counts must be positive")
    validate_refinement_configuration(
        args.refinement_steps,
        args.evaluation_steps,
    )

    data_dir = assert_safe_path(args.data, safe_root=args.safe_root, must_exist=True)
    output = assert_safe_path(args.output, safe_root=args.safe_root, writable=True)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = assert_safe_path(
        data_dir / "manifest.json",
        safe_root=args.safe_root,
        must_exist=True,
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("format") != "moge3-hypersim-generalization-v1":
        raise ValueError(f"Unsupported manifest: {manifest.get('format')}")
    if tuple(manifest.get("counts", {}).keys()) != SPLITS:
        raise ValueError(f"Manifest must contain {SPLITS}")
    if args.max_samples_per_split is not None:
        if args.max_samples_per_split <= 0:
            raise ValueError("--max-samples-per-split must be positive")
        limited_samples = []
        for split in SPLITS:
            candidates = [
                sample for sample in manifest["samples"] if sample["split"] == split
            ]
            limited_samples.extend(candidates[: args.max_samples_per_split])
        manifest = {
            **manifest,
            "samples": limited_samples,
            "counts": {
                split: sum(
                    sample["split"] == split for sample in limited_samples
                )
                for split in SPLITS
            },
        }

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.detect_anomaly:
        torch.autograd.set_detect_anomaly(True, check_nan=True)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats(device)

    run_start = time.perf_counter()
    model = MoGeModel.from_pretrained(args.pretrained).to(device).eval()
    ssr_only_guard = configure_ssr_only(model)
    base_digest_before_cache = base_state_digest(model)
    cache_start = time.perf_counter()
    samples = cache_base_predictions(
        data_dir,
        manifest,
        model,
        height=args.height,
        width=args.width,
        num_tokens=args.num_tokens,
        device=device,
        safe_root=args.safe_root,
    )
    cache_elapsed = time.perf_counter() - cache_start
    verify_cached_base_detached(samples)
    base_digest_after_cache = base_state_digest(model)
    if base_digest_after_cache != base_digest_before_cache:
        raise RuntimeError("Base-model state changed while caching frozen features")
    ssr_only_guard.update(
        {
            "cache_tensors_detached": True,
            "base_digest_before_cache": base_digest_before_cache,
            "base_digest_after_cache": base_digest_after_cache,
            "base_digest_unchanged": True,
        }
    )
    split_samples = {
        split: [sample for sample in samples if sample.split == split]
        for split in SPLITS
    }
    if any(not split_samples[split] for split in SPLITS):
        raise ValueError("Train, validation, and test must all be non-empty")

    refiner = model.ssr
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    refiner.to(device).train()
    optimizer = torch.optim.AdamW(
        refiner.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    ssr_only_guard.update(
        {
            "optimizer_parameter_tensors": verify_optimizer_ssr_only(
                optimizer,
                refiner,
            ),
            "base_released_before_optimizer_step": True,
            "checkpoint_contains_ssr_only": True,
        }
    )
    cpu_generator = torch.Generator().manual_seed(args.seed + 1)
    loss_generator = torch.Generator(device=device).manual_seed(args.seed + 2)
    start_step = 0
    resume_path = None
    rng_restored_exactly = True
    resume_checkpoint = None
    if args.resume is not None:
        resume_path = assert_safe_path(
            args.resume,
            safe_root=args.safe_root,
            must_exist=True,
        )
        resume_checkpoint = torch.load(
            resume_path,
            # Explicit generator states are serialized as CPU ByteTensors.
            # Loading the whole checkpoint directly onto CUDA converts the
            # CPU generator state and makes Generator.set_state reject it.
            map_location="cpu",
            weights_only=False,
        )
        if resume_checkpoint.get("pretrained") != args.pretrained:
            raise ValueError("Resume checkpoint uses a different base model")
        saved_args = resume_checkpoint.get("args", {})
        for key in (
            "height",
            "width",
            "num_tokens",
            "refinement_steps",
            "batch_size",
            "weight_decay",
        ):
            if saved_args.get(key) != getattr(args, key):
                raise ValueError(
                    f"Resume checkpoint disagrees on {key}: "
                    f"{saved_args.get(key)} != {getattr(args, key)}"
                )
        saved_learning_rate = saved_args.get("learning_rate")
        if (
            saved_learning_rate != args.learning_rate
            and not args.reset_optimizer_on_resume
        ):
            raise ValueError(
                "Resume checkpoint uses a different learning rate; pass "
                "--reset-optimizer-on-resume to start a fresh optimizer"
            )
        refiner.load_state_dict(resume_checkpoint["ssr"], strict=True)
        if not args.reset_optimizer_on_resume:
            optimizer.load_state_dict(resume_checkpoint["optimizer"])
        start_step = int(resume_checkpoint["step"])
        if args.steps <= start_step:
            raise ValueError(
                f"--steps must exceed resumed step {start_step}, got {args.steps}"
            )
        if (
            "cpu_generator_state" in resume_checkpoint
            and "loss_generator_state" in resume_checkpoint
        ):
            cpu_generator.set_state(resume_checkpoint["cpu_generator_state"])
            loss_generator.set_state(resume_checkpoint["loss_generator_state"])
        else:
            rng_restored_exactly = False
            advance_generators(
                steps=start_step,
                num_train_samples=len(split_samples["train"]),
                batch_size=args.batch_size,
                refinement_steps=args.refinement_steps,
                local_scales=args.local_scales,
                device=device,
                cpu_generator=cpu_generator,
                loss_generator=loss_generator,
            )

    periodic_train = stratified_subset(
        split_samples["train"],
        args.periodic_train_samples,
    )
    periodic_samples = periodic_train + split_samples["val"]
    initial_records, _ = evaluate(
        refiner,
        periodic_samples,
        device=device,
        refinement_steps=1,
        boundary_threshold=args.boundary_threshold,
        use_refiner=False,
        batch_size=args.batch_size,
    )
    initial_refined_records, _ = evaluate(
        refiner,
        periodic_samples,
        device=device,
        refinement_steps=args.refinement_steps,
        boundary_threshold=args.boundary_threshold,
        use_refiner=True,
        batch_size=args.batch_size,
    )
    periodic_baseline = {
        split: aggregate_metrics(
            [record for record in initial_records if record["split"] == split]
        )
        for split in ("train", "val")
    }
    initial_refined = {
        split: aggregate_metrics(
            [
                record
                for record in initial_refined_records
                if record["split"] == split
            ]
        )
        for split in ("train", "val")
    }
    identity_error = (
        zero_initialization_error(
            refiner,
            periodic_samples[: min(16, len(periodic_samples))],
            device=device,
            refinement_steps=args.refinement_steps,
            batch_size=args.batch_size,
        )
        if start_step == 0
        else None
    )

    if start_step == 0:
        best_step = 0
        best_score = initial_refined["val"]["point_rel"]
        save_checkpoint(
            output / "checkpoint.pt",
            refiner=refiner,
            optimizer=optimizer,
            step=0,
            args=args,
            selection_score=best_score,
            cpu_generator=cpu_generator,
            loss_generator=loss_generator,
        )
        eval_history: List[Dict[str, float]] = [
            {
                "step": 0,
                **{
                    f"{split}/{key}": initial_refined[split][key]
                    for split in ("train", "val")
                    for key in METRIC_KEYS
                },
            }
        ]
        training_records: List[Dict[str, object]] = []
    else:
        best_checkpoint_metadata = torch.load(
            output / "checkpoint.pt",
            map_location="cpu",
            weights_only=False,
        )
        best_step = int(best_checkpoint_metadata["step"])
        best_score = float(best_checkpoint_metadata["selection"]["score"])
        training_records = load_numeric_csv(output / "training_history.csv")
        eval_history = load_numeric_csv(output / "evaluation_history.csv")
        if (
            len(training_records) != start_step
            or not eval_history
            or int(eval_history[-1]["step"]) != start_step
        ):
            raise ValueError("Resume histories do not end at the checkpoint step")
    losses: List[float] = [
        float(record["loss"]) for record in training_records
    ]
    train_start = time.perf_counter()
    for step in range(start_step + 1, args.steps + 1):
        indices = torch.randint(
            len(split_samples["train"]),
            (args.batch_size,),
            generator=cpu_generator,
        ).tolist()
        if step == args.debug_step and args.debug_sample_indices is not None:
            if args.batch_size != len(args.debug_sample_indices):
                raise ValueError("--debug-sample-indices must match --batch-size")
            if not all(
                0 <= index < len(split_samples["train"])
                for index in args.debug_sample_indices
            ):
                raise ValueError("--debug-sample-indices contains an invalid index")
            indices = list(args.debug_sample_indices)
        batch = [split_samples["train"][index] for index in indices]

        optimizer.zero_grad(set_to_none=True)
        debug_this_step = step == args.debug_step
        loss_value = 0.0
        terms: Dict[str, float] = defaultdict(float)
        for microbatch_start in range(0, len(batch), microbatch_size):
            microbatch = batch[
                microbatch_start : microbatch_start + microbatch_size
            ]
            microbatch_weight = len(microbatch) / len(batch)
            base = torch.stack(
                [sample.base_points for sample in microbatch]
            ).to(device)
            visual = torch.stack(
                [sample.visual_features for sample in microbatch]
            ).to(device)
            gt = torch.stack([sample.gt_points for sample in microbatch]).to(device)
            sequence = refine_cached(
                refiner,
                base,
                visual,
                args.refinement_steps,
            )
            tensor_terms = {} if args.diagnose_loss_gradients else None
            microbatch_loss, microbatch_terms = geometry_loss(
                sequence,
                gt,
                global_weight=args.global_weight,
                local_weight=args.local_weight,
                edge_weight=args.edge_weight,
                local_scales=tuple(args.local_scales),
                generator=loss_generator,
                tensor_records=tensor_terms,
            )
            if not torch.isfinite(microbatch_loss):
                raise RuntimeError(
                    f"Non-finite loss at step {step}, "
                    f"microbatch {microbatch_start // microbatch_size}"
                )
            loss_value += microbatch_weight * float(microbatch_loss.detach())
            for key, value in microbatch_terms.items():
                terms[key] += microbatch_weight * value
            scaled_loss = microbatch_weight * microbatch_loss
            if args.diagnose_loss_gradients:
                point_gradients = torch.autograd.grad(
                    microbatch_loss,
                    sequence,
                    retain_graph=True,
                    allow_unused=True,
                )
                if any(
                    gradient is not None
                    and not torch.isfinite(gradient).all()
                    for gradient in point_gradients
                ):
                    component_status = {}
                    assert tensor_terms is not None
                    for name, tensor_term in tensor_terms.items():
                        sequence_index = int(name[1 : name.index("/")]) - 1
                        component_gradient = torch.autograd.grad(
                            tensor_term,
                            sequence[sequence_index],
                            retain_graph=True,
                            allow_unused=True,
                        )[0]
                        component_status[name] = {
                            "finite": bool(
                                component_gradient is None
                                or torch.isfinite(component_gradient).all()
                            ),
                            "max_abs": (
                                float(component_gradient.detach().abs().amax())
                                if component_gradient is not None
                                and torch.isfinite(component_gradient).all()
                                else None
                            ),
                        }
                    print(
                        json.dumps(
                            {
                                "event": "non_finite_point_gradient",
                                "step": step,
                                "microbatch": microbatch_start
                                // microbatch_size,
                                "sample_ids": [
                                    sample.sample_id for sample in microbatch
                                ],
                                "components": component_status,
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )
                    raise RuntimeError(
                        f"Non-finite point-output gradient at step {step}"
                    )
            if debug_this_step:
                print(
                    json.dumps(
                        {
                            "debug_step": step,
                            "microbatch": microbatch_start // microbatch_size,
                            "sample_indices": indices[
                                microbatch_start : microbatch_start
                                + microbatch_size
                            ],
                            "sample_ids": [
                                sample.sample_id for sample in microbatch
                            ],
                            "loss": float(microbatch_loss.detach()),
                            **microbatch_terms,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
            if args.detect_anomaly:
                scaled_loss.backward()
            elif debug_this_step:
                with torch.autograd.detect_anomaly(check_nan=True):
                    scaled_loss.backward()
            else:
                scaled_loss.backward()
            # Sparse graphs can have very different rulebooks from one sample
            # to the next, so release each microbatch before building another.
            del base, visual, gt, sequence, microbatch_loss, scaled_loss
        grad_norm = torch.nn.utils.clip_grad_norm_(refiner.parameters(), 1.0)
        if not torch.isfinite(grad_norm):
            raise RuntimeError(f"Non-finite gradient at step {step}")
        if debug_this_step:
            print(
                json.dumps(
                    {
                        "debug_step": step,
                        "gradient_is_finite": True,
                        "grad_norm": float(grad_norm),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
            return
        optimizer.step()

        losses.append(loss_value)
        training_records.append(
            {
                "step": step,
                "loss": loss_value,
                "grad_norm": float(grad_norm),
                **terms,
            }
        )

        if step % args.eval_every == 0 or step == args.steps:
            evaluation, _ = evaluate(
                refiner,
                periodic_samples,
                device=device,
                refinement_steps=args.refinement_steps,
                boundary_threshold=args.boundary_threshold,
                use_refiner=True,
                batch_size=args.batch_size,
            )
            aggregate = {
                split: aggregate_metrics(
                    [record for record in evaluation if record["split"] == split]
                )
                for split in ("train", "val")
            }
            eval_history.append(
                {
                    "step": step,
                    **{
                        f"{split}/{key}": aggregate[split][key]
                        for split in ("train", "val")
                        for key in METRIC_KEYS
                    },
                }
            )
            if aggregate["val"]["point_rel"] < best_score:
                best_score = aggregate["val"]["point_rel"]
                best_step = step
                save_checkpoint(
                    output / "checkpoint.pt",
                    refiner=refiner,
                    optimizer=optimizer,
                    step=step,
                    args=args,
                    selection_score=best_score,
                    cpu_generator=cpu_generator,
                    loss_generator=loss_generator,
                )
            save_checkpoint(
                output / "latest_checkpoint.pt",
                refiner=refiner,
                optimizer=optimizer,
                step=step,
                args=args,
                selection_score=aggregate["val"]["point_rel"],
                cpu_generator=cpu_generator,
                loss_generator=loss_generator,
            )
            save_csv(output / "training_history.csv", training_records)
            save_csv(output / "evaluation_history.csv", eval_history)

        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(
                json.dumps(
                    {
                        "step": step,
                        "loss": loss_value,
                        "grad_norm": float(grad_norm),
                        "best_step": best_step,
                        "best_val_point_rel": best_score,
                        "elapsed_seconds": time.perf_counter() - train_start,
                        **terms,
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )
    training_elapsed = time.perf_counter() - train_start
    best_checkpoint = torch.load(
        output / "checkpoint.pt",
        map_location=device,
        weights_only=False,
    )
    refiner.load_state_dict(best_checkpoint["ssr"], strict=True)
    best_step = int(best_checkpoint["step"])
    best_score = float(best_checkpoint["selection"]["score"])

    records_by_k: Dict[int, List[Dict[str, object]]] = {}
    predictions_by_k: Dict[int, Dict[str, torch.Tensor]] = {}
    metrics_by_k: Dict[str, Dict[str, Dict[str, float]]] = {}
    for refinement_step in args.evaluation_steps:
        records, predictions = evaluate(
            refiner,
            samples,
            device=device,
            refinement_steps=max(1, refinement_step),
            boundary_threshold=args.boundary_threshold,
            use_refiner=refinement_step > 0,
            batch_size=args.batch_size,
        )
        records_by_k[refinement_step] = records
        metrics_by_k[str(refinement_step)] = aggregate_by_split(records)
        if refinement_step in (0, 3):
            predictions_by_k[refinement_step] = predictions
        else:
            del predictions

    per_frame = []
    records_by_id = {
        step: {record["id"]: record for record in records}
        for step, records in records_by_k.items()
    }
    fractions: Dict[str, Dict[str, float]] = {}
    training_k_fractions: Dict[str, Dict[str, float]] = {}
    for split in SPLITS:
        split_rows = []
        for sample in split_samples[split]:
            row: Dict[str, object] = {
                "id": sample.sample_id,
                "split": split,
                "scene": sample.scene,
                "frame": sample.frame,
            }
            for refinement_step in args.evaluation_steps:
                source = records_by_id[refinement_step][sample.sample_id]
                for key in METRIC_KEYS:
                    row[f"k{refinement_step}_{key}"] = source[key]
            split_rows.append(row)
            per_frame.append(row)
        fractions[split] = {
            "point_rel_improved": float(
                np.mean(
                    [
                        row["k3_point_rel"] < row["k0_point_rel"]
                        for row in split_rows
                    ]
                )
            ),
            "depth_rel_improved": float(
                np.mean(
                    [
                        row["k3_depth_rel"] < row["k0_depth_rel"]
                        for row in split_rows
                    ]
                )
            ),
            "boundary_f1_improved": float(
                np.mean(
                    [
                        row["k3_boundary_f1"] > row["k0_boundary_f1"]
                        for row in split_rows
                    ]
                )
            ),
        }
        training_prefix = f"k{args.refinement_steps}"
        training_k_fractions[split] = {
            "point_rel_improved": float(
                np.mean(
                    [
                        row[f"{training_prefix}_point_rel"] < row["k0_point_rel"]
                        for row in split_rows
                    ]
                )
            ),
            "depth_rel_improved": float(
                np.mean(
                    [
                        row[f"{training_prefix}_depth_rel"] < row["k0_depth_rel"]
                        for row in split_rows
                    ]
                )
            ),
            "boundary_f1_improved": float(
                np.mean(
                    [
                        row[f"{training_prefix}_boundary_f1"]
                        > row["k0_boundary_f1"]
                        for row in split_rows
                    ]
                )
            ),
        }

    save_csv(output / "training_history.csv", training_records)
    save_csv(output / "evaluation_history.csv", eval_history)
    save_csv(output / "per_frame_metrics.csv", per_frame)
    save_training_plot(
        output,
        losses,
        eval_history,
        periodic_baseline,
        args.loss_smoothing_window,
        args.refinement_steps,
    )
    selected = [
        split_samples["train"][0],
        split_samples["train"][len(split_samples["train"]) // 2],
        split_samples["val"][0],
        split_samples["test"][0],
    ]
    save_qualitative(
        output,
        selected,
        predictions_by_k[0],
        predictions_by_k[3],
    )

    report = {
        "status": "complete",
        "experiment": "Hypersim multi-scene SSR generalization",
        "paper_alignment": {
            "training_mode": (
                "paper ablation-style refiner-only setting with frozen base model"
            ),
            "training_refinement_steps": args.refinement_steps,
            "evaluation_refinement_steps": args.evaluation_steps,
            "paper_main_evaluation_resolution": "approximately 840^2",
            "current_resolution": [args.height, args.width],
        },
        "counts": manifest["counts"],
        "scene_groups": manifest["scene_groups"],
        "shape": [args.height, args.width],
        "num_tokens": args.num_tokens,
        "spconv_algorithm": getattr(
            getattr(refiner.unet, "conv_algo", None),
            "name",
            None,
        ),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pretrained": args.pretrained,
        "base_frozen": True,
        "ssr_only_guard": ssr_only_guard,
        "normal_prediction": False,
        "optimization_steps": args.steps,
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "resumed_step": start_step,
        "resume_rng_restored_exactly": rng_restored_exactly,
        "resume_optimizer_reset": args.reset_optimizer_on_resume,
        "batch_size": args.batch_size,
        "microbatch_size": microbatch_size,
        "gradient_accumulation_steps": int(
            np.ceil(args.batch_size / microbatch_size)
        ),
        "learning_rate": args.learning_rate,
        "loss_weights": {
            "global": args.global_weight,
            "local": args.local_weight,
            "edge": args.edge_weight,
        },
        "best_checkpoint_step": best_step,
        "checkpoint_selection": {
            "split": "val",
            "metric": "point_rel",
            "mode": "min",
            "score": best_score,
            "test_evaluated_only_after_selection": True,
        },
        "identity_max_error": identity_error,
        "periodic_evaluation": {
            "train_samples": len(periodic_train),
            "val_samples": len(split_samples["val"]),
            "interval_steps": args.eval_every,
            "baseline": periodic_baseline,
            "initial_refined": initial_refined,
        },
        "metrics_by_k": metrics_by_k,
        "k3_changes_from_k0": changes_from_k0(metrics_by_k, 3),
        "training_k_changes_from_k0": changes_from_k0(
            metrics_by_k,
            args.refinement_steps,
        ),
        "fraction_improved_k3_vs_k0": fractions,
        "fraction_improved_at_training_k_vs_k0": training_k_fractions,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "cache_seconds": cache_elapsed,
        "training_seconds_including_periodic_eval": training_elapsed,
        "total_seconds": time.perf_counter() - run_start,
        "seconds_per_continuation_step_including_periodic_eval": (
            training_elapsed / (args.steps - start_step)
        ),
        "peak_memory_bytes": (
            torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0
        ),
        "boundary_definition": {
            "log_depth_jump_threshold": args.boundary_threshold,
            "matching_radius_pixels": 1,
        },
        "artifacts": {
            "best_checkpoint": "checkpoint.pt",
            "latest_checkpoint": "latest_checkpoint.pt",
            "training_history": "training_history.csv",
            "evaluation_history": "evaluation_history.csv",
            "per_frame_metrics": "per_frame_metrics.csv",
            "training_curves": "training_curves.png",
            "qualitative_comparison": "qualitative_comparison.png",
        },
    }
    (output / "report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
