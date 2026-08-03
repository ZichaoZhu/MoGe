from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from moge.train.stability_v3 import relative_window_improvement
from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


DATA_RELATIVES = (
    Path(
        "experiment/stage2_training_strategy/runs/"
        "exp11_hypersim_100_train_staged_joint_overfit/data"
    ),
    Path("experiment/exp11_hypersim_100_train_staged_joint_overfit/data"),
)
DEFAULT_EXPERIMENT_RELATIVE = Path(
    "experiment/stage5_scaling_validation/runs/"
    "exp30_hypersim_100_long_two_stage_overfit"
)


@dataclass(frozen=True)
class GpuState:
    index: int
    free_mib: int
    utilization: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run and monitor Exp30.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("run", "status"):
        child = subparsers.add_parser(command)
        child.add_argument("--experiment-dir", type=Path, required=True)
        child.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    return parser.parse_args()


def now() -> str:
    return dt.datetime.now().astimezone().isoformat()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_data_root(repository: Path, *, safe_root: Path) -> Path:
    for relative in DATA_RELATIVES:
        candidate = repository / relative
        if candidate.is_dir():
            return assert_safe_path(
                candidate,
                safe_root=safe_root,
                must_exist=True,
            )
    expected = ", ".join(str(repository / relative) for relative in DATA_RELATIVES)
    raise FileNotFoundError(f"Exp11 data was not found at: {expected}")


def query_gpus() -> list[GpuState]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    result = []
    for line in completed.stdout.splitlines():
        index, free_mib, utilization = (
            int(field.strip()) for field in line.split(",")
        )
        result.append(GpuState(index, free_mib, utilization))
    return result


def eligible_gpus(states: Sequence[GpuState], minimum_free_mib: int) -> list[int]:
    return [
        state.index
        for state in states
        if state.free_mib >= minimum_free_mib
    ]


def update_status(
    experiment: Path,
    *,
    state: str,
    **values: Any,
) -> None:
    path = experiment / "artifacts" / "status.json"
    previous = read_json(path) if path.is_file() else {"started_at": now()}
    atomic_json(
        path,
        {
            **previous,
            "state": state,
            "updated_at": now(),
            **values,
        },
    )


def wait_for_gpus(
    experiment: Path,
    *,
    count: int,
    minimum_free_mib: int,
    interval_seconds: int = 600,
) -> list[int]:
    while True:
        states = query_gpus()
        candidates = eligible_gpus(states, minimum_free_mib)
        if len(candidates) >= count:
            return candidates[:count]
        update_status(
            experiment,
            state="waiting_for_gpu",
            requested_gpu_count=count,
            eligible_gpu_indices=candidates,
            gpu_states=[state.__dict__ for state in states],
            next_check_seconds=interval_seconds,
        )
        time.sleep(interval_seconds)


def training_launcher(
    python: Path,
    *,
    process_count: int,
) -> list[str]:
    if process_count == 1:
        return [str(python), "-m", "moge.scripts.train_hypersim_joint_v3"]
    return [
        str(python),
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc_per_node={process_count}",
        "-m",
        "moge.scripts.train_hypersim_joint_v3",
    ]


def common_training_args(
    *,
    data: Path,
    output: Path,
    rois: Path,
    safe_root: Path,
    steps: int,
    detach_steps: int,
    ssr_lr: float,
    head_lr: float,
    backbone_lr: float,
    backbone_freeze_steps: int,
    backbone_warmup_end: int,
    decay_start: int,
    decay_end: int,
    full_eval_every: int,
    periodic_train_samples: int,
    selection_scope: str,
    checkpoint_option: tuple[str, Path] | None,
) -> list[str]:
    values = [
        "--data",
        str(data),
        "--output",
        str(output),
        "--safe-root",
        str(safe_root),
        "--device",
        "cuda:0",
        "--ddp-backend",
        "gloo",
        "--height",
        "384",
        "--width",
        "512",
        "--num-tokens",
        "2500",
        "--ssr-normalization",
        "batch_norm",
        "--refinement-steps",
        "3",
        "--steps",
        str(steps),
        "--batch-size",
        "8",
        "--microbatch-size",
        "2",
        "--refiner-detach-steps",
        str(detach_steps),
        "--backbone-freeze-steps",
        str(backbone_freeze_steps),
        "--backbone-warmup-end",
        str(backbone_warmup_end),
        "--fine-structure-rois",
        str(rois),
        "--ssr-learning-rate",
        str(ssr_lr),
        "--head-learning-rate",
        str(head_lr),
        "--backbone-learning-rate",
        str(backbone_lr),
        "--learning-rate-schedule",
        "cosine",
        "--learning-rate-decay-start-step",
        str(decay_start),
        "--learning-rate-decay-end-step",
        str(decay_end),
        "--learning-rate-final-scale",
        "0.1",
        "--weight-decay",
        "0.01",
        "--gradient-clip-norm",
        "1.0",
        "--max-preclip-grad-norm",
        "25",
        "--max-skipped-preclip-steps",
        "5",
        "--max-consecutive-skipped-preclip-steps",
        "2",
        "--smooth-log-depth-residual-bound",
        "0.1",
        "--max-abs-log-depth-residual",
        "0.100001",
        "--max-abs-raw-log-depth-residual",
        "1.5",
        "--raw-residual-warning-threshold",
        "1.0",
        "--raw-residual-tail-threshold",
        "0.3",
        "--raw-residual-tail-weight",
        "0.01",
        "--max-bound-saturation-fraction",
        "0.05",
        "--max-consecutive-saturated-steps",
        "3",
        "--max-voxel-depth-span",
        "512",
        "--max-refined-point-rel",
        "0.3",
        "--max-refined-to-base-ratio",
        "3",
        "--max-base-to-best-ratio",
        "2",
        "--global-weight",
        "1",
        "--local-weight",
        "1",
        "--edge-weight",
        "1",
        "--local-scales",
        "4",
        "16",
        "64",
        "--eval-every",
        "500",
        "--full-eval-every",
        str(full_eval_every),
        "--periodic-train-samples",
        str(periodic_train_samples),
        "--selection-split",
        "train",
        "--selection-scope",
        selection_scope,
        "--best-checkpoint-include-optimizer",
        "--eval-batch-size",
        "1",
        "--log-every",
        "25",
        "--loss-smoothing-window",
        "100",
        "--boundary-threshold",
        "0.03",
        "--seed",
        "173",
    ]
    if checkpoint_option is not None:
        values.extend((checkpoint_option[0], str(checkpoint_option[1])))
    return values


def run_process(
    command: Sequence[str],
    *,
    log_path: Path,
    gpu_indices: Sequence[int],
    experiment: Path,
    status_values: dict[str, Any],
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_indices))
    update_status(
        experiment,
        state="running",
        physical_gpu_indices=list(gpu_indices),
        log=str(log_path),
        command=list(command),
        **status_values,
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(
            json.dumps(
                {
                    "event": "launch",
                    "at": now(),
                    "gpus": list(gpu_indices),
                    "command": list(command),
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        log.flush()
        completed = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            env=environment,
        )
    return completed.returncode


def benchmark_result(output: Path) -> dict[str, Any]:
    rows = read_csv(output / "training_history.csv")
    durations = [
        float(row["optimization_step_seconds"])
        for row in rows
        if int(row["step"]) >= 5
    ]
    if not durations:
        raise RuntimeError(f"Benchmark contains no timed steps: {output}")
    report = read_json(output / "report.json")
    return {
        "process_count": int(report["distributed"]["world_size"]),
        "median_step_seconds": statistics.median(durations),
        "mean_step_seconds": statistics.fmean(durations),
        "peak_memory_bytes_by_process": report["peak_memory_bytes_by_process"],
        "output": str(output),
    }


def run_benchmarks(
    *,
    python: Path,
    experiment: Path,
    data: Path,
    rois: Path,
    safe_root: Path,
    minimum_free_mib: int,
) -> dict[str, Any]:
    destination = experiment / "artifacts" / "benchmarks"
    results = []
    for process_count in (1, 2, 4):
        output = destination / f"{process_count}_gpu"
        existing_report = output / "report.json"
        if existing_report.is_file():
            results.append(benchmark_result(output))
            continue
        gpu_indices = wait_for_gpus(
            experiment,
            count=process_count,
            minimum_free_mib=minimum_free_mib,
        )
        args = common_training_args(
            data=data,
            output=output,
            rois=rois,
            safe_root=safe_root,
            steps=20,
            detach_steps=62500,
            ssr_lr=1e-5,
            head_lr=1e-6,
            backbone_lr=5e-8,
            backbone_freeze_steps=1000,
            backbone_warmup_end=2000,
            decay_start=5000,
            decay_end=62500,
            full_eval_every=0,
            periodic_train_samples=32,
            selection_scope="full",
            checkpoint_option=None,
        )
        return_code = run_process(
            training_launcher(python, process_count=process_count) + args,
            log_path=destination / f"{process_count}_gpu.log",
            gpu_indices=gpu_indices,
            experiment=experiment,
            status_values={
                "phase": "benchmark",
                "process_count": process_count,
            },
        )
        if return_code:
            raise RuntimeError(
                f"{process_count}-GPU benchmark failed with exit code {return_code}"
            )
        results.append(benchmark_result(output))
    selected = min(results, key=lambda item: item["median_step_seconds"])
    payload = {
        "completed_at": now(),
        "criterion": "minimum median optimization-step wall time over steps 5-20",
        "results": results,
        "selected_process_count": selected["process_count"],
    }
    atomic_json(destination / "summary.json", payload)
    return payload


def full_scores(paths: Iterable[Path]) -> list[tuple[int, float]]:
    by_step: dict[int, float] = {}
    for path in paths:
        for row in read_csv(path):
            step = int(row["step"])
            by_step[step] = (
                float(row["train/k3_point_rel"])
                + float(row["train/k3_structure_point_rel"])
            )
    return sorted(by_step.items())


def window_improvement(
    scores: Sequence[tuple[int, float]],
    *,
    end_step: int,
    window_steps: int,
) -> float:
    eligible_end = [item for item in scores if item[0] <= end_step]
    eligible_start = [
        item for item in scores if item[0] <= end_step - window_steps
    ]
    if not eligible_end or not eligible_start:
        return float("inf")
    return relative_window_improvement(
        eligible_start[-1][1],
        eligible_end[-1][1],
    )


def latest_safe_checkpoint(output: Path) -> Path:
    milestones = sorted((output / "milestones").glob("step_*.pt"))
    if milestones:
        return milestones[-1]
    resume = output / "resume_checkpoint.pt"
    if resume.is_file():
        return resume
    checkpoint = output / "checkpoint.pt"
    if checkpoint.is_file():
        return checkpoint
    raise FileNotFoundError(f"No safe checkpoint exists in {output}")


def best_checkpoint(attempts: Sequence[Path]) -> tuple[Path, dict[str, Any]]:
    candidates = []
    for attempt in attempts:
        metadata_path = attempt / "checkpoint_metadata.json"
        checkpoint_path = attempt / "checkpoint.pt"
        if metadata_path.is_file() and checkpoint_path.is_file():
            candidates.append(
                (float(read_json(metadata_path)["score"]), checkpoint_path)
            )
    if not candidates:
        raise RuntimeError("No completed best checkpoint is available")
    _, checkpoint = min(candidates, key=lambda item: item[0])
    return checkpoint, read_json(checkpoint.parent / "checkpoint_metadata.json")


def log_contains_oom(path: Path) -> bool:
    if not path.is_file():
        return False
    tail = path.read_text(encoding="utf-8", errors="replace")[-200_000:].lower()
    return "out of memory" in tail or "cuda error: out of memory" in tail


def run_stage(
    *,
    stage: str,
    python: Path,
    process_count: int,
    gpu_indices: Sequence[int],
    experiment: Path,
    data: Path,
    rois: Path,
    safe_root: Path,
    rung_targets: Sequence[int],
    detach_steps: int,
    learning_rates: tuple[float, float, float],
    freeze_steps: int,
    warmup_end: int,
    decay_start: int,
    decay_end: int,
    plateau_window: int,
    minimum_improvement: float,
    transition_from: Path | None,
) -> list[Path]:
    root = experiment / "artifacts" / "training" / stage
    attempts: list[Path] = []
    attempt_index = 0
    recovery_count = 0
    lr_scale = 1.0
    checkpoint_option = (
        ("--transition-from", transition_from)
        if transition_from is not None
        else None
    )
    current_output = root / f"attempt_{attempt_index:02d}"
    attempts.append(current_output)
    repeated_model_oom = 0

    for target in rung_targets:
        while True:
            gpu_indices = wait_for_gpus(
                experiment,
                count=process_count,
                minimum_free_mib=28672,
            )
            if (current_output / "resume_checkpoint.pt").is_file():
                checkpoint_option = (
                    "--resume",
                    current_output / "resume_checkpoint.pt",
                )
            args = common_training_args(
                data=data,
                output=current_output,
                rois=rois,
                safe_root=safe_root,
                steps=target,
                detach_steps=detach_steps,
                ssr_lr=learning_rates[0] * lr_scale,
                head_lr=learning_rates[1] * lr_scale,
                backbone_lr=learning_rates[2] * lr_scale,
                backbone_freeze_steps=freeze_steps,
                backbone_warmup_end=warmup_end,
                decay_start=decay_start,
                decay_end=decay_end,
                full_eval_every=2500,
                periodic_train_samples=32,
                selection_scope="composite",
                checkpoint_option=checkpoint_option,
            )
            log_path = current_output / f"run_to_{target:06d}.log"
            return_code = run_process(
                training_launcher(python, process_count=process_count) + args,
                log_path=log_path,
                gpu_indices=gpu_indices,
                experiment=experiment,
                status_values={
                    "phase": stage,
                    "target_step": target,
                    "attempt": attempt_index,
                    "recovery_count": recovery_count,
                    "learning_rate_scale": lr_scale,
                },
            )
            if return_code == 0:
                repeated_model_oom = 0
                break

            if log_contains_oom(log_path):
                states = query_gpus()
                selected_states = {
                    state.index: state for state in states
                }
                external_pressure = any(
                    selected_states[index].free_mib < 28672
                    for index in gpu_indices
                )
                if external_pressure:
                    gpu_indices = wait_for_gpus(
                        experiment,
                        count=process_count,
                        minimum_free_mib=28672,
                    )
                    checkpoint_option = (
                        "--resume",
                        current_output / "resume_checkpoint.pt",
                    )
                    continue
                repeated_model_oom += 1
                if repeated_model_oom < 2:
                    checkpoint_option = (
                        "--resume",
                        current_output / "resume_checkpoint.pt",
                    )
                    continue
                raise RuntimeError(
                    "The same safe checkpoint produced model-intrinsic OOM twice"
                )

            recovery_count += 1
            if recovery_count > 2:
                raise RuntimeError(
                    f"{stage} exceeded two automatic recoveries"
                )
            source = latest_safe_checkpoint(current_output)
            attempt_index += 1
            lr_scale *= 0.5
            current_output = root / f"attempt_{attempt_index:02d}"
            attempts.append(current_output)
            checkpoint_option = ("--transition-from", source)

        scores = full_scores(
            attempt / "full_evaluation_history.csv"
            for attempt in attempts
        )
        improvement = window_improvement(
            scores,
            end_step=target,
            window_steps=plateau_window,
        )
        atomic_json(
            root / f"rung_{target:06d}.json",
            {
                "stage": stage,
                "target_step": target,
                "relative_window_improvement": improvement,
                "minimum_required": minimum_improvement,
                "continue": improvement >= minimum_improvement,
                "attempts": [str(path) for path in attempts],
                "completed_at": now(),
            },
        )
        if improvement < minimum_improvement:
            break
    return attempts


def evaluate_checkpoint(
    *,
    python: Path,
    checkpoint: Path,
    name: str,
    experiment: Path,
    data: Path,
    rois: Path,
    safe_root: Path,
    gpu_index: int,
) -> None:
    output = experiment / "artifacts" / "evaluation" / name
    if (output / "report.json").is_file():
        return
    command = [
        str(python),
        "-m",
        "moge.scripts.evaluate_hypersim_checkpoint_rois_v3",
        "--data",
        str(data),
        "--checkpoint",
        str(checkpoint),
        "--fine-structure-rois",
        str(rois),
        "--output",
        str(output),
        "--safe-root",
        str(safe_root),
        "--device",
        "cuda:0",
        "--evaluation-steps",
        "0",
        "1",
        "3",
        "5",
        "--batch-size",
        "1",
        "--smooth-log-depth-residual-bound",
        "0.1",
        "--max-voxel-depth-span",
        "512",
    ]
    return_code = run_process(
        command,
        log_path=output / "evaluation.log",
        gpu_indices=[gpu_index],
        experiment=experiment,
        status_values={"phase": "final_evaluation", "checkpoint_name": name},
    )
    if return_code:
        raise RuntimeError(f"Final evaluation failed for {name}")


def run(args: argparse.Namespace) -> None:
    safe_root = args.safe_root.resolve()
    experiment = assert_safe_path(
        args.experiment_dir,
        safe_root=safe_root,
        must_exist=True,
        writable=True,
    )
    repository = experiment.parents[3]
    repository = assert_safe_path(
        repository,
        safe_root=safe_root,
        must_exist=True,
    )
    data = find_data_root(repository, safe_root=safe_root)
    python = assert_safe_path(
        safe_root / "2026_TPAMI_InfiniGeometry/envs/moge3/bin/python",
        safe_root=safe_root,
        must_exist=True,
    )
    config = read_json(experiment / "config.json")
    rois = experiment / "artifacts" / "fine_structure_rois.json"
    if not rois.is_file():
        completed = subprocess.run(
            [
                str(python),
                "tools/moge3/prepare_exp30_rois.py",
                "--data",
                str(data),
                "--output",
                str(rois),
                "--safe-root",
                str(safe_root),
            ],
            cwd=repository,
        )
        if completed.returncode:
            raise RuntimeError("Failed to prepare Exp30 fine-structure ROIs")
    rois = assert_safe_path(
        rois,
        safe_root=safe_root,
        must_exist=True,
    )

    benchmark = run_benchmarks(
        python=python,
        experiment=experiment,
        data=data,
        rois=rois,
        safe_root=safe_root,
        minimum_free_mib=int(
            config["resources"]["minimum_free_memory_mib_per_gpu"]
        ),
    )
    process_count = int(benchmark["selected_process_count"])
    gpu_indices = wait_for_gpus(
        experiment,
        count=process_count,
        minimum_free_mib=int(
            config["resources"]["minimum_free_memory_mib_per_gpu"]
        ),
    )

    stage1_attempts = run_stage(
        stage="stage1",
        python=python,
        process_count=process_count,
        gpu_indices=gpu_indices,
        experiment=experiment,
        data=data,
        rois=rois,
        safe_root=safe_root,
        rung_targets=(20000, 40000, 62500),
        detach_steps=62500,
        learning_rates=(1e-5, 1e-6, 5e-8),
        freeze_steps=1000,
        warmup_end=2000,
        decay_start=5000,
        decay_end=62500,
        plateau_window=5000,
        minimum_improvement=0.01,
        transition_from=None,
    )
    stage1_best, stage1_metadata = best_checkpoint(stage1_attempts)
    stage1_step = int(stage1_metadata["step"])
    stage2_targets = tuple(
        stage1_step + additional for additional in (10000, 17500, 25000)
    )
    stage2_attempts = run_stage(
        stage="stage2",
        python=python,
        process_count=process_count,
        gpu_indices=gpu_indices,
        experiment=experiment,
        data=data,
        rois=rois,
        safe_root=safe_root,
        rung_targets=stage2_targets,
        detach_steps=stage1_step,
        learning_rates=(1e-6, 2.5e-7, 1e-8),
        freeze_steps=0,
        warmup_end=1,
        decay_start=stage1_step,
        decay_end=stage1_step + 25000,
        plateau_window=2500,
        minimum_improvement=0.005,
        transition_from=stage1_best,
    )
    final_best, final_metadata = best_checkpoint(stage2_attempts)
    initial_checkpoint = stage1_attempts[0] / "initial_checkpoint.pt"

    evaluation_gpu = wait_for_gpus(
        experiment,
        count=1,
        minimum_free_mib=28672,
    )[0]
    for name, checkpoint in (
        ("initial", initial_checkpoint),
        ("stage1_best", stage1_best),
        ("final", final_best),
    ):
        evaluate_checkpoint(
            python=python,
            checkpoint=checkpoint,
            name=name,
            experiment=experiment,
            data=data,
            rois=rois,
            safe_root=safe_root,
            gpu_index=evaluation_gpu,
        )
    update_status(
        experiment,
        state="training_complete",
        stage1_best_checkpoint=str(stage1_best),
        stage1_best=stage1_metadata,
        final_best_checkpoint=str(final_best),
        final_best=final_metadata,
        completed_at=now(),
    )


def status(args: argparse.Namespace) -> None:
    experiment = assert_safe_path(
        args.experiment_dir,
        safe_root=args.safe_root,
        must_exist=True,
    )
    status_path = experiment / "artifacts" / "status.json"
    payload: dict[str, Any] = (
        read_json(status_path)
        if status_path.is_file()
        else {"state": "not_started"}
    )
    histories = sorted(
        (experiment / "artifacts" / "training").glob(
            "stage*/attempt_*/training_history.csv"
        )
    )
    latest = None
    if histories:
        rows = read_csv(histories[-1])
        if rows:
            latest = {
                "history": str(histories[-1]),
                **rows[-1],
            }
    payload["latest_training_record"] = latest
    payload["gpu_states"] = [state.__dict__ for state in query_gpus()]
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.command == "run":
        try:
            run(args)
        except Exception as error:
            experiment = assert_safe_path(
                args.experiment_dir,
                safe_root=args.safe_root,
                must_exist=True,
                writable=True,
            )
            update_status(
                experiment,
                state="failed",
                error=type(error).__name__,
                message=str(error),
                failed_at=now(),
            )
            raise
    else:
        status(args)


if __name__ == "__main__":
    main()
