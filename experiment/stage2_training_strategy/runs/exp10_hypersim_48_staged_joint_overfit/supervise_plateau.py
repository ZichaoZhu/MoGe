from __future__ import annotations

import argparse
import csv
import datetime
import json
import os
import pathlib
import signal
import time
from typing import Any


SAFE_ROOT = pathlib.Path("/mnt/data/home/zhuzichao")


def safe_path(path: pathlib.Path, *, must_exist: bool = False) -> pathlib.Path:
    resolved = path.resolve(strict=False)
    if resolved == SAFE_ROOT or SAFE_ROOT not in resolved.parents:
        raise ValueError(f"Path escapes safe root: {resolved}")
    if must_exist and not resolved.exists():
        raise FileNotFoundError(resolved)
    return resolved


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_evaluations(path: pathlib.Path, metric: str) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for record in records:
        if metric not in record:
            raise KeyError(f"Metric {metric!r} is absent from {path}")
        parsed.append(
            {
                "step": int(record["step"]),
                "score": float(record[metric]),
            }
        )
    return parsed


def plateau_state(
    evaluations: list[dict[str, Any]],
    *,
    start_step: int,
    patience: int,
    minimum_relative_improvement: float,
) -> dict[str, Any]:
    eligible = [record for record in evaluations if record["step"] >= start_step]
    if not eligible:
        return {
            "triggered": False,
            "reason": "waiting_for_initial_evaluation",
            "consecutive_non_improving": 0,
            "best_score": None,
            "best_step": None,
        }

    reference_score = float(eligible[0]["score"])
    reference_step = int(eligible[0]["step"])
    actual_best_score = reference_score
    actual_best_step = reference_step
    consecutive = 0
    window: list[int] = []
    for record in eligible[1:]:
        score = float(record["score"])
        step = int(record["step"])
        if score < actual_best_score:
            actual_best_score = score
            actual_best_step = step
        threshold = reference_score * (1.0 - minimum_relative_improvement)
        if score < threshold:
            reference_score = score
            reference_step = step
            consecutive = 0
            window = []
        else:
            consecutive += 1
            window.append(step)

    return {
        "triggered": consecutive >= patience,
        "reason": (
            "plateau"
            if consecutive >= patience
            else "collecting_evaluations"
        ),
        "consecutive_non_improving": consecutive,
        "non_improving_window": window[-patience:],
        "significant_reference_score": reference_score,
        "significant_reference_step": reference_step,
        "best_score": actual_best_score,
        "best_step": actual_best_step,
    }


def matching_torchrun_pids(output: pathlib.Path) -> list[int]:
    needle = str(output)
    matches: list[int] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = entry.stat()
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError):
            continue
        if stat.st_uid != os.getuid():
            continue
        if (
            "torchrun" in command
            and "moge.scripts.train_hypersim_joint_v3" in command
            and needle in command
        ):
            matches.append(int(entry.name))
    return sorted(matches)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stop one Exp10 training branch at a verified evaluation plateau."
    )
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--status", type=pathlib.Path, required=True)
    parser.add_argument("--decision", type=pathlib.Path, required=True)
    parser.add_argument("--start-step", type=int, required=True)
    parser.add_argument("--metric", default="train/k3_point_rel")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--minimum-relative-improvement", type=float, default=0.005)
    parser.add_argument("--interval-seconds", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.start_step < 0:
        raise ValueError("--start-step must be non-negative")
    if args.patience <= 0 or args.interval_seconds <= 0:
        raise ValueError("Patience and interval must be positive")
    if not 0.0 < args.minimum_relative_improvement < 1.0:
        raise ValueError("Relative improvement must be in (0, 1)")

    output = safe_path(args.output, must_exist=True)
    status = safe_path(args.status)
    decision = safe_path(args.decision)
    history = safe_path(output / "evaluation_history.csv")
    report = safe_path(output / "report.json")
    pipeline_result = safe_path(
        output.parent / "logs" / "adaptive_joint_lr5e7_pipeline_result.json"
    )

    while True:
        now = datetime.datetime.now().astimezone().isoformat()
        evaluations = read_evaluations(history, args.metric)
        state = plateau_state(
            evaluations,
            start_step=args.start_step,
            patience=args.patience,
            minimum_relative_improvement=args.minimum_relative_improvement,
        )
        pids = matching_torchrun_pids(output)
        payload = {
            "checked_at": now,
            "output": str(output),
            "metric": args.metric,
            "start_step": args.start_step,
            "patience": args.patience,
            "minimum_relative_improvement": args.minimum_relative_improvement,
            "latest_evaluation_step": (
                evaluations[-1]["step"] if evaluations else None
            ),
            "torchrun_pids": pids,
            **state,
        }
        atomic_json(status, payload)

        if report.is_file():
            payload["action"] = "training_completed_without_plateau_stop"
            atomic_json(decision, payload)
            return
        if pipeline_result.is_file() and not pids:
            payload["action"] = "pipeline_exited_before_plateau_decision"
            atomic_json(decision, payload)
            return
        if state["triggered"]:
            if len(pids) != 1:
                payload["action"] = "refused_to_signal_ambiguous_processes"
                atomic_json(decision, payload)
                raise RuntimeError(
                    f"Expected exactly one matching torchrun process, found {pids}"
                )
            os.kill(pids[0], signal.SIGINT)
            payload["action"] = "sent_sigint_after_evaluation_plateau"
            payload["signaled_pid"] = pids[0]
            atomic_json(decision, payload)
            return
        if not pids and evaluations:
            payload["action"] = "training_process_missing"
            atomic_json(decision, payload)
            raise RuntimeError("Training process disappeared before a terminal state")
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
