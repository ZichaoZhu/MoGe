from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import json
import os
import pathlib
import signal
import subprocess
import time
from typing import Any

from experiment.stage2_training_strategy.runs.exp10_hypersim_48_staged_joint_overfit.supervise_plateau import (
    plateau_state,
)
from experiment.stage2_training_strategy.runs.exp11_hypersim_100_train_staged_joint_overfit.supervise_training import (
    gpu_status,
    select_gpus,
)


SAFE_ROOT = pathlib.Path("/mnt/data/home/zhuzichao")


def safe_path(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve(strict=False)
    if resolved == SAFE_ROOT or SAFE_ROOT not in resolved.parents:
        raise ValueError(f"Path escapes safe root: {resolved}")
    return resolved


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def last_csv_row(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def read_evaluations(
    path: pathlib.Path,
    metric: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {"step": int(row["step"]), "score": float(row[metric])}
        for row in rows
    ]


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        state = (pathlib.Path("/proc") / str(pid) / "stat").read_text().split()[2]
    except FileNotFoundError:
        return False
    except (PermissionError, IndexError):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        return True
    return state != "Z"


def matching_torchrun_pids(output: pathlib.Path) -> list[int]:
    needle = str(output)
    matches: list[int] = []
    for entry in pathlib.Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            stat = entry.stat()
            command = (
                (entry / "cmdline")
                .read_bytes()
                .replace(b"\0", b" ")
                .decode()
            )
        except (
            FileNotFoundError,
            PermissionError,
            ProcessLookupError,
            UnicodeDecodeError,
        ):
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


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def terminal_state(
    *,
    report: dict[str, Any] | None,
    result: dict[str, Any] | None,
    decision: dict[str, Any] | None,
) -> str | None:
    if report is not None:
        return "complete"
    if result is None:
        return None
    if (
        decision is not None
        and decision.get("action") == "sent_sigint_after_evaluation_plateau"
    ):
        return "stopped_on_plateau"
    return "complete" if result.get("status") == "complete" else "failed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch, monitor, and plateau-stop Exp12."
    )
    parser.add_argument("--experiment-dir", type=pathlib.Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=60)
    parser.add_argument("--min-free-mib", type=int, default=18000)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--max-gpus", type=int, default=2)
    parser.add_argument("--plateau-patience", type=int, default=5)
    parser.add_argument(
        "--minimum-relative-improvement",
        type=float,
        default=0.002,
    )
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(
        args.interval_seconds,
        args.min_free_mib,
        args.plateau_patience,
    ) <= 0:
        raise ValueError("Intervals, memory, and patience must be positive")
    if not 1 <= args.max_gpus <= 2:
        raise ValueError("--max-gpus must be in [1, 2]")
    if not 0.0 < args.minimum_relative_improvement < 1.0:
        raise ValueError("Relative improvement must be in (0, 1)")

    experiment = safe_path(args.experiment_dir)
    logs = safe_path(experiment / "artifacts" / "logs")
    logs.mkdir(parents=True, exist_ok=True)
    pipeline = safe_path(experiment / "run_pipeline.sh")
    smoke = safe_path(
        experiment / "artifacts" / "smoke_joint_from_exp11_best3000"
    )
    output = safe_path(
        experiment / "artifacts" / "joint_from_exp11_best3000_cosine"
    )
    status_path = safe_path(logs / "supervisor_status.json")
    checks_path = safe_path(logs / "supervisor_checks.jsonl")
    selection_path = safe_path(logs / "gpu_selection.json")
    pid_path = safe_path(logs / "pipeline.pid")
    result_path = safe_path(logs / "pipeline_result.json")
    decision_path = safe_path(logs / "plateau_decision.json")
    lock_path = safe_path(logs / "supervisor.lock")
    metric = "train/k3_point_rel"

    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Exp12 supervisor is already running") from error

        pipeline_pid = (
            int(pid_path.read_text(encoding="utf-8").strip())
            if pid_path.is_file()
            else None
        )
        selected_gpus = None
        if selection_path.is_file():
            selected_gpus = read_json(selection_path)["physical_gpus"]

        while True:
            now = datetime.datetime.now().astimezone().isoformat()
            records = gpu_status()
            result = read_json(result_path)
            report = read_json(output / "report.json")
            decision = read_json(decision_path)
            instability = (
                read_json(output / "instability_event.json")
                or read_json(smoke / "instability_event.json")
            )
            alive = process_alive(pipeline_pid)
            terminal = terminal_state(
                report=report,
                result=result,
                decision=decision,
            )

            if terminal is not None:
                state = terminal
            elif alive:
                state = "running"
            elif pipeline_pid is not None:
                state = "failed"
            else:
                selected_gpus = select_gpus(
                    records,
                    minimum_free_mib=args.min_free_mib,
                    maximum_utilization=args.max_utilization,
                    maximum_count=args.max_gpus,
                )
                if not selected_gpus:
                    state = "waiting_for_safe_gpu"
                elif args.once:
                    state = "safe_gpus_available"
                else:
                    selection = {
                        "selected_at": now,
                        "physical_gpus": selected_gpus,
                        "world_size": len(selected_gpus),
                        "launch_snapshot": records,
                    }
                    atomic_json(selection_path, selection)
                    launcher_log = safe_path(logs / "pipeline_launcher.log")
                    with launcher_log.open("a", encoding="utf-8") as handle:
                        process = subprocess.Popen(
                            [
                                "bash",
                                str(pipeline),
                                ",".join(
                                    str(index) for index in selected_gpus
                                ),
                            ],
                            cwd=experiment,
                            stdin=subprocess.DEVNULL,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    pipeline_pid = process.pid
                    pid_path.write_text(
                        f"{pipeline_pid}\n",
                        encoding="utf-8",
                    )
                    alive = True
                    state = "running"

            evaluations = read_evaluations(
                output / "evaluation_history.csv",
                metric,
            )
            plateau = plateau_state(
                evaluations,
                start_step=3000,
                patience=args.plateau_patience,
                minimum_relative_improvement=(
                    args.minimum_relative_improvement
                ),
            )
            torchrun_pids = matching_torchrun_pids(output)
            if (
                state == "running"
                and decision is None
                and plateau["triggered"]
            ):
                if len(torchrun_pids) != 1:
                    raise RuntimeError(
                        "Plateau triggered, but formal torchrun target is "
                        f"ambiguous: {torchrun_pids}"
                    )
                os.kill(torchrun_pids[0], signal.SIGINT)
                decision = {
                    "decided_at": now,
                    "action": "sent_sigint_after_evaluation_plateau",
                    "signaled_pid": torchrun_pids[0],
                    "metric": metric,
                    **plateau,
                }
                atomic_json(decision_path, decision)
                state = "stopping_on_plateau"

            payload = {
                "checked_at": now,
                "state": state,
                "check_interval_seconds": args.interval_seconds,
                "pipeline_pid": pipeline_pid,
                "pipeline_alive": alive,
                "selected_gpus": selected_gpus,
                "gpus": records,
                "smoke_report_available": (smoke / "report.json").is_file(),
                "formal_report_available": report is not None,
                "latest_training": last_csv_row(
                    output / "training_history.csv"
                ),
                "latest_evaluation": last_csv_row(
                    output / "evaluation_history.csv"
                ),
                "plateau": plateau,
                "plateau_decision": decision,
                "formal_torchrun_pids": torchrun_pids,
                "instability_event": instability,
                "pipeline_result": result,
            }
            atomic_json(status_path, payload)
            with checks_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if args.once or state in {
                "complete",
                "failed",
                "stopped_on_plateau",
            }:
                return
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
