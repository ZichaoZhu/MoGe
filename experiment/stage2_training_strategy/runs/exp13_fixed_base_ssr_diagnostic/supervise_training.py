from __future__ import annotations

import argparse
import csv
import datetime
import fcntl
import json
import os
import pathlib
import subprocess
import time
from typing import Any


SAFE_ROOT = pathlib.Path("/mnt/data/home/zhuzichao")
ARMS = ("lr_2e6", "lr_1e5", "lr_2e5")
SMOKE_ARM = "lr_1e5"


def safe_path(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve(strict=False)
    if resolved == SAFE_ROOT or SAFE_ROOT not in resolved.parents:
        raise ValueError(f"Path escapes safe root: {resolved}")
    return resolved


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def process_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def gpu_status() -> list[dict[str, int]]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = []
    for line in completed.stdout.splitlines():
        index, used, free, utilization = (
            int(value.strip()) for value in line.split(",")
        )
        records.append(
            {
                "index": index,
                "memory_used_mib": used,
                "memory_free_mib": free,
                "utilization_percent": utilization,
            }
        )
    return records


def select_gpus(
    records: list[dict[str, int]],
    *,
    minimum_free_mib: int,
    maximum_utilization: int,
    excluded: set[int],
    maximum_count: int,
) -> list[int]:
    eligible = [
        record
        for record in records
        if record["index"] not in excluded
        and record["memory_free_mib"] >= minimum_free_mib
        and record["utilization_percent"] <= maximum_utilization
    ]
    eligible.sort(
        key=lambda record: (
            -record["memory_free_mib"],
            record["utilization_percent"],
            record["index"],
        )
    )
    return [record["index"] for record in eligible[:maximum_count]]


def last_csv_row(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch and monitor the Exp13 fixed-base SSR LR sweep."
    )
    parser.add_argument("--experiment-dir", type=pathlib.Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--minimum-free-mib", type=int, default=18000)
    parser.add_argument("--maximum-utilization", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def run_state(
    experiment: pathlib.Path,
    *,
    arm: str,
    mode: str,
) -> dict[str, Any]:
    status_path = experiment / "artifacts" / "status" / f"{mode}_{arm}.json"
    pid_path = experiment / "artifacts" / "status" / f"{mode}_{arm}.pid"
    launch_path = experiment / "artifacts" / "status" / f"{mode}_{arm}.launch.json"
    status = read_json(status_path)
    launch_record = read_json(launch_path)
    pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
    alive = process_alive(pid)
    output_name = f"smoke_{arm}" if mode == "smoke" else arm
    output = experiment / "artifacts" / output_name
    return {
        "arm": arm,
        "mode": mode,
        "state": (
            status["status"]
            if status is not None
            else "running"
            if alive
            else "not_started"
            if pid is None
            else "failed_without_status"
        ),
        "pid": pid,
        "physical_gpu": (
            status.get("physical_gpu")
            if status is not None
            else launch_record.get("physical_gpu")
            if launch_record is not None
            else None
        ),
        "last_training": last_csv_row(output / "training_history.csv"),
        "last_evaluation": last_csv_row(output / "evaluation_history.csv"),
        "report_exists": (output / "report.json").is_file(),
    }


def launch(
    experiment: pathlib.Path,
    *,
    gpu: int,
    arm: str,
    mode: str,
) -> int:
    logs = experiment / "artifacts" / "logs"
    status_dir = experiment / "artifacts" / "status"
    logs.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"launcher_{mode}_{arm}.log"
    with log_path.open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [
                "bash",
                str(experiment / "run_single.sh"),
                str(gpu),
                arm,
                mode,
            ],
            cwd=experiment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (status_dir / f"{mode}_{arm}.pid").write_text(
        f"{process.pid}\n",
        encoding="utf-8",
    )
    atomic_json(
        status_dir / f"{mode}_{arm}.launch.json",
        {
            "launched_at": datetime.datetime.now().astimezone().isoformat(),
            "physical_gpu": gpu,
            "arm": arm,
            "mode": mode,
            "pid": process.pid,
        },
    )
    return process.pid


def main() -> None:
    args = parse_args()
    if min(args.interval_seconds, args.minimum_free_mib) <= 0:
        raise ValueError("Intervals and memory thresholds must be positive")
    if not 0 <= args.maximum_utilization <= 100:
        raise ValueError("GPU utilization threshold must be in [0, 100]")
    experiment = safe_path(args.experiment_dir)
    logs = safe_path(experiment / "artifacts" / "logs")
    logs.mkdir(parents=True, exist_ok=True)
    lock_path = safe_path(logs / "supervisor.lock")
    checks_path = safe_path(logs / "monitor_checks.jsonl")
    status_path = safe_path(logs / "supervisor_status.json")

    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Exp13 supervisor is already running") from error

        while True:
            now = datetime.datetime.now().astimezone().isoformat()
            gpus = gpu_status()
            smoke = run_state(experiment, arm=SMOKE_ARM, mode="smoke")
            formal = [
                run_state(experiment, arm=arm, mode="formal")
                for arm in ARMS
            ]
            action: dict[str, Any] = {"type": "none"}

            if smoke["state"] == "not_started":
                available = select_gpus(
                    gpus,
                    minimum_free_mib=args.minimum_free_mib,
                    maximum_utilization=args.maximum_utilization,
                    excluded=set(),
                    maximum_count=1,
                )
                if available and not args.once:
                    pid = launch(
                        experiment,
                        gpu=available[0],
                        arm=SMOKE_ARM,
                        mode="smoke",
                    )
                    action = {
                        "type": "launched_smoke",
                        "arm": SMOKE_ARM,
                        "gpu": available[0],
                        "pid": pid,
                    }
                elif available:
                    action = {"type": "smoke_gpu_available", "gpu": available[0]}
                else:
                    action = {"type": "waiting_for_smoke_gpu"}
            elif smoke["state"] == "complete":
                running_gpus = {
                    int(state["physical_gpu"])
                    for state in formal
                    if state["state"] == "running"
                    and state["physical_gpu"] is not None
                }
                pending = [
                    state for state in formal if state["state"] == "not_started"
                ]
                available = select_gpus(
                    gpus,
                    minimum_free_mib=args.minimum_free_mib,
                    maximum_utilization=args.maximum_utilization,
                    excluded=running_gpus,
                    maximum_count=len(pending),
                )
                launched = []
                if not args.once:
                    for state, gpu in zip(pending, available):
                        pid = launch(
                            experiment,
                            gpu=gpu,
                            arm=state["arm"],
                            mode="formal",
                        )
                        launched.append(
                            {"arm": state["arm"], "gpu": gpu, "pid": pid}
                        )
                action = (
                    {"type": "launched_formal", "runs": launched}
                    if launched
                    else {"type": "monitoring_formal"}
                )
            else:
                action = {"type": "smoke_terminal_or_running"}

            smoke = run_state(experiment, arm=SMOKE_ARM, mode="smoke")
            formal = [
                run_state(experiment, arm=arm, mode="formal")
                for arm in ARMS
            ]
            if smoke["state"] == "failed" or smoke["state"] == "failed_without_status":
                state = "smoke_failed"
            elif all(run["state"] == "complete" for run in formal):
                state = "complete"
            elif any(
                run["state"] in {"failed", "failed_without_status"}
                for run in formal
            ):
                state = "formal_failed"
            elif any(run["state"] == "running" for run in formal):
                state = "formal_running"
            elif smoke["state"] == "complete":
                state = "waiting_for_formal_gpu"
            elif smoke["state"] == "running":
                state = "smoke_running"
            else:
                state = "waiting_for_smoke_gpu"

            record = {
                "checked_at": now,
                "interval_seconds": args.interval_seconds,
                "state": state,
                "action": action,
                "gpus": gpus,
                "smoke": smoke,
                "formal": formal,
            }
            with checks_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            atomic_json(status_path, record)

            if args.once or state in {"complete", "smoke_failed", "formal_failed"}:
                break
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
