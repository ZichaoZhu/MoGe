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


def safe_path(path: pathlib.Path) -> pathlib.Path:
    resolved = path.resolve(strict=False)
    if resolved == SAFE_ROOT or SAFE_ROOT not in resolved.parents:
        raise ValueError(f"Path escapes safe root: {resolved}")
    return resolved


def gpu_status() -> list[dict[str, int]]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    records: list[dict[str, int]] = []
    for line in output.splitlines():
        index, used, free, utilization = [
            int(value.strip()) for value in line.split(",")
        ]
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
    maximum_count: int,
) -> list[int]:
    eligible = [
        record
        for record in records
        if record["memory_free_mib"] >= minimum_free_mib
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


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Exp11 on the largest safe GPU set and audit it every 10 minutes."
        )
    )
    parser.add_argument("--experiment-dir", type=pathlib.Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--min-free-mib", type=int, default=18000)
    parser.add_argument("--max-utilization", type=int, default=30)
    parser.add_argument("--max-gpus", type=int, default=4)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval_seconds <= 0 or args.min_free_mib <= 0:
        raise ValueError("Interval and memory threshold must be positive")
    if not 1 <= args.max_gpus <= 4:
        raise ValueError("--max-gpus must be in [1, 4]")
    experiment = safe_path(args.experiment_dir)
    logs = safe_path(experiment / "artifacts" / "logs")
    logs.mkdir(parents=True, exist_ok=True)
    pipeline = safe_path(experiment / "run_pipeline.sh")
    output = safe_path(experiment / "artifacts" / "training")
    status_path = safe_path(logs / "supervisor_status.json")
    checks_path = safe_path(logs / "supervisor_checks.jsonl")
    selection_path = safe_path(logs / "gpu_selection.json")
    pid_path = safe_path(logs / "pipeline.pid")
    result_path = safe_path(logs / "pipeline_result.json")
    lock_path = safe_path(logs / "supervisor.lock")

    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Exp11 supervisor is already running") from error

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
            instability = read_json(output / "instability_event.json")
            alive = process_alive(pipeline_pid)

            if report is not None:
                state = "complete"
            elif result is not None and result.get("status") == "failed":
                state = "failed"
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
                        "policy": {
                            "minimum_free_mib": args.min_free_mib,
                            "maximum_utilization_percent": args.max_utilization,
                            "maximum_gpu_count": args.max_gpus,
                        },
                    }
                    atomic_json(selection_path, selection)
                    launcher_log = safe_path(logs / "pipeline_launcher.log")
                    with launcher_log.open("a", encoding="utf-8") as handle:
                        process = subprocess.Popen(
                            [
                                "bash",
                                str(pipeline),
                                ",".join(str(index) for index in selected_gpus),
                            ],
                            cwd=experiment,
                            stdin=subprocess.DEVNULL,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    pipeline_pid = process.pid
                    pid_path.write_text(f"{pipeline_pid}\n", encoding="utf-8")
                    alive = True
                    state = "running"

            payload = {
                "checked_at": now,
                "state": state,
                "check_interval_seconds": args.interval_seconds,
                "pipeline_pid": pipeline_pid,
                "pipeline_alive": alive,
                "selected_gpus": selected_gpus,
                "gpus": records,
                "smoke_reports": [
                    str(path)
                    for path in sorted((experiment / "artifacts").glob("smoke_*/report.json"))
                ],
                "formal_report_available": report is not None,
                "latest_training": last_csv_row(output / "training_history.csv"),
                "latest_evaluation": last_csv_row(output / "evaluation_history.csv"),
                "instability_event": instability,
                "pipeline_result": result,
            }
            atomic_json(status_path, payload)
            with checks_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

            if args.once or state in {"complete", "failed"}:
                return
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
