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
        raise ValueError(f"路径越过安全根目录：{resolved}")
    return resolved


def read_json(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def last_csv_row(path: pathlib.Path) -> dict[str, str] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def process_alive(pid: int | None) -> bool:
    if pid is None:
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


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_state(experiment: pathlib.Path, mode: str) -> dict[str, Any]:
    status_dir = experiment / "artifacts" / "status"
    status = read_json(status_dir / f"{mode}.json")
    launch = read_json(status_dir / f"{mode}.launch.json")
    pid_path = status_dir / f"{mode}.pid"
    pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
    alive = process_alive(pid)
    output = experiment / "artifacts" / mode
    state = (
        status["status"]
        if status is not None
        else "running"
        if alive
        else "not_started"
        if pid is None
        else "failed_without_status"
    )
    return {
        "mode": mode,
        "state": state,
        "pid": pid,
        "physical_gpu": (
            status.get("physical_gpu")
            if status is not None
            else launch.get("physical_gpu")
            if launch is not None
            else None
        ),
        "last_training": last_csv_row(output / "training_history.csv"),
        "last_evaluation": last_csv_row(output / "evaluation_history.csv"),
        "report_exists": (output / "report.json").is_file(),
    }


def launch(experiment: pathlib.Path, *, gpu: int, mode: str) -> int:
    logs = experiment / "artifacts" / "logs"
    status_dir = experiment / "artifacts" / "status"
    logs.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    with (logs / f"launcher_{mode}.log").open("a", encoding="utf-8") as handle:
        process = subprocess.Popen(
            ["bash", str(experiment / "run_single.sh"), str(gpu), mode],
            cwd=experiment,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    (status_dir / f"{mode}.pid").write_text(f"{process.pid}\n", encoding="utf-8")
    atomic_json(
        status_dir / f"{mode}.launch.json",
        {
            "launched_at": datetime.datetime.now().astimezone().isoformat(),
            "physical_gpu": gpu,
            "mode": mode,
            "pid": process.pid,
        },
    )
    return process.pid


def select_gpu(
    records: list[dict[str, int]],
    *,
    minimum_free_mib: int,
    maximum_utilization: int,
) -> int | None:
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
    return eligible[0]["index"] if eligible else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="每十分钟检查一次 Exp14，并依次启动冒烟与正式训练。"
    )
    parser.add_argument("--experiment-dir", type=pathlib.Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--minimum-free-mib", type=int, default=30000)
    parser.add_argument("--maximum-utilization", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if min(args.interval_seconds, args.minimum_free_mib) <= 0:
        raise ValueError("检查间隔和最低空闲显存必须为正数")
    experiment = safe_path(args.experiment_dir)
    logs = safe_path(experiment / "artifacts" / "logs")
    logs.mkdir(parents=True, exist_ok=True)
    checks_path = safe_path(logs / "monitor_checks.jsonl")
    status_path = safe_path(logs / "monitor_status.json")

    with (logs / "monitor.lock").open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Exp14 监控程序已经运行") from error

        while True:
            checked_at = datetime.datetime.now().astimezone().isoformat()
            gpus = gpu_status()
            smoke = run_state(experiment, "smoke")
            formal = run_state(experiment, "formal")
            action: dict[str, Any] = {"type": "none"}

            if smoke["state"] == "not_started":
                gpu = select_gpu(
                    gpus,
                    minimum_free_mib=args.minimum_free_mib,
                    maximum_utilization=args.maximum_utilization,
                )
                if gpu is None:
                    action = {"type": "waiting_for_smoke_gpu"}
                elif args.once:
                    action = {"type": "smoke_gpu_available", "gpu": gpu}
                else:
                    action = {
                        "type": "launched_smoke",
                        "gpu": gpu,
                        "pid": launch(experiment, gpu=gpu, mode="smoke"),
                    }
            elif smoke["state"] == "complete" and formal["state"] == "not_started":
                gpu = select_gpu(
                    gpus,
                    minimum_free_mib=args.minimum_free_mib,
                    maximum_utilization=args.maximum_utilization,
                )
                if gpu is None:
                    action = {"type": "waiting_for_formal_gpu"}
                elif args.once:
                    action = {"type": "formal_gpu_available", "gpu": gpu}
                else:
                    action = {
                        "type": "launched_formal",
                        "gpu": gpu,
                        "pid": launch(experiment, gpu=gpu, mode="formal"),
                    }
            else:
                action = {"type": "monitoring"}

            smoke = run_state(experiment, "smoke")
            formal = run_state(experiment, "formal")
            if smoke["state"] in {"failed", "failed_without_status"}:
                overall = "smoke_failed"
            elif formal["state"] in {"failed", "failed_without_status"}:
                overall = "formal_failed"
            elif formal["state"] == "complete":
                overall = "complete"
            elif formal["state"] == "running":
                overall = "formal_running"
            elif smoke["state"] == "complete":
                overall = "waiting_for_formal_gpu"
            elif smoke["state"] == "running":
                overall = "smoke_running"
            else:
                overall = "waiting_for_smoke_gpu"

            record = {
                "checked_at": checked_at,
                "overall_state": overall,
                "action": action,
                "gpus": gpus,
                "smoke": smoke,
                "formal": formal,
            }
            with checks_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            atomic_json(status_path, record)

            if args.once or overall in {
                "complete",
                "smoke_failed",
                "formal_failed",
            }:
                return
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
