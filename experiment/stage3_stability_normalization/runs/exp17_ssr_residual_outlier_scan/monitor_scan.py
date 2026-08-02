from __future__ import annotations

import argparse
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
    eligible.sort(key=lambda item: (-item["memory_free_mib"], item["index"]))
    return eligible[0]["index"] if eligible else None


def main() -> None:
    parser = argparse.ArgumentParser(description="每十分钟监督一次Exp17残差扫描。")
    parser.add_argument("--experiment-dir", type=pathlib.Path, required=True)
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--minimum-free-mib", type=int, default=30000)
    parser.add_argument("--maximum-utilization", type=int, default=10)
    args = parser.parse_args()
    if min(args.interval_seconds, args.minimum_free_mib) <= 0:
        raise ValueError("检查间隔和最低空闲显存必须为正数")

    experiment = safe_path(args.experiment_dir)
    logs = safe_path(experiment / "artifacts" / "logs")
    status_dir = safe_path(experiment / "artifacts" / "status")
    logs.mkdir(parents=True, exist_ok=True)
    status_dir.mkdir(parents=True, exist_ok=True)
    checks_path = logs / "monitor_checks.jsonl"
    monitor_status = logs / "monitor_status.json"
    pid_path = status_dir / "scan.pid"
    launch_path = status_dir / "scan.launch.json"
    scan_status_path = status_dir / "scan.json"

    with (logs / "monitor.lock").open("w", encoding="utf-8") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            checked_at = datetime.datetime.now().astimezone().isoformat()
            gpus = gpu_status()
            status = read_json(scan_status_path)
            pid = int(pid_path.read_text().strip()) if pid_path.is_file() else None
            alive = process_alive(pid)
            action: dict[str, Any] = {"type": "monitoring"}
            if status is None and pid is None:
                gpu = select_gpu(
                    gpus,
                    minimum_free_mib=args.minimum_free_mib,
                    maximum_utilization=args.maximum_utilization,
                )
                if gpu is None:
                    action = {"type": "waiting_for_gpu"}
                else:
                    with (logs / "launcher.log").open("a", encoding="utf-8") as handle:
                        process = subprocess.Popen(
                            ["bash", str(experiment / "run_scan.sh"), str(gpu)],
                            cwd=experiment,
                            stdin=subprocess.DEVNULL,
                            stdout=handle,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                    pid = process.pid
                    pid_path.write_text(f"{pid}\n", encoding="utf-8")
                    atomic_json(
                        launch_path,
                        {
                            "launched_at": checked_at,
                            "physical_gpu": gpu,
                            "pid": pid,
                        },
                    )
                    action = {"type": "launched", "gpu": gpu, "pid": pid}
                    alive = True
            status = read_json(scan_status_path)
            if status is not None:
                overall = status["status"]
            elif alive:
                overall = "running"
            elif pid is not None:
                overall = "failed_without_status"
            else:
                overall = "waiting_for_gpu"
            record = {
                "checked_at": checked_at,
                "overall_state": overall,
                "action": action,
                "gpus": gpus,
                "pid": pid,
                "process_alive": alive,
                "scan_status": status,
                "report_exists": (experiment / "artifacts" / "scan" / "report.json").is_file(),
            }
            with checks_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            atomic_json(monitor_status, record)
            if overall in {"complete", "failed", "failed_without_status"}:
                return
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
