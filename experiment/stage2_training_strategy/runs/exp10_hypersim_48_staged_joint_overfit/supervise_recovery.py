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
    records = []
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


def last_csv_row(path: pathlib.Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


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


def atomic_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".incomplete")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wait for a safe GPU, launch Exp10 recovery, and audit it every 10 minutes."
    )
    parser.add_argument("--experiment-dir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--run-id",
        default="recovery",
        help="Stable identifier used for lock, pid, result, and status filenames.",
    )
    parser.add_argument(
        "--pipeline-script",
        default="run_recovery_pipeline.sh",
        help="Pipeline script path relative to the experiment directory.",
    )
    parser.add_argument(
        "--formal-output",
        default="artifacts/recovery_step3300_batch8_lr1e6",
        help="Formal output path relative to the experiment directory.",
    )
    parser.add_argument(
        "--smoke-output",
        default="artifacts/recovery_smoke_step3300_lr1e6",
        help="Smoke output path relative to the experiment directory.",
    )
    parser.add_argument("--interval-seconds", type=int, default=600)
    parser.add_argument("--min-free-mib", type=int, default=40000)
    parser.add_argument("--max-utilization", type=int, default=10)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.interval_seconds <= 0 or args.min_free_mib <= 0:
        raise ValueError("Monitoring interval and free-memory threshold must be positive")
    if not args.run_id or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-"
        for character in args.run_id
    ):
        raise ValueError("--run-id may contain only letters, digits, underscore, and dash")
    experiment = safe_path(args.experiment_dir)
    logs = safe_path(experiment / "artifacts" / "logs")
    logs.mkdir(parents=True, exist_ok=True)
    status_path = safe_path(logs / f"{args.run_id}_supervisor_status.json")
    checks_path = safe_path(logs / f"{args.run_id}_supervisor_checks.jsonl")
    pid_path = safe_path(logs / f"{args.run_id}_pipeline.pid")
    result_path = safe_path(logs / f"{args.run_id}_pipeline_result.json")
    lock_path = safe_path(logs / f"{args.run_id}_supervisor.lock")
    pipeline = safe_path(experiment / args.pipeline_script)
    formal = safe_path(experiment / args.formal_output)
    smoke = safe_path(experiment / args.smoke_output)

    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Exp10 recovery supervisor is already running") from error

        pipeline_pid = (
            int(pid_path.read_text(encoding="utf-8").strip())
            if pid_path.is_file()
            else None
        )
        while True:
            now = datetime.datetime.now().astimezone().isoformat()
            gpus = gpu_status()
            result = read_json(result_path)
            alive = process_alive(pipeline_pid)
            formal_report = read_json(formal / "report.json")
            instability = (
                read_json(formal / "instability_event.json")
                or read_json(smoke / "instability_event.json")
            )

            if formal_report is not None:
                state = "complete"
            elif result is not None and result.get("status") == "failed":
                state = "failed"
            elif instability is not None and not alive:
                state = "failed"
            elif alive:
                state = "running"
            else:
                candidate = next(
                    (
                        gpu["index"]
                        for gpu in gpus
                        if gpu["memory_free_mib"] >= args.min_free_mib
                        and gpu["utilization_percent"] <= args.max_utilization
                    ),
                    None,
                )
                if candidate is None:
                    state = "waiting_for_safe_gpu"
                elif args.once:
                    state = "safe_gpu_available"
                else:
                    launcher_log = safe_path(logs / "recovery_pipeline_launcher.log")
                    with launcher_log.open("a", encoding="utf-8") as handle:
                        environment = {
                            **os.environ,
                            "MOGE3_MIN_FREE_MIB": str(args.min_free_mib),
                        }
                        process = subprocess.Popen(
                            ["bash", str(pipeline), str(candidate)],
                            cwd=experiment,
                            env=environment,
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
                "launch_policy": {
                    "min_free_mib": args.min_free_mib,
                    "max_utilization_percent": args.max_utilization,
                },
                "pipeline_pid": pipeline_pid,
                "pipeline_alive": alive,
                "gpus": gpus,
                "smoke_report_available": (smoke / "report.json").is_file(),
                "formal_report_available": formal_report is not None,
                "latest_training": last_csv_row(formal / "training_history.csv"),
                "latest_evaluation": last_csv_row(formal / "evaluation_history.csv"),
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
