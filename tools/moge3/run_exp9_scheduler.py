from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Sequence, Tuple

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch exp9 samples on currently idle GPUs without DDP"
    )
    parser.add_argument("--experiment-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument("--required-free-mib", type=int, required=True)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--max-utilization-percent", type=int, default=10)
    return parser.parse_args()


def gpu_status() -> Dict[int, Tuple[int, int]]:
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
    status: Dict[int, Tuple[int, int]] = {}
    for line in completed.stdout.splitlines():
        index, memory, utilization = (
            part.strip() for part in line.split(",", maxsplit=2)
        )
        status[int(index)] = (int(memory), int(utilization))
    return status


def load_samples(root: Path) -> list[tuple[str, Path]]:
    samples = []
    for directory in sorted(root.iterdir()):
        config_path = directory / "config.json"
        if not directory.is_dir() or not config_path.is_file():
            continue
        config = json.loads(config_path.read_text(encoding="utf-8"))
        samples.append((str(config["sample_id"]), directory))
    if len(samples) != 5:
        raise ValueError(f"Expected five samples, found {len(samples)}")
    return samples


def completed(directory: Path) -> bool:
    report = directory / "report.json"
    if not report.is_file():
        return False
    return json.loads(report.read_text(encoding="utf-8")).get("status") == "complete"


def training_command(
    *,
    sample_id: str,
    directory: Path,
    data: Path,
    selection: Path,
    safe_root: Path,
) -> Sequence[str]:
    command = [
        sys.executable,
        "-m",
        "moge.scripts.overfit_hypersim_staged_v3",
        "--data",
        str(data),
        "--selection",
        str(selection),
        "--sample-id",
        sample_id,
        "--output",
        str(directory),
        "--safe-root",
        str(safe_root),
        "--device",
        "cuda:0",
    ]
    resume = directory / "checkpoints" / "resume.pt"
    if resume.is_file():
        command.extend(("--resume", str(resume)))
    return command


def run(args: argparse.Namespace) -> None:
    if args.required_free_mib <= 4096:
        raise ValueError("Required free memory must include a meaningful safety margin")
    if not 1 <= args.max_parallel <= 4:
        raise ValueError("Exp9 supports one to four independent workers")
    if args.poll_seconds < 10:
        raise ValueError("Polling more frequently than every 10 seconds is unnecessary")
    if not 0 <= args.max_utilization_percent <= 100:
        raise ValueError("GPU utilization threshold must be in [0, 100]")

    safe_root = args.safe_root.resolve()
    root = assert_safe_path(
        args.experiment_root,
        safe_root=safe_root,
        must_exist=True,
        writable=True,
    )
    data = assert_safe_path(args.data, safe_root=safe_root, must_exist=True)
    selection = assert_safe_path(
        args.selection,
        safe_root=safe_root,
        must_exist=True,
    )
    pending = [
        (sample_id, directory)
        for sample_id, directory in load_samples(root)
        if not completed(directory)
    ]
    running: Dict[int, tuple[str, Path, subprocess.Popen[str], object]] = {}
    failures = []
    logs = root / "logs"
    logs.mkdir(exist_ok=True)

    while pending or running:
        for gpu, (sample_id, directory, process, handle) in list(running.items()):
            return_code = process.poll()
            if return_code is None:
                continue
            handle.close()
            del running[gpu]
            record = {
                "sample_id": sample_id,
                "directory": directory.name,
                "gpu": gpu,
                "return_code": return_code,
                "complete_report_present": completed(directory),
            }
            print(json.dumps({"finished": record}, ensure_ascii=False), flush=True)
            if return_code != 0 and not record["complete_report_present"]:
                failures.append(record)

        if pending and len(running) < args.max_parallel:
            status = gpu_status()
            available = [
                gpu
                for gpu, (memory, utilization) in sorted(
                    status.items(),
                    key=lambda item: item[1][0],
                    reverse=True,
                )
                if (
                    gpu not in running
                    and memory >= args.required_free_mib
                    and utilization <= args.max_utilization_percent
                )
            ]
            while pending and available and len(running) < args.max_parallel:
                gpu = available.pop(0)
                sample_id, directory = pending.pop(0)
                log_path = logs / f"{directory.name}.log"
                handle = log_path.open("a", encoding="utf-8")
                environment = os.environ.copy()
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
                command = training_command(
                    sample_id=sample_id,
                    directory=directory,
                    data=data,
                    selection=selection,
                    safe_root=safe_root,
                )
                process = subprocess.Popen(
                    command,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    env=environment,
                )
                running[gpu] = (sample_id, directory, process, handle)
                print(
                    json.dumps(
                        {
                            "launched": {
                                "sample_id": sample_id,
                                "directory": directory.name,
                                "gpu": gpu,
                                "free_mib_at_launch": status[gpu][0],
                                "utilization_percent_at_launch": status[gpu][1],
                                "pid": process.pid,
                                "log": str(log_path),
                            }
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

        if pending or running:
            time.sleep(args.poll_seconds)

    status = {
        "status": "failed" if failures else "complete",
        "failures": failures,
    }
    (root / "scheduler_report.json").write_text(
        json.dumps(status, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if failures:
        raise RuntimeError(f"{len(failures)} exp9 samples failed")


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
