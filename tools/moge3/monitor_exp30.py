from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from moge.utils.remote_guard import DEFAULT_SAFE_ROOT, assert_safe_path
from tools.moge3.run_exp30 import (
    latest_launch_failure_kind,
    now,
    query_gpus,
    read_json,
    update_status,
)


STEP_PATTERN = re.compile(r'^\{"step":\s*(\d+),', re.MULTILINE)


@dataclass(frozen=True)
class MonitorDecision:
    action: str
    reason: str
    effective_recovery_count: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor Exp30 every 30 minutes.")
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--safe-root", type=Path, default=DEFAULT_SAFE_ROOT)
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=1800,
        help="Daemon check interval; production default is 1800 seconds.",
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Keep checking until training completes or reaches a terminal failure.",
    )
    return parser.parse_args()


def process_is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    stat = Path(f"/proc/{pid}/stat")
    if stat.is_file():
        fields = stat.read_text(encoding="utf-8").split()
        if len(fields) >= 3 and fields[2] == "Z":
            return False
    return True


def pipeline_is_alive(experiment: Path) -> tuple[bool, int | None]:
    pid_file = experiment / "artifacts" / "pipeline.pid"
    if not pid_file.is_file():
        return False, None
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
    except ValueError:
        return False, None
    return process_is_alive(pid), pid


def latest_attempt(experiment: Path, phase: str) -> Path | None:
    root = experiment / "artifacts" / "training" / phase
    attempts = sorted(path for path in root.glob("attempt_*") if path.is_dir())
    return attempts[-1] if attempts else None


def latest_run_log(attempt: Path | None) -> Path | None:
    if attempt is None:
        return None
    logs = sorted(
        attempt.glob("run_to_*.log"),
        key=lambda path: path.stat().st_mtime_ns,
    )
    return logs[-1] if logs else None


def latest_logged_step(log: Path | None) -> int | None:
    if log is None or not log.is_file():
        return None
    matches = STEP_PATTERN.findall(
        log.read_text(encoding="utf-8", errors="replace")
    )
    return int(matches[-1]) if matches else None


def decide_action(
    *,
    state: str,
    phase: str,
    pipeline_alive: bool,
    failure_kind: str,
    recovery_count: int,
    maximum_recoveries: int,
) -> MonitorDecision:
    effective_recovery_count = recovery_count
    if state in {"training_complete", "complete"}:
        return MonitorDecision("stop_complete", state, effective_recovery_count)
    if pipeline_alive:
        return MonitorDecision("observe", state, effective_recovery_count)
    if phase != "stage1":
        return MonitorDecision(
            "stop_terminal",
            f"unsupported automatic continuation phase: {phase}",
            effective_recovery_count,
        )
    if state == "running":
        return MonitorDecision(
            "mark_failed_and_restart",
            "pipeline PID exited while status still said running",
            effective_recovery_count,
        )
    if state not in {"failed", "waiting_for_gpu"}:
        return MonitorDecision(
            "stop_terminal",
            f"non-running pipeline has unsupported state: {state}",
            effective_recovery_count,
        )
    if failure_kind == "stability":
        effective_recovery_count += 1
    if effective_recovery_count > maximum_recoveries:
        return MonitorDecision(
            "stop_terminal",
            "automatic recovery budget exhausted",
            effective_recovery_count,
        )
    if failure_kind in {"oom", "stability"}:
        return MonitorDecision(
            "restart",
            f"recognized {failure_kind} failure",
            effective_recovery_count,
        )
    return MonitorDecision(
        "stop_terminal",
        f"unknown failure kind: {failure_kind}",
        effective_recovery_count,
    )


def append_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def check_once(experiment: Path, safe_root: Path) -> MonitorDecision:
    status_path = experiment / "artifacts" / "status.json"
    status = (
        read_json(status_path)
        if status_path.is_file()
        else {"state": "not_started", "phase": "stage1"}
    )
    config = read_json(experiment / "config.json")
    maximum_recoveries = int(
        config["residual_control"]["maximum_recoveries_per_stage"]
    )
    phase = str(status.get("phase", "stage1"))
    attempt = latest_attempt(experiment, phase)
    log = latest_run_log(attempt)
    failure_kind = (
        latest_launch_failure_kind(log)
        if log is not None
        else "unknown"
    )
    alive, pid = pipeline_is_alive(experiment)
    decision = decide_action(
        state=str(status.get("state", "unknown")),
        phase=phase,
        pipeline_alive=alive,
        failure_kind=failure_kind,
        recovery_count=int(status.get("recovery_count", 0)),
        maximum_recoveries=maximum_recoveries,
    )
    report: dict[str, Any] = {
        "checked_at": now(),
        "state": status.get("state"),
        "phase": phase,
        "attempt": attempt.name if attempt is not None else None,
        "latest_step": latest_logged_step(log),
        "pipeline_alive": alive,
        "pipeline_pid": pid,
        "failure_kind": failure_kind,
        "recovery_count": status.get("recovery_count", 0),
        "maximum_recoveries": maximum_recoveries,
        "gpu_states": [asdict(state) for state in query_gpus()],
        "decision": asdict(decision),
    }

    if decision.action == "mark_failed_and_restart":
        update_status(
            experiment,
            state="failed",
            error="MonitorDetectedDeadPipeline",
            message=decision.reason,
            failed_at=now(),
        )

    if decision.action in {"restart", "mark_failed_and_restart"}:
        script = assert_safe_path(
            experiment / "continue_pipeline.sh",
            safe_root=safe_root,
            must_exist=True,
        )
        completed = subprocess.run(
            ["bash", str(script)],
            cwd=script.parent,
            check=False,
            capture_output=True,
            text=True,
        )
        report["restart"] = {
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip(),
            "stderr": completed.stderr.strip(),
        }
        if completed.returncode != 0:
            alive_after_restart, _ = pipeline_is_alive(experiment)
            decision = (
                MonitorDecision(
                    "observe",
                    "another launcher started the pipeline concurrently",
                    decision.effective_recovery_count,
                )
                if alive_after_restart
                else MonitorDecision(
                    "stop_terminal",
                    (
                        "automatic restart failed with code "
                        f"{completed.returncode}"
                    ),
                    decision.effective_recovery_count,
                )
            )
            report["decision"] = asdict(decision)

    append_report(
        experiment / "artifacts" / "monitor" / "history.jsonl",
        report,
    )
    return decision


def main() -> None:
    args = parse_args()
    if args.interval_seconds < 60:
        raise ValueError("--interval-seconds must be at least 60")
    safe_root = args.safe_root.resolve(strict=True)
    if safe_root != DEFAULT_SAFE_ROOT:
        raise PermissionError(
            f"Exp30 monitor requires the fixed safe root {DEFAULT_SAFE_ROOT}"
        )
    experiment = assert_safe_path(
        args.experiment_dir,
        safe_root=safe_root,
        must_exist=True,
        writable=True,
    )
    lock_path = experiment / "artifacts" / "monitor" / "monitor.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Exp30 monitor is already running") from error
        while True:
            decision = check_once(experiment, safe_root)
            if not args.daemon or decision.action.startswith("stop_"):
                break
            time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
