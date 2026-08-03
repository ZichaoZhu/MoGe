#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
EXP_DIR="$SAFE_ROOT/2026_TPAMI_InfiniGeometry/third_party/MoGe-3/experiment/stage3_stability_normalization/runs/exp29_smooth_bounded_residual_long_joint"
GPU="${1:?GPU编号不能为空}"
INTERVAL_SECONDS=600
MONITOR_LOG="$EXP_DIR/artifacts/logs/monitor.log"
STATUS="$EXP_DIR/artifacts/status/formal.json"
PID_FILE="$EXP_DIR/artifacts/status/formal_launcher.pid"

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac
case "$(realpath -m "$EXP_DIR")/" in "$SAFE_ROOT/"*) ;; *) exit 91 ;; esac
install -d "$(dirname "$MONITOR_LOG")" "$(dirname "$STATUS")"

while true; do
  timestamp="$(date --iso-8601=seconds)"
  if [[ -f "$STATUS" ]]; then
    printf '%s final_status=%s\n' "$timestamp" "$(tr '\n' ' ' < "$STATUS")" \
      >>"$MONITOR_LOG"
    exit 0
  fi
  if [[ -f "$PID_FILE" ]]; then
    pid="$(cat "$PID_FILE")"
    if ! kill -0 "$pid" 2>/dev/null; then
      printf '%s launcher_not_running pid=%s\n' "$timestamp" "$pid" \
        >>"$MONITOR_LOG"
      exit 1
    fi
  fi
  gpu_line="$(nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu \
    --format=csv,noheader | sed -n "$((GPU + 1))p")"
  latest="$(tail -n 1 "$EXP_DIR/artifacts/logs/formal.log" 2>/dev/null || true)"
  printf '%s gpu=%s latest=%s\n' "$timestamp" "$gpu_line" "$latest" \
    >>"$MONITOR_LOG"
  sleep "$INTERVAL_SECONDS"
done
