#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp10_hypersim_48_staged_joint_overfit"
LOGS="$EXP_DIR/artifacts/logs"
SCHEDULER_LOG="$LOGS/scheduler.log"
MIN_FREE_MIB="${MOGE3_MIN_FREE_MIB:-24576}"
MAX_UTIL="${MOGE3_MAX_UTIL:-10}"
POLL_SECONDS="${MOGE3_POLL_SECONDS:-60}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
for path in "$REPO" "$EXP_DIR" "$LOGS"; do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done
install -d "$LOGS"

while true; do
  mapfile -t candidates < <(
    nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
      --format=csv,noheader,nounits |
      awk -F',' -v free="$MIN_FREE_MIB" -v util="$MAX_UTIL" '
        {
          gsub(/ /, "", $1); gsub(/ /, "", $2); gsub(/ /, "", $3);
          if ($2 >= free && $3 <= util) print $1;
        }'
  )
  printf '%s 等待GPU：候选=%s，要求空闲显存>=%sMiB、利用率<=%s%%\n' \
    "$(date --iso-8601=seconds)" "${candidates[*]:-无}" \
    "$MIN_FREE_MIB" "$MAX_UTIL" >>"$SCHEDULER_LOG"
  if (( ${#candidates[@]} >= 2 )); then
    pair="${candidates[0]},${candidates[1]}"
    printf '%s 选择GPU %s，启动两卡冒烟测试\n' \
      "$(date --iso-8601=seconds)" "$pair" >>"$SCHEDULER_LOG"
    export MOGE3_GPUS="$pair"
    "$EXP_DIR/run_smoke.sh"
    printf '%s 两卡冒烟测试通过，启动Exp10正式训练\n' \
      "$(date --iso-8601=seconds)" >>"$SCHEDULER_LOG"
    exec "$EXP_DIR/run_training.sh"
  fi
  sleep "$POLL_SECONDS"
done
