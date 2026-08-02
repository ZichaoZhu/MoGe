#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
REPO="$SAFE_ROOT/2026_TPAMI_InfiniGeometry/third_party/MoGe-3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp10_hypersim_48_staged_joint_overfit"
RUNNER="$EXP_DIR/run_recovery_single.sh"
RESULT="$EXP_DIR/artifacts/logs/recovery_pipeline_result.json"
GPU_INDEX="${1:?GPU index is required}"
MIN_FREE_MIB="${MOGE3_MIN_FREE_MIB:-40000}"
MAX_UTILIZATION="${MOGE3_MAX_UTILIZATION:-10}"
COOLDOWN_TIMEOUT="${MOGE3_COOLDOWN_TIMEOUT:-600}"

case "$(realpath -m "$RESULT")/" in
  "$SAFE_ROOT/"*) ;;
  *) exit 91 ;;
esac

wait_for_gpu_cooldown() {
  local waited=0
  local free_mib utilization
  while (( waited <= COOLDOWN_TIMEOUT )); do
    read -r free_mib utilization < <(
      nvidia-smi --id="$GPU_INDEX" \
        --query-gpu=memory.free,utilization.gpu \
        --format=csv,noheader,nounits |
        awk -F',' \
          '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
    )
    if (( free_mib >= MIN_FREE_MIB && utilization <= MAX_UTILIZATION )); then
      return 0
    fi
    sleep 30
    waited=$((waited + 30))
  done
  echo "GPU $GPU_INDEX 在${COOLDOWN_TIMEOUT}秒内未恢复启动条件" >&2
  return 1
}

run_status=0
if [[ ! -f "$EXP_DIR/artifacts/recovery_smoke_step3300_lr1e6/report.json" ]]; then
  MOGE3_GPU="$GPU_INDEX" MOGE3_RUN_MODE=smoke bash "$RUNNER"
  run_status=$?
fi
if (( run_status == 0 )) \
  && [[ ! -f "$EXP_DIR/artifacts/recovery_step3300_batch8_lr1e6/report.json" ]]
then
  if wait_for_gpu_cooldown; then
    MOGE3_GPU="$GPU_INDEX" MOGE3_RUN_MODE=formal bash "$RUNNER"
    run_status=$?
  else
    run_status=8
  fi
fi

/mnt/data/home/zhuzichao/2026_TPAMI_InfiniGeometry/envs/moge3/bin/python \
  - "$RESULT" "$run_status" "$GPU_INDEX" <<'PY'
import datetime
import json
import os
import pathlib
import sys

destination = pathlib.Path(sys.argv[1])
status = int(sys.argv[2])
payload = {
    "status": "complete" if status == 0 else "failed",
    "exit_code": status,
    "physical_gpu": int(sys.argv[3]),
    "finished_at": datetime.datetime.now().astimezone().isoformat(),
}
temporary = destination.with_suffix(destination.suffix + ".incomplete")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, destination)
PY
exit "$run_status"
