#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
REPO="$SAFE_ROOT/2026_TPAMI_InfiniGeometry/third_party/MoGe-3"
ENVIRONMENT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp12_hypersim_100_immediate_joint_finetuning"
LOGS="$EXP_DIR/artifacts/logs"
RESULT="$LOGS/pipeline_result.json"
GPU_LIST="${1:?逗号分隔的GPU编号不能为空}"

case "$(realpath -m "$RESULT")/" in
  "$SAFE_ROOT/"*) ;;
  *) exit 91 ;;
esac
install -d "$LOGS"

status=0
MOGE3_RUN_MODE=smoke \
  bash "$EXP_DIR/run_single.sh" "$GPU_LIST" || status=$?
if (( status == 0 )); then
  MOGE3_RUN_MODE=formal \
    bash "$EXP_DIR/run_single.sh" "$GPU_LIST" || status=$?
fi

"$ENVIRONMENT/bin/python" - "$RESULT" "$status" "$GPU_LIST" <<'PY'
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
    "physical_gpus": [int(value) for value in sys.argv[3].split(",")],
    "finished_at": datetime.datetime.now().astimezone().isoformat(),
}
temporary = destination.with_suffix(destination.suffix + ".incomplete")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, destination)
PY
exit "$status"
