#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage4_loss_gradient_audit/runs/exp25_loss_gradient_structure_audit"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data"
ROIS="$REPO/experiment/stage2_training_strategy/runs/exp13_fixed_base_ssr_diagnostic/fine_structure_rois.json"
CHECKPOINT="$REPO/experiment/stage3_stability_normalization/runs/exp24_microbatch2_long_detached/artifacts/formal/training/latest_checkpoint.pt"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU="${1:?GPU编号不能为空}"
MODE="${2:-formal}"

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac
case "$MODE" in
  smoke) TRAIN_ROIS=1 ;;
  formal) TRAIN_ROIS=12 ;;
  *) exit 93 ;;
esac

ROOT="$EXP_DIR/artifacts/$MODE"
STATUS="$EXP_DIR/artifacts/status/${MODE}.json"
LOGS="$EXP_DIR/artifacts/logs"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$ROIS" "$CHECKPOINT" "$ROOT" \
  "$STATUS" "$LOGS" "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
for file in "$ROIS" "$CHECKPOINT" "$DATA/manifest.json"
do
  test -f "$file"
done
if [[ -f "$ROOT/report.json" ]]; then
  exit 0
fi

install -d "$ROOT" "$LOGS" "$(dirname "$STATUS")" \
  "$CACHE/huggingface" "$CACHE/torch" "$CACHE/pip" "$CACHE/xdg" "$TMP"
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"

STARTED_AT="$(date --iso-8601=seconds)"
status=0
"$ENVIRONMENT/bin/python" tools/moge3/audit_loss_gradients.py \
  --data "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --fine-structure-rois "$ROIS" \
  --output "$ROOT" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 0 3 \
  --train-roi-samples "$TRAIN_ROIS" \
  --local-scales 4 16 64 \
  --boundary-threshold 0.03 \
  --seed 251 >>"$LOGS/${MODE}.log" 2>&1 || status=$?

"$ENVIRONMENT/bin/python" - "$STATUS" "$status" "$GPU" "$MODE" "$STARTED_AT" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "status": "complete" if int(sys.argv[2]) == 0 else "failed",
    "exit_code": int(sys.argv[2]),
    "physical_gpu": int(sys.argv[3]),
    "mode": sys.argv[4],
    "started_at": sys.argv[5],
    "finished_at": datetime.datetime.now().astimezone().isoformat(),
}
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".incomplete")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
exit "$status"
