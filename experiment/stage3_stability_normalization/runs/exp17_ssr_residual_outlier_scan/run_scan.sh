#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp17_ssr_residual_outlier_scan"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data"
CHECKPOINT="$REPO/experiment/stage2_training_strategy/runs/exp15_detached_head_ssr_coadaptation/artifacts/formal/checkpoint.pt"
OUTPUT="$EXP_DIR/artifacts/scan"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU="${1:?GPU编号不能为空}"

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac
test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$CHECKPOINT" "$OUTPUT" "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
test -f "$CHECKPOINT"
if [[ -f "$OUTPUT/report.json" ]]; then
  exit 0
fi

install -d "$OUTPUT" "$EXP_DIR/artifacts/logs" "$EXP_DIR/artifacts/status" \
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
"$ENVIRONMENT/bin/python" tools/moge3/scan_checkpoint_residuals.py \
  --data "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --train-mode-repeats 3 \
  --threshold 0.5 \
  --seed 171 \
  --splits train val test \
  --highlight-ids \
    ai_045_004_cam_01_frame.0099 \
    ai_001_001_cam_00_frame.0099 \
    ai_026_019_cam_00_frame.0000 \
    ai_052_008_cam_01_frame.0099 \
    ai_044_008_cam_01_frame.0099 \
    ai_027_009_cam_02_frame.0099 \
    ai_008_010_cam_00_frame.0000 \
    ai_002_008_cam_00_frame.0099 \
  >>"$EXP_DIR/artifacts/logs/scan.log" 2>&1 || status=$?

"$ENVIRONMENT/bin/python" - "$EXP_DIR/artifacts/status/scan.json" \
  "$status" "$GPU" "$STARTED_AT" <<'PY'
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
    "started_at": sys.argv[4],
    "finished_at": datetime.datetime.now().astimezone().isoformat(),
}
temporary = path.with_suffix(path.suffix + ".incomplete")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
exit "$status"
