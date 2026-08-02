#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp13_fixed_base_ssr_diagnostic"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data"
ROIS="$EXP_DIR/fine_structure_rois.json"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU="${1:?GPU编号不能为空}"
ARM="${2:?学习率分支不能为空}"
MODE="${3:-formal}"

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac
case "$ARM" in
  lr_2e6) SSR_LR=2e-6 ;;
  lr_1e5) SSR_LR=1e-5 ;;
  lr_2e5) SSR_LR=2e-5 ;;
  *) exit 93 ;;
esac
case "$MODE" in
  smoke)
    OUTPUT="$EXP_DIR/artifacts/smoke_$ARM"
    TOTAL_STEPS=4
    BATCH_SIZE=2
    EVAL_EVERY=4
    PERIODIC_TRAIN=4
    SELECTION_SCOPE=full
    LIMIT_ARGS=(--max-samples-per-split 4)
    ;;
  formal)
    OUTPUT="$EXP_DIR/artifacts/$ARM"
    TOTAL_STEPS=800
    BATCH_SIZE=8
    EVAL_EVERY=100
    PERIODIC_TRAIN=100
    SELECTION_SCOPE=structure
    LIMIT_ARGS=()
    ;;
  *)
    exit 94
    ;;
esac

LOGS="$EXP_DIR/artifacts/logs"
STATUS="$EXP_DIR/artifacts/status/${MODE}_${ARM}.json"
test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$ROIS" "$OUTPUT" "$LOGS" "$STATUS" \
  "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
test -f "$DATA/manifest.json"
test -f "$ROIS"
if [[ -f "$OUTPUT/report.json" ]]; then
  exit 0
fi

install -d "$OUTPUT" "$LOGS" "$(dirname "$STATUS")" \
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

CHECKPOINT_ARGS=()
if [[ -f "$OUTPUT/resume_checkpoint.pt" ]]; then
  CHECKPOINT_ARGS=(--resume "$OUTPUT/resume_checkpoint.pt")
fi

STARTED_AT="$(date --iso-8601=seconds)"
status=0
"$ENVIRONMENT/bin/python" -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$OUTPUT" \
  "${CHECKPOINT_ARGS[@]}" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --steps "$TOTAL_STEPS" \
  --batch-size "$BATCH_SIZE" \
  --microbatch-size 1 \
  --refiner-detach-steps 2000 \
  --backbone-freeze-steps 0 \
  --backbone-warmup-end 1 \
  --train-refiner-only \
  --fine-structure-rois "$ROIS" \
  --ssr-learning-rate "$SSR_LR" \
  --head-learning-rate 0 \
  --backbone-learning-rate 0 \
  --weight-decay 1e-2 \
  --gradient-clip-norm 1 \
  --max-preclip-grad-norm 100 \
  --max-skipped-preclip-steps 5 \
  --max-consecutive-skipped-preclip-steps 2 \
  --max-abs-log-depth-residual 0.5 \
  --max-refined-point-rel 0.3 \
  --max-refined-to-base-ratio 3 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every "$EVAL_EVERY" \
  --periodic-train-samples "$PERIODIC_TRAIN" \
  --selection-split train \
  --selection-scope "$SELECTION_SCOPE" \
  --eval-batch-size 1 \
  --log-every 10 \
  --loss-smoothing-window 100 \
  --boundary-threshold 0.03 \
  --seed 131 \
  "${LIMIT_ARGS[@]}" >>"$LOGS/${MODE}_${ARM}.log" 2>&1 || status=$?

"$ENVIRONMENT/bin/python" - "$STATUS" "$status" "$GPU" "$ARM" "$MODE" \
  "$STARTED_AT" <<'PY'
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
    "arm": sys.argv[4],
    "mode": sys.argv[5],
    "started_at": sys.argv[6],
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
