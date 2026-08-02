#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp14_head_ssr_short_joint"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data"
ROIS="$REPO/experiment/stage2_training_strategy/runs/exp13_fixed_base_ssr_diagnostic/fine_structure_rois.json"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU="${1:?GPU编号不能为空}"
VARIANT="${2:-final}"

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac
case "$VARIANT" in
  final)
    CHECKPOINT="$EXP_DIR/artifacts/formal/latest_checkpoint.pt"
    OUTPUT="$EXP_DIR/artifacts/posthoc_final"
    ;;
  best)
    CHECKPOINT="$EXP_DIR/artifacts/formal/checkpoint.pt"
    OUTPUT="$EXP_DIR/artifacts/posthoc_best"
    ;;
  *)
    exit 93
    ;;
esac
test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$ROIS" "$CHECKPOINT" "$OUTPUT" "$CACHE" "$TMP"
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

install -d "$OUTPUT" "$EXP_DIR/artifacts/logs" "$CACHE/huggingface" \
  "$CACHE/torch" "$CACHE/pip" "$CACHE/xdg" "$TMP"
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

"$ENVIRONMENT/bin/python" \
  -m moge.scripts.evaluate_hypersim_checkpoint_rois_v3 \
  --data "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --fine-structure-rois "$ROIS" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --evaluation-steps 0 1 3 5 \
  --batch-size 1 \
  --boundary-threshold 0.03 \
  >>"$EXP_DIR/artifacts/logs/posthoc_${VARIANT}.log" 2>&1
