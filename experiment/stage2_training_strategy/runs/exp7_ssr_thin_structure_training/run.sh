#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
SOURCE=/nas1/datasets/hypersim/raw
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training"
DATA="$EXP_DIR/data"
TRAINING="$EXP_DIR/artifacts/training"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for writable_path in "$EXP_DIR" "$DATA" "$TRAINING" "$CACHE" "$TMP"; do
  case "$writable_path/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "拒绝越界写入：$writable_path" >&2; exit 91 ;;
  esac
done

install -d \
  "$DATA" \
  "$TRAINING" \
  "$CACHE/huggingface" \
  "$CACHE/torch" \
  "$CACHE/pip" \
  "$CACHE/xdg" \
  "$TMP"

cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=1

"$ENVIRONMENT/bin/python" tools/moge3/prepare_hypersim_generalization.py \
  --spec "$EXP_DIR/data_spec.json" \
  --source-root "$SOURCE" \
  --output "$DATA" \
  --safe-root "$SAFE_ROOT"

"$ENVIRONMENT/bin/python" -m moge.scripts.train_hypersim_generalization_v3 \
  --data "$DATA" \
  --output "$TRAINING" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --evaluation-steps 0 1 3 5 \
  --steps 4000 \
  --batch-size 2 \
  --microbatch-size 1 \
  --learning-rate 2e-5 \
  --weight-decay 1e-2 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every 200 \
  --log-every 20 \
  --periodic-train-samples 24 \
  --loss-smoothing-window 50 \
  --boundary-threshold 0.03 \
  --seed 71 \
  --diagnose-loss-gradients
