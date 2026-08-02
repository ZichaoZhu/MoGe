#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
SOURCE=/nas1/datasets/hypersim/raw
EXP_DIR="$REPO/experiment/stage1_learning_capability/runs/exp2_hypersim_smallset_overfit"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
install -d \
  "$EXP_DIR/data" \
  "$EXP_DIR/artifacts" \
  "$EXP_DIR/metrics" \
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
export CUDA_VISIBLE_DEVICES=2

"$ENVIRONMENT/bin/python" tools/moge3/prepare_hypersim_smallset.py \
  --spec "$EXP_DIR/data_spec.json" \
  --source-root "$SOURCE" \
  --output "$EXP_DIR/data" \
  --safe-root "$SAFE_ROOT"

"$ENVIRONMENT/bin/python" -m moge.scripts.train_hypersim_smallset_v3 \
  --data "$EXP_DIR/data" \
  --output "$EXP_DIR/artifacts" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 192 \
  --width 256 \
  --num-tokens 1200 \
  --refinement-steps 3 \
  --steps 2000 \
  --batch-size 2 \
  --learning-rate 2e-4 \
  --weight-decay 1e-2 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every 200 \
  --log-every 20 \
  --loss-smoothing-window 50 \
  --boundary-threshold 0.03 \
  --seed 23

mv "$EXP_DIR/artifacts/report.json" "$EXP_DIR/metrics/report.json"
