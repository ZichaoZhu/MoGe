#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage1_learning_capability/runs/exp4_hypersim_multiscene_generalization"
DATA="$EXP_DIR/data"
CHECKPOINT="$EXP_DIR/artifacts/training_final/latest_checkpoint.pt"
OUTPUT="$EXP_DIR/artifacts/fine_structure/presentation"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
test "$(realpath -e "$DATA")" = "$DATA"
test "$(realpath -e "$CHECKPOINT")" = "$CHECKPOINT"
test ! -e "$OUTPUT/report.json"
install -d \
  "$OUTPUT" \
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
export CUDA_VISIBLE_DEVICES=0

"$ENVIRONMENT/bin/python" tools/moge3/visualize_fine_structure.py \
  --data "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --max-k 5 \
  --crop-height 192 \
  --crop-width 192 \
  --crop-stride 32 \
  --min-fine-pixels 96 \
  --yaw 28 \
  --pitch -8 \
  --splits train test \
  --orbit-splits train test \
  --train-sample-id ai_036_002_cam_00_frame.0040 \
  --test-sample-id ai_048_001_cam_00_frame.0046 \
  --selection-provenance posthoc_rgb_gt_only \
  --seed 59
