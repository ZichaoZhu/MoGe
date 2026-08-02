#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP4="$REPO/experiment/stage1_learning_capability/runs/exp4_hypersim_multiscene_generalization"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp6_two_stage_joint_finetuning"
DATA="$EXP4/data"
SELECTION="$EXP4/rod_selection.json"
CHECKPOINT="$EXP_DIR/artifacts/training/checkpoint.pt"
OUTPUT="$EXP_DIR/artifacts/rod_visualization"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
test "$(realpath -e "$DATA")" = "$DATA"
test "$(realpath -e "$SELECTION")" = "$SELECTION"
test "$(realpath -e "$CHECKPOINT")" = "$CHECKPOINT"
case "$(realpath -m "$OUTPUT")" in
  "$SAFE_ROOT"/*) ;;
  *) echo "输出目录越过安全根目录" >&2; exit 91 ;;
esac
test ! -e "$OUTPUT/report.json"
install -d "$OUTPUT" "$CACHE/huggingface" "$CACHE/torch" "$CACHE/pip" \
  "$CACHE/xdg" "$TMP"

cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3

"$ENVIRONMENT/bin/python" tools/moge3/visualize_split_rods.py \
  --data "$DATA" \
  --selection "$SELECTION" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --series K=3 "$CHECKPOINT" 3 \
  --experiment-label exp6-best-step250 \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --mask-dilation 0 \
  --yaw-min -90 \
  --yaw-max 90 \
  --orbit-frames 46 \
  --motion one-way \
  --frame-duration-ms 110
