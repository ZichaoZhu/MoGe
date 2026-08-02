#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage1_learning_capability/runs/exp1_hypersim_single_batch_overfit"
DATA="$EXP_DIR/data"
CHECKPOINT="$EXP_DIR/artifacts/checkpoint.pt"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
test "$(realpath -e "$DATA")" = "$DATA"
test "$(realpath -e "$CHECKPOINT")" = "$CHECKPOINT"
for selection_file in \
  "$EXP_DIR/rod_selection.json" \
  "$EXP_DIR/rod_selection_window.json" \
  "$EXP_DIR/rod_selection_faucet.json"; do
  test "$(realpath -e "$selection_file")" = "$selection_file"
done
for output_dir in \
  "$EXP_DIR/artifacts/rod_visualization" \
  "$EXP_DIR/artifacts/rod_visualization_window" \
  "$EXP_DIR/artifacts/rod_visualization_faucet"; do
  case "$(realpath -m "$output_dir")" in
    "$SAFE_ROOT"/*) ;;
    *) echo "输出目录越过安全根目录" >&2; exit 91 ;;
  esac
  test ! -e "$output_dir/report.json"
done
install -d "$CACHE/huggingface" "$CACHE/torch" "$CACHE/pip" "$CACHE/xdg" "$TMP"

cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3

render_region() {
  local region_label=$1
  local selection_file=$2
  local output_dir=$3
  "$ENVIRONMENT/bin/python" tools/moge3/visualize_split_rods.py \
    --data "$DATA" \
    --selection "$selection_file" \
    --output "$output_dir" \
    --safe-root "$SAFE_ROOT" \
    --series K=3 "$CHECKPOINT" 3 \
    --experiment-label "exp1-$region_label" \
    --device cuda:0 \
    --height 192 \
    --width 256 \
    --num-tokens 1200 \
    --mask-dilation 0 \
    --yaw-min -90 \
    --yaw-max 90 \
    --orbit-frames 46 \
    --motion one-way \
    --frame-duration-ms 110
}

render_region towel-rack \
  "$EXP_DIR/rod_selection.json" \
  "$EXP_DIR/artifacts/rod_visualization"
render_region window \
  "$EXP_DIR/rod_selection_window.json" \
  "$EXP_DIR/artifacts/rod_visualization_window"
render_region faucet \
  "$EXP_DIR/rod_selection_faucet.json" \
  "$EXP_DIR/artifacts/rod_visualization_faucet"
