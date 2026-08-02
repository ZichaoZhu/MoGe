#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP4="$REPO/experiment/stage1_learning_capability/runs/exp4_hypersim_multiscene_generalization"
EXP_DIR="$REPO/experiment/stage1_learning_capability/runs/exp5_ssr_refinement_stability_ablation"
DATA="$EXP4/data"
SELECTION="$EXP4/rod_selection.json"
RUNS="$EXP_DIR/artifacts/runs"
OUTPUT_K1="$EXP_DIR/artifacts/rod_visualization_k1"
OUTPUT_K3="$EXP_DIR/artifacts/rod_visualization_k3"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
test "$(realpath -e "$DATA")" = "$DATA"
test "$(realpath -e "$SELECTION")" = "$SELECTION"
for checkpoint_file in \
  "$RUNS/arm_a_k1_edge1/latest_checkpoint.pt" \
  "$RUNS/arm_b_k1_edge4/latest_checkpoint.pt" \
  "$RUNS/arm_c_k3_edge1/latest_checkpoint.pt" \
  "$RUNS/arm_d_k3_edge4/latest_checkpoint.pt"; do
  test "$(realpath -e "$checkpoint_file")" = "$checkpoint_file"
done
for output_dir in "$OUTPUT_K1" "$OUTPUT_K3"; do
  case "$(realpath -m "$output_dir")" in
    "$SAFE_ROOT"/*) ;;
    *) echo "输出目录越过安全根目录" >&2; exit 91 ;;
  esac
  test ! -e "$output_dir/report.json"
done
install -d "$OUTPUT_K1" "$OUTPUT_K3" "$CACHE/huggingface" "$CACHE/torch" \
  "$CACHE/pip" "$CACHE/xdg" "$TMP"

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
  --output "$OUTPUT_K1" \
  --safe-root "$SAFE_ROOT" \
  --series A-K1-edge1 "$RUNS/arm_a_k1_edge1/latest_checkpoint.pt" 1 \
  --series B-K1-edge4 "$RUNS/arm_b_k1_edge4/latest_checkpoint.pt" 1 \
  --experiment-label exp5-K1 \
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

"$ENVIRONMENT/bin/python" tools/moge3/visualize_split_rods.py \
  --data "$DATA" \
  --selection "$SELECTION" \
  --output "$OUTPUT_K3" \
  --safe-root "$SAFE_ROOT" \
  --series C-K3-edge1 "$RUNS/arm_c_k3_edge1/latest_checkpoint.pt" 3 \
  --series D-K3-edge4 "$RUNS/arm_d_k3_edge4/latest_checkpoint.pt" 3 \
  --experiment-label exp5-K3 \
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
