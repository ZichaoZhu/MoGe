#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage1_learning_capability/runs/exp2_hypersim_smallset_overfit"
OUTPUT="$EXP_DIR/artifacts/orbits"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
case "$(realpath -m "$OUTPUT")" in
  "$SAFE_ROOT"/*) ;;
  *) echo "输出目录越过安全根目录" >&2; exit 91 ;;
esac
if [[ -e "$OUTPUT/report.json" ]]; then
  echo "exp2 旋转动画已经生成，拒绝覆盖：$OUTPUT/report.json" >&2
  exit 5
fi

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
export CUDA_VISIBLE_DEVICES=3

"$ENVIRONMENT/bin/python" tools/moge3/visualize_fine_structure.py \
  --data "$EXP_DIR/data" \
  --checkpoint "$EXP_DIR/artifacts/checkpoint.pt" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 192 \
  --width 256 \
  --num-tokens 1200 \
  --max-k 5 \
  --crop-height 96 \
  --crop-width 96 \
  --crop-stride 16 \
  --min-fine-pixels 24 \
  --yaw 28 \
  --pitch -8 \
  --splits train val \
  --orbit-splits train val \
  --train-sample-id ai_036_002_cam_00_frame.0042 \
  --val-sample-id ai_055_003_cam_00_frame.0057 \
  --seed 29 >"$EXP_DIR/artifacts/orbit_generation.log" 2>&1
