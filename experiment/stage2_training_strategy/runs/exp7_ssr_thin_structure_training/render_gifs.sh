#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training"
DATA="$EXP_DIR/data"
SELECTION="$EXP_DIR/gif_selection.json"
CHECKPOINT="$EXP_DIR/artifacts/train_best_tracking/train_best_checkpoint.pt"
OUTPUT="$EXP_DIR/artifacts/gif_gallery"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for writable_path in "$EXP_DIR" "$OUTPUT" "$CACHE" "$TMP"; do
  case "$writable_path/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "拒绝越界写入：$writable_path" >&2; exit 91 ;;
  esac
done
test -f "$CHECKPOINT"
install -d "$OUTPUT" "$CACHE/huggingface" "$CACHE/torch" "$CACHE/xdg" "$TMP"

cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

render_shard() {
  local shard_index="$1"
  local physical_gpu="$2"
  CUDA_VISIBLE_DEVICES="$physical_gpu" "$ENVIRONMENT/bin/python" \
    tools/moge3/render_rod_gallery.py \
    --data "$DATA" \
    --selection "$SELECTION" \
    --output "$OUTPUT" \
    --checkpoint "$CHECKPOINT" \
    --safe-root "$SAFE_ROOT" \
    --device cuda:0 \
    --height 384 \
    --width 512 \
    --num-tokens 2500 \
    --shard-index "$shard_index" \
    --num-shards 4 \
    --mask-dilation 0 \
    --yaw-min -90 \
    --yaw-max 90 \
    --orbit-frames 46 \
    --frame-duration-ms 110 \
    --experiment-label exp7_ssr_thin_structure_training
}

render_shard 0 0 &
pid_0=$!
render_shard 1 1 &
pid_1=$!
render_shard 2 2 &
pid_2=$!
render_shard 3 3 &
pid_3=$!
wait "$pid_0"
wait "$pid_1"
wait "$pid_2"
wait "$pid_3"

"$ENVIRONMENT/bin/python" tools/moge3/render_rod_gallery.py \
  --data "$DATA" \
  --selection "$SELECTION" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --safe-root "$SAFE_ROOT" \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --experiment-label exp7_ssr_thin_structure_training \
  --aggregate-only
