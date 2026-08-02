#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage1_learning_capability/runs/exp2_hypersim_smallset_overfit"
DATA="$EXP_DIR/data"
SELECTION="$EXP_DIR/train_rod_gallery_selection.json"
CHECKPOINT="$EXP_DIR/artifacts/checkpoint.pt"
OUTPUT="$EXP_DIR/artifacts/train_rod_gallery"
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
test ! -e "$OUTPUT/gallery_report.json"
install -d "$OUTPUT/logs" "$CACHE/huggingface" "$CACHE/torch" "$CACHE/pip" \
  "$CACHE/xdg" "$TMP"

cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1

GPUS=(3 1 2 0)
PIDS=()
for SHARD in 0 1 2 3; do
  GPU="${GPUS[$SHARD]}"
  CUDA_VISIBLE_DEVICES="$GPU" "$ENVIRONMENT/bin/python" \
    tools/moge3/render_rod_gallery.py \
      --data "$DATA" \
      --selection "$SELECTION" \
      --output "$OUTPUT" \
      --checkpoint "$CHECKPOINT" \
      --safe-root "$SAFE_ROOT" \
      --device cuda:0 \
      --height 192 \
      --width 256 \
      --num-tokens 1200 \
      --shard-index "$SHARD" \
      --num-shards 4 \
      --mask-dilation 0 \
      --yaw-min -90 \
      --yaw-max 90 \
      --orbit-frames 46 \
      --frame-duration-ms 110 \
      >"$OUTPUT/logs/shard_${SHARD}_gpu_${GPU}.log" 2>&1 &
  PIDS+=("$!")
done

FAILED=0
for PID in "${PIDS[@]}"; do
  if ! wait "$PID"; then
    FAILED=1
  fi
done
test "$FAILED" -eq 0

"$ENVIRONMENT/bin/python" tools/moge3/render_rod_gallery.py \
  --data "$DATA" \
  --selection "$SELECTION" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --safe-root "$SAFE_ROOT" \
  --height 192 \
  --width 256 \
  --aggregate-only \
  >"$OUTPUT/logs/aggregate.log" 2>&1

echo "24 张训练图细杆 GIF 已生成：$OUTPUT"
