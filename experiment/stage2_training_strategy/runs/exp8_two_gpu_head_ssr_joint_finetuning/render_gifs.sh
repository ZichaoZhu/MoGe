#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp8_two_gpu_head_ssr_joint_finetuning"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/data"
SELECTION="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/gif_selection.json"
CHECKPOINT="$EXP_DIR/artifacts/training/latest_checkpoint.pt"
OUTPUT="$EXP_DIR/artifacts/gif_gallery"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU_PAIR="${MOGE3_GPUS:-0,2}"
GPU_A="${GPU_PAIR%%,*}"
GPU_B="${GPU_PAIR##*,}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
case "$(realpath -m "$OUTPUT")/" in
  "$SAFE_ROOT/"*) ;;
  *) echo "GIF目录越过安全根目录" >&2; exit 91 ;;
esac
case "$GPU_PAIR" in
  0,1|0,2|0,3|1,0|1,2|1,3|2,0|2,1|2,3|3,0|3,1|3,2) ;;
  *) echo "GPU组合必须是两张不同的0至3号GPU" >&2; exit 92 ;;
esac
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
    --num-shards 2 \
    --mask-dilation 0 \
    --yaw-min -90 \
    --yaw-max 90 \
    --orbit-frames 46 \
    --frame-duration-ms 110 \
    --experiment-label exp8_two_gpu_head_ssr_joint_finetuning
}

render_shard 0 "$GPU_A" &
pid_a=$!
render_shard 1 "$GPU_B" &
pid_b=$!
wait "$pid_a"
wait "$pid_b"

"$ENVIRONMENT/bin/python" tools/moge3/render_rod_gallery.py \
  --data "$DATA" \
  --selection "$SELECTION" \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --safe-root "$SAFE_ROOT" \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --experiment-label exp8_two_gpu_head_ssr_joint_finetuning \
  --aggregate-only
