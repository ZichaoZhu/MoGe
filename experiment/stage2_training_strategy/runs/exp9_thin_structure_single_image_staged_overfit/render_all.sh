#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp9_thin_structure_single_image_staged_overfit"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/data"
SELECTION="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/gif_selection.json"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
PHYSICAL_GPU="${MOGE3_RENDER_GPU:-0}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
case "$PHYSICAL_GPU" in
  0|1|2|3) ;;
  *) echo "GPU必须是0至3中的一个" >&2; exit 92 ;;
esac
case "$(realpath -m "$EXP_DIR")/" in
  "$SAFE_ROOT/"*) ;;
  *) echo "实验目录越过安全根目录" >&2; exit 91 ;;
esac

install -d "$CACHE/huggingface" "$CACHE/torch" "$CACHE/xdg" "$TMP"
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"

for sample_dir in "$EXP_DIR"/0*; do
  test -f "$sample_dir/report.json"
  sample_id="$(
    "$ENVIRONMENT/bin/python" -c \
      'import json,sys; print(json.load(open(sys.argv[1]))["sample_id"])' \
      "$sample_dir/config.json"
  )"
  "$ENVIRONMENT/bin/python" tools/moge3/visualize_staged_single_overfit.py \
    --data "$DATA" \
    --selection "$SELECTION" \
    --sample-id "$sample_id" \
    --experiment-output "$sample_dir" \
    --safe-root "$SAFE_ROOT" \
    --device cuda:0 \
    --yaw-min -90 \
    --yaw-max 90 \
    --orbit-frames 46 \
    --frame-duration-ms 110
done

"$ENVIRONMENT/bin/python" tools/moge3/summarize_staged_single_overfit.py \
  --experiment-root "$EXP_DIR" \
  --hash-checkpoints
