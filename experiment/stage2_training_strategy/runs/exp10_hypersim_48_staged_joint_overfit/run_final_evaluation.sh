#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp10_hypersim_48_staged_joint_overfit"
DATA="$EXP_DIR/data"
TRAINING="$EXP_DIR/artifacts/adaptive_joint_lr5e7_from_best3500"
CHECKPOINT="$TRAINING/checkpoint.pt"
OUTPUT="$EXP_DIR/artifacts/final_evaluation_best"
LOGS="$EXP_DIR/artifacts/logs"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU_INDEX="${MOGE3_GPU:?MOGE3_GPU must be one GPU index}"
MIN_FREE_MIB="${MOGE3_MIN_FREE_MIB:-20000}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$TRAINING" "$CHECKPOINT" "$OUTPUT" \
  "$LOGS" "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done
case "$GPU_INDEX" in
  0|1|2|3) ;;
  *) echo "GPU编号必须是0至3" >&2; exit 92 ;;
esac

test -f "$DATA/manifest.json"
test -f "$CHECKPOINT"
if [[ -f "$OUTPUT/report.json" ]]; then
  echo "最终评测已经完成，拒绝覆盖" >&2
  exit 5
fi
if pgrep -af "moge.scripts.train_hypersim_joint_v3.*$TRAINING" \
  | grep -v "$$" >/dev/null
then
  echo "训练仍在运行，拒绝提前评测可变检查点" >&2
  exit 7
fi

read -r free_mib < <(
  nvidia-smi --id="$GPU_INDEX" \
    --query-gpu=memory.free \
    --format=csv,noheader,nounits |
    awk '{gsub(/ /, "", $1); print $1}'
)
if (( free_mib < MIN_FREE_MIB )); then
  echo "GPU $GPU_INDEX 空闲显存不足：${free_mib}MiB" >&2
  exit 8
fi

install -d "$OUTPUT" "$LOGS" "$CACHE/huggingface" "$CACHE/torch" \
  "$CACHE/pip" "$CACHE/xdg" "$TMP"
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU_INDEX"

"$ENVIRONMENT/bin/python" -u \
  -m moge.scripts.evaluate_hypersim_checkpoint_v3 \
  --data "$DATA" \
  --checkpoint "$CHECKPOINT" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --evaluation-steps 0 1 3 5 \
  --batch-size 1 \
  --boundary-threshold 0.03 \
  >>"$LOGS/final_evaluation.log" 2>&1

sha256sum "$CHECKPOINT" "$OUTPUT/report.json" \
  "$OUTPUT/per_frame_metrics.csv" >"$OUTPUT/sha256sums.txt"
