#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp10_hypersim_48_staged_joint_overfit"
DATA="$EXP_DIR/data"
TRAINING="$EXP_DIR/artifacts/training"
LOGS="$EXP_DIR/artifacts/logs"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU_INDEX="${MOGE3_GPU:-3}"
TOTAL_STEPS="${MOGE3_STEPS:-10000}"
DETACH_STEPS="${MOGE3_DETACH_STEPS:-5000}"
EVAL_EVERY="${MOGE3_EVAL_EVERY:-250}"
MIN_FREE_MIB="${MOGE3_MIN_FREE_MIB:-22000}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$TRAINING" "$LOGS" "$CACHE" "$TMP"; do
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
test -f "$TRAINING/resume_checkpoint.pt"
if [[ -f "$TRAINING/report.json" ]]; then
  echo "Exp10正式训练结果已经存在，拒绝覆盖" >&2
  exit 5
fi
if pgrep -af "moge.scripts.train_hypersim_joint_v3.*$TRAINING" \
  | grep -v "$$" >/dev/null; then
  echo "Exp10训练进程已经存在，拒绝重复启动" >&2
  exit 6
fi

read -r free_mib utilization < <(
  nvidia-smi --id="$GPU_INDEX" \
    --query-gpu=memory.free,utilization.gpu \
    --format=csv,noheader,nounits |
    awk -F',' '{gsub(/ /, "", $1); gsub(/ /, "", $2); print $1, $2}'
)
if (( free_mib < MIN_FREE_MIB || utilization > 10 )); then
  echo "GPU $GPU_INDEX 当前不满足启动条件：空闲${free_mib}MiB，利用率${utilization}%" >&2
  exit 7
fi

install -d "$TRAINING" "$LOGS" "$CACHE/huggingface" "$CACHE/torch" \
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

"$ENVIRONMENT/bin/torchrun" --standalone --nproc-per-node=1 \
  -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$TRAINING" \
  --resume "$TRAINING/resume_checkpoint.pt" \
  --safe-root "$SAFE_ROOT" \
  --ddp-backend gloo \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --steps "$TOTAL_STEPS" \
  --batch-size 2 \
  --microbatch-size 1 \
  --refiner-detach-steps "$DETACH_STEPS" \
  --backbone-freeze-steps 1000 \
  --backbone-warmup-end 2000 \
  --ssr-learning-rate 2e-5 \
  --head-learning-rate 1e-6 \
  --backbone-learning-rate 1e-7 \
  --weight-decay 1e-2 \
  --gradient-clip-norm 1 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every "$EVAL_EVERY" \
  --periodic-train-samples 48 \
  --selection-split train \
  --eval-batch-size 2 \
  --log-every 10 \
  --loss-smoothing-window 100 \
  --boundary-threshold 0.03 \
  --seed 101 >>"$LOGS/train_single_gpu.log" 2>&1
