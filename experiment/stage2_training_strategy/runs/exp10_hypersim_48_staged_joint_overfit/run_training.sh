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
GPU_PAIR="${MOGE3_GPUS:?MOGE3_GPUS must contain two GPU indices}"
TOTAL_STEPS="${MOGE3_STEPS:-10000}"
DETACH_STEPS="${MOGE3_DETACH_STEPS:-5000}"
EVAL_EVERY="${MOGE3_EVAL_EVERY:-250}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$TRAINING" "$LOGS" "$CACHE" "$TMP"; do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done
case "$GPU_PAIR" in
  0,1|0,2|0,3|1,0|1,2|1,3|2,0|2,1|2,3|3,0|3,1|3,2) ;;
  *) echo "GPU组合必须是两张不同的0至3号GPU" >&2; exit 92 ;;
esac
test -f "$DATA/manifest.json"
if [[ -f "$TRAINING/report.json" ]]; then
  echo "Exp10正式训练结果已经存在，拒绝覆盖" >&2
  exit 5
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
export CUDA_VISIBLE_DEVICES="$GPU_PAIR"

resume_args=()
if [[ -f "$TRAINING/resume_checkpoint.pt" ]]; then
  resume_args=(--resume "$TRAINING/resume_checkpoint.pt")
fi

"$ENVIRONMENT/bin/torchrun" --standalone --nproc-per-node=2 \
  -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$TRAINING" \
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
  --seed 101 \
  "${resume_args[@]}" >>"$LOGS/train.log" 2>&1
