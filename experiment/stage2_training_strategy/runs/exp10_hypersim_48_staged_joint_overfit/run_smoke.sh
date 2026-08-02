#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp10_hypersim_48_staged_joint_overfit"
DATA="$EXP_DIR/data"
SMOKE="$EXP_DIR/artifacts/smoke"
LOGS="$EXP_DIR/artifacts/logs"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU_PAIR="${MOGE3_GPUS:?MOGE3_GPUS must contain two GPU indices}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$SMOKE" "$LOGS" "$CACHE" "$TMP"; do
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
if [[ -f "$SMOKE/report.json" ]]; then
  echo "Exp10两卡冒烟测试已经通过，跳过重复执行"
  exit 0
fi

install -d "$SMOKE" "$LOGS" "$CACHE/huggingface" "$CACHE/torch" \
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

"$ENVIRONMENT/bin/torchrun" --standalone --nproc-per-node=2 \
  -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$SMOKE" \
  --safe-root "$SAFE_ROOT" \
  --ddp-backend gloo \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --steps 20 \
  --batch-size 2 \
  --microbatch-size 1 \
  --refiner-detach-steps 10 \
  --backbone-freeze-steps 2 \
  --backbone-warmup-end 4 \
  --ssr-learning-rate 2e-5 \
  --head-learning-rate 1e-6 \
  --backbone-learning-rate 1e-7 \
  --weight-decay 1e-2 \
  --gradient-clip-norm 1 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every 10 \
  --periodic-train-samples 2 \
  --selection-split train \
  --eval-batch-size 2 \
  --log-every 1 \
  --loss-smoothing-window 4 \
  --boundary-threshold 0.03 \
  --max-samples-per-split 2 \
  --seed 101 >>"$LOGS/smoke.log" 2>&1
