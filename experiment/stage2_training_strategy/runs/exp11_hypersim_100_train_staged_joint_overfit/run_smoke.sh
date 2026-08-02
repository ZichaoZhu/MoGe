#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit"
DATA="$EXP_DIR/data"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU_LIST="${1:?逗号分隔的GPU编号不能为空}"

IFS=',' read -r -a GPU_INDICES <<<"$GPU_LIST"
WORLD_SIZE="${#GPU_INDICES[@]}"
if (( WORLD_SIZE < 1 || WORLD_SIZE > 4 )); then
  echo "GPU数量必须为1至4" >&2
  exit 90
fi
declare -A SEEN=()
for gpu in "${GPU_INDICES[@]}"; do
  case "$gpu" in 0|1|2|3) ;; *) echo "GPU编号必须是0至3" >&2; exit 92 ;; esac
  if [[ -n "${SEEN[$gpu]:-}" ]]; then
    echo "GPU编号不能重复" >&2
    exit 93
  fi
  SEEN[$gpu]=1
done
GPU_TAG="${GPU_LIST//,/_}"
OUTPUT="$EXP_DIR/artifacts/smoke_w${WORLD_SIZE}_g${GPU_TAG}"
LOGS="$EXP_DIR/artifacts/logs"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$OUTPUT" "$LOGS" "$CACHE" "$TMP"; do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done
test -f "$DATA/manifest.json"
if [[ -f "$OUTPUT/report.json" ]]; then
  echo "该GPU组合的Exp11冒烟测试已经通过"
  exit 0
fi
if [[ -e "$OUTPUT/resume_checkpoint.pt" ]]; then
  echo "发现未完成的同名冒烟目录，拒绝覆盖：$OUTPUT" >&2
  exit 5
fi

case "$WORLD_SIZE" in
  1|2|4) GLOBAL_BATCH=8 ;;
  3) GLOBAL_BATCH=6 ;;
esac

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
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

"$ENVIRONMENT/bin/torchrun" --standalone --nproc-per-node="$WORLD_SIZE" \
  -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --ddp-backend gloo \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --steps 12 \
  --batch-size "$GLOBAL_BATCH" \
  --microbatch-size 1 \
  --refiner-detach-steps 10 \
  --backbone-freeze-steps 2 \
  --backbone-warmup-end 4 \
  --ssr-learning-rate 1e-6 \
  --head-learning-rate 1e-6 \
  --backbone-learning-rate 1e-7 \
  --weight-decay 1e-2 \
  --gradient-clip-norm 1 \
  --max-preclip-grad-norm 100 \
  --max-skipped-preclip-steps 5 \
  --max-consecutive-skipped-preclip-steps 2 \
  --max-abs-log-depth-residual 5 \
  --max-refined-point-rel 0.1 \
  --max-refined-to-base-ratio 3 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every 6 \
  --periodic-train-samples 2 \
  --selection-split train \
  --eval-batch-size 1 \
  --log-every 1 \
  --loss-smoothing-window 4 \
  --boundary-threshold 0.03 \
  --max-samples-per-split 2 \
  --seed 111 >>"$LOGS/smoke_w${WORLD_SIZE}_g${GPU_TAG}.log" 2>&1
