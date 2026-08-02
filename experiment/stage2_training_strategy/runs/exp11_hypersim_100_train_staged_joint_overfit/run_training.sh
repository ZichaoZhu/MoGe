#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit"
DATA="$EXP_DIR/data"
OUTPUT="$EXP_DIR/artifacts/training"
LOGS="$EXP_DIR/artifacts/logs"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU_LIST="${1:?逗号分隔的GPU编号不能为空}"
TOTAL_STEPS="${MOGE3_STEPS:-10000}"

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
case "$WORLD_SIZE" in
  1|2|4) GLOBAL_BATCH=8 ;;
  3) GLOBAL_BATCH=6 ;;
esac

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
  echo "Exp11正式训练已经完成，拒绝覆盖"
  exit 0
fi
if pgrep -af "moge.scripts.train_hypersim_joint_v3.*$OUTPUT" \
  | grep -v "$$" >/dev/null
then
  echo "Exp11正式训练进程已经存在，拒绝重复启动" >&2
  exit 6
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
export CUDA_VISIBLE_DEVICES="$GPU_LIST"

resume_args=()
if [[ -f "$OUTPUT/resume_checkpoint.pt" ]]; then
  resume_args=(--resume "$OUTPUT/resume_checkpoint.pt")
fi

"$ENVIRONMENT/bin/torchrun" --standalone --nproc-per-node="$WORLD_SIZE" \
  -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$OUTPUT" \
  "${resume_args[@]}" \
  --safe-root "$SAFE_ROOT" \
  --ddp-backend gloo \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --steps "$TOTAL_STEPS" \
  --batch-size "$GLOBAL_BATCH" \
  --microbatch-size 1 \
  --refiner-detach-steps 5000 \
  --backbone-freeze-steps 1000 \
  --backbone-warmup-end 2000 \
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
  --eval-every 250 \
  --periodic-train-samples 100 \
  --selection-split train \
  --eval-batch-size 1 \
  --log-every 10 \
  --loss-smoothing-window 100 \
  --boundary-threshold 0.03 \
  --seed 111 >>"$LOGS/training.log" 2>&1
