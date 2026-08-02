#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp12_hypersim_100_immediate_joint_finetuning"
EXP11_DIR="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit"
DATA="$EXP11_DIR/data"
WARM_START="$EXP11_DIR/artifacts/training/checkpoint.pt"
EXPECTED_WARM_START_SHA=1a665e760516fde4e00082b9743a0fd8c7eed98d298cb76bb012b813227c98ce
LOGS="$EXP_DIR/artifacts/logs"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU_LIST="${1:?逗号分隔的GPU编号不能为空}"
RUN_MODE="${MOGE3_RUN_MODE:-formal}"

IFS=',' read -r -a GPU_INDICES <<<"$GPU_LIST"
WORLD_SIZE="${#GPU_INDICES[@]}"
if (( WORLD_SIZE < 1 || WORLD_SIZE > 2 )); then
  echo "Exp12仅允许1至2张GPU" >&2
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

case "$RUN_MODE" in
  smoke)
    OUTPUT="$EXP_DIR/artifacts/smoke_joint_from_exp11_best3000"
    TOTAL_STEPS=3004
    EVAL_EVERY=4
    PERIODIC_TRAIN_SAMPLES=4
    MAX_SAMPLE_ARGS=(--max-samples-per-split 4)
    ;;
  formal)
    OUTPUT="$EXP_DIR/artifacts/joint_from_exp11_best3000_cosine"
    TOTAL_STEPS=5000
    EVAL_EVERY=100
    PERIODIC_TRAIN_SAMPLES=100
    MAX_SAMPLE_ARGS=()
    ;;
  *)
    echo "MOGE3_RUN_MODE必须是smoke或formal" >&2
    exit 94
    ;;
esac

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$EXP11_DIR" "$DATA" "$WARM_START" "$OUTPUT" \
  "$LOGS" "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done
test -f "$DATA/manifest.json"
test -f "$WARM_START"
actual_sha="$(sha256sum "$WARM_START" | awk '{print $1}')"
if [[ "$actual_sha" != "$EXPECTED_WARM_START_SHA" ]]; then
  echo "Exp11最佳检查点SHA-256不匹配：$actual_sha" >&2
  exit 95
fi
if [[ -f "$OUTPUT/report.json" ]]; then
  echo "$RUN_MODE 已完成，拒绝覆盖"
  exit 0
fi
if pgrep -af "moge.scripts.train_hypersim_joint_v3.*$OUTPUT" \
  | grep -v "$$" >/dev/null
then
  echo "$RUN_MODE 训练进程已存在，拒绝重复启动" >&2
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

checkpoint_args=(--warm-start "$WARM_START")
if [[ -f "$OUTPUT/resume_checkpoint.pt" ]]; then
  checkpoint_args=(--resume "$OUTPUT/resume_checkpoint.pt")
fi

"$ENVIRONMENT/bin/torchrun" --standalone --nproc-per-node="$WORLD_SIZE" \
  -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$OUTPUT" \
  "${checkpoint_args[@]}" \
  --safe-root "$SAFE_ROOT" \
  --ddp-backend gloo \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --steps "$TOTAL_STEPS" \
  --batch-size 16 \
  --microbatch-size 1 \
  --refiner-detach-steps 3000 \
  --backbone-freeze-steps 1000 \
  --backbone-warmup-end 2000 \
  --ssr-learning-rate 5e-7 \
  --head-learning-rate 2e-7 \
  --backbone-learning-rate 1e-8 \
  --learning-rate-schedule cosine \
  --learning-rate-decay-start-step 3000 \
  --learning-rate-decay-end-step 5000 \
  --learning-rate-final-scale 0.1 \
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
  --eval-every "$EVAL_EVERY" \
  --periodic-train-samples "$PERIODIC_TRAIN_SAMPLES" \
  --selection-split train \
  --eval-batch-size 1 \
  --log-every 10 \
  --loss-smoothing-window 100 \
  --boundary-threshold 0.03 \
  --seed 121 \
  "${MAX_SAMPLE_ARGS[@]}" >>"$LOGS/${RUN_MODE}.log" 2>&1
