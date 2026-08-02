#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp8_two_gpu_head_ssr_joint_finetuning"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/data"
BASELINE="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/artifacts/train_best_evaluation/report.json"
TRAINING="$EXP_DIR/artifacts/training"
BEST="$EXP_DIR/artifacts/best_metrics"
LATEST="$EXP_DIR/artifacts/latest_metrics"
SUMMARY="$EXP_DIR/artifacts/summary"
LOGS="$EXP_DIR/artifacts/logs"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU_PAIR="${MOGE3_GPUS:-0,1}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$TRAINING" "$BEST" "$LATEST" "$SUMMARY" "$LOGS"; do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "实验目录越过安全根目录：$path" >&2; exit 91 ;;
  esac
done
case "$GPU_PAIR" in
  0,1|0,2|0,3|1,0|1,2|1,3|2,0|2,1|2,3|3,0|3,1|3,2) ;;
  *) echo "GPU组合必须是两张不同的0至3号GPU" >&2; exit 92 ;;
esac
if [[ -e "$TRAINING/report.json" || -e "$SUMMARY/report.json" ]]; then
  echo "exp8正式结果已存在，拒绝覆盖" >&2
  exit 5
fi

install -d "$TRAINING" "$BEST" "$LATEST" "$SUMMARY" "$LOGS" \
  "$CACHE/huggingface" "$CACHE/torch" "$CACHE/pip" "$CACHE/xdg" "$TMP"
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
  --output "$TRAINING" \
  --safe-root "$SAFE_ROOT" \
  --ddp-backend gloo \
  --freeze-backbone \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --steps 600 \
  --batch-size 2 \
  --microbatch-size 1 \
  --refiner-detach-steps 300 \
  --backbone-freeze-steps 0 \
  --backbone-warmup-end 1 \
  --ssr-learning-rate 2e-5 \
  --head-learning-rate 1e-6 \
  --backbone-learning-rate 0 \
  --weight-decay 1e-2 \
  --gradient-clip-norm 1 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every 50 \
  --periodic-train-samples 24 \
  --eval-batch-size 2 \
  --log-every 10 \
  --loss-smoothing-window 50 \
  --boundary-threshold 0.03 \
  --seed 81 >"$LOGS/train.log" 2>&1

export CUDA_VISIBLE_DEVICES="${GPU_PAIR%%,*}"

"$ENVIRONMENT/bin/python" -m moge.scripts.evaluate_hypersim_checkpoint_v3 \
  --data "$DATA" \
  --checkpoint "$TRAINING/checkpoint.pt" \
  --output "$BEST" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --evaluation-steps 0 1 3 5 \
  --batch-size 2 \
  --boundary-threshold 0.03 >"$LOGS/evaluate_best.log" 2>&1

"$ENVIRONMENT/bin/python" -m moge.scripts.evaluate_hypersim_checkpoint_v3 \
  --data "$DATA" \
  --checkpoint "$TRAINING/latest_checkpoint.pt" \
  --output "$LATEST" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --evaluation-steps 0 1 3 5 \
  --batch-size 2 \
  --boundary-threshold 0.03 >"$LOGS/evaluate_latest.log" 2>&1

"$ENVIRONMENT/bin/python" tools/moge3/summarize_joint_finetuning.py \
  --experiment exp8_two_gpu_head_ssr_joint_finetuning \
  --baseline-report "$BASELINE" \
  --training-report "$TRAINING/report.json" \
  --best-report "$BEST/report.json" \
  --latest-report "$LATEST/report.json" \
  --output "$SUMMARY" \
  --safe-root "$SAFE_ROOT" \
  --maximum-boundary-drop 0.005 >"$LOGS/summary.log" 2>&1

"$ENVIRONMENT/bin/python" -m pytest -q >"$EXP_DIR/artifacts/test_results.txt"

find "$TRAINING" -maxdepth 1 -type f \
  \( -name 'checkpoint.pt' -o -name 'latest_checkpoint.pt' -o -name 'resume_checkpoint.pt' \) \
  -print0 | sort -z | xargs -0 sha256sum >"$EXP_DIR/artifacts/checkpoint_sha256.txt"
