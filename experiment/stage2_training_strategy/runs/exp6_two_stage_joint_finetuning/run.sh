#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp6_two_stage_joint_finetuning"
DATA="$REPO/experiment/stage1_learning_capability/runs/exp4_hypersim_multiscene_generalization/data"
BASELINE_REPORT="$REPO/experiment/stage1_learning_capability/runs/exp4_hypersim_multiscene_generalization/metrics/training_report.json"
SELECTION="$REPO/experiment/stage1_learning_capability/runs/exp4_hypersim_multiscene_generalization/artifacts/selection/fine_sample_selection.json"
TRAINING="$EXP_DIR/artifacts/training"
BEST="$EXP_DIR/artifacts/best_metrics"
LATEST="$EXP_DIR/artifacts/latest_metrics"
FINE="$EXP_DIR/artifacts/fine_structure"
SUMMARY="$EXP_DIR/artifacts/summary"
LOGS="$EXP_DIR/artifacts/logs"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
case "$(realpath -m "$EXP_DIR")" in
  "$SAFE_ROOT"/*) ;;
  *) echo "实验目录越过安全根目录" >&2; exit 91 ;;
esac
if [[ -e "$SUMMARY/report.json" ]]; then
  echo "exp6 已有正式汇总，拒绝覆盖：$SUMMARY/report.json" >&2
  exit 5
fi

install -d "$TRAINING" "$BEST" "$LATEST" "$FINE" "$SUMMARY" "$LOGS" \
  "$CACHE/huggingface" "$CACHE/torch" "$CACHE/pip" "$CACHE/xdg" "$TMP"
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=3

"$ENVIRONMENT/bin/python" -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$TRAINING" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --steps 2500 \
  --batch-size 2 \
  --microbatch-size 1 \
  --refiner-detach-steps 500 \
  --backbone-freeze-steps 100 \
  --backbone-warmup-end 200 \
  --ssr-learning-rate 2e-5 \
  --head-learning-rate 1e-5 \
  --backbone-learning-rate 5e-7 \
  --weight-decay 1e-2 \
  --gradient-clip-norm 1 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every 250 \
  --periodic-train-samples 64 \
  --eval-batch-size 2 \
  --log-every 25 \
  --loss-smoothing-window 100 \
  --boundary-threshold 0.03 \
  --seed 61 >"$LOGS/train.log" 2>&1

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

"$ENVIRONMENT/bin/python" tools/moge3/visualize_fine_structure.py \
  --data "$DATA" \
  --checkpoint "$TRAINING/checkpoint.pt" \
  --output "$FINE" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --max-k 5 \
  --crop-height 192 \
  --crop-width 192 \
  --crop-stride 32 \
  --min-fine-pixels 48 \
  --yaw 28 \
  --pitch -8 \
  --splits val test \
  --orbit-splits val test \
  --selection-file "$SELECTION" \
  --seed 61 >"$LOGS/fine_structure.log" 2>&1

"$ENVIRONMENT/bin/python" tools/moge3/summarize_joint_finetuning.py \
  --baseline-report "$BASELINE_REPORT" \
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
