#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
SOURCE=/nas1/datasets/hypersim/raw
EXP_DIR="$REPO/experiment/stage1_learning_capability/runs/exp4_hypersim_multiscene_generalization"
DATA="$EXP_DIR/data"
ARTIFACTS="$EXP_DIR/artifacts"
SELECTION="$ARTIFACTS/selection"
TRAINING="$ARTIFACTS/training_final"
FINE="$ARTIFACTS/fine_structure"
METRICS="$EXP_DIR/metrics"
LATEST_METRICS="$METRICS/latest_checkpoint"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
install -d \
  "$DATA" \
  "$SELECTION" \
  "$TRAINING" \
  "$FINE" \
  "$METRICS" \
  "$LATEST_METRICS" \
  "$CACHE/huggingface" \
  "$CACHE/torch" \
  "$CACHE/pip" \
  "$CACHE/xdg" \
  "$TMP"

cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES=2

"$ENVIRONMENT/bin/python" tools/moge3/prepare_hypersim_generalization.py \
  --spec "$EXP_DIR/data_spec.json" \
  --source-root "$SOURCE" \
  --output "$DATA" \
  --safe-root "$SAFE_ROOT"

"$ENVIRONMENT/bin/python" tools/moge3/select_hypersim_fine_samples.py \
  --data "$DATA" \
  --output "$SELECTION" \
  --safe-root "$SAFE_ROOT" \
  --height 384 \
  --width 512 \
  --crop-height 192 \
  --crop-width 192 \
  --crop-stride 32 \
  --minimum-fine-pixels 96 \
  --val-sample-id ai_055_003_cam_00_frame.0013 \
  --test-sample-id ai_046_001_cam_00_frame.0079

"$ENVIRONMENT/bin/python" -m moge.scripts.train_hypersim_generalization_v3 \
  --data "$DATA" \
  --output "$TRAINING" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --evaluation-steps 0 1 3 5 \
  --steps 8000 \
  --batch-size 2 \
  --microbatch-size 1 \
  --learning-rate 2e-5 \
  --weight-decay 1e-2 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every 500 \
  --log-every 50 \
  --periodic-train-samples 64 \
  --loss-smoothing-window 100 \
  --boundary-threshold 0.03 \
  --seed 47 \
  --diagnose-loss-gradients

install -m 0644 "$TRAINING/report.json" "$METRICS/training_report.json"

"$ENVIRONMENT/bin/python" -m moge.scripts.evaluate_hypersim_checkpoint_v3 \
  --data "$DATA" \
  --checkpoint "$TRAINING/latest_checkpoint.pt" \
  --output "$LATEST_METRICS" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --evaluation-steps 0 1 3 5 \
  --batch-size 2 \
  --boundary-threshold 0.03

"$ENVIRONMENT/bin/python" tools/moge3/visualize_fine_structure.py \
  --data "$DATA" \
  --checkpoint "$TRAINING/checkpoint.pt" \
  --output "$FINE/best" \
  --selection-file "$SELECTION/fine_sample_selection.json" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --max-k 5 \
  --crop-height 192 \
  --crop-width 192 \
  --crop-stride 32 \
  --min-fine-pixels 96 \
  --yaw 28 \
  --pitch -8 \
  --splits val test \
  --orbit-splits val test \
  --seed 53

"$ENVIRONMENT/bin/python" tools/moge3/visualize_fine_structure.py \
  --data "$DATA" \
  --checkpoint "$TRAINING/latest_checkpoint.pt" \
  --output "$FINE/latest" \
  --selection-file "$SELECTION/fine_sample_selection.json" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --max-k 5 \
  --crop-height 192 \
  --crop-width 192 \
  --crop-stride 32 \
  --min-fine-pixels 96 \
  --yaw 28 \
  --pitch -8 \
  --splits val test \
  --orbit-splits val test \
  --seed 53
