#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "用法：$0 <arm_id> <physical_gpu>" >&2
  exit 2
fi

ARM_ID="$1"
PHYSICAL_GPU="$2"
case "$ARM_ID" in
  arm_a_k1_edge1)
    TRAINING_K=1
    EDGE_WEIGHT=1
    ;;
  arm_b_k1_edge4)
    TRAINING_K=1
    EDGE_WEIGHT=4
    ;;
  arm_c_k3_edge1)
    TRAINING_K=3
    EDGE_WEIGHT=1
    ;;
  arm_d_k3_edge4)
    TRAINING_K=3
    EDGE_WEIGHT=4
    ;;
  *)
    echo "未知实验组：$ARM_ID" >&2
    exit 3
    ;;
esac
case "$PHYSICAL_GPU" in
  2|3) ;;
  *)
    echo "exp5 只允许使用物理 GPU 2 或 3" >&2
    exit 4
    ;;
esac

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage1_learning_capability/runs/exp5_ssr_refinement_stability_ablation"
DATA="$REPO/experiment/stage1_learning_capability/runs/exp4_hypersim_multiscene_generalization/data"
OUTPUT="$EXP_DIR/artifacts/runs/$ARM_ID"
LATEST="$OUTPUT/latest_metrics"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
case "$(realpath -m "$OUTPUT")" in
  "$SAFE_ROOT"/*) ;;
  *) echo "输出路径越过安全根目录" >&2; exit 91 ;;
esac
if [[ -e "$OUTPUT/report.json" ]]; then
  echo "实验组已有完整报告，拒绝覆盖：$OUTPUT" >&2
  exit 5
fi

install -d "$OUTPUT" "$LATEST" "$CACHE/huggingface" "$CACHE/torch" \
  "$CACHE/pip" "$CACHE/xdg" "$TMP"
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"

"$ENVIRONMENT/bin/python" -m moge.scripts.train_hypersim_generalization_v3 \
  --data "$DATA" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps "$TRAINING_K" \
  --evaluation-steps 0 1 3 5 \
  --steps 2000 \
  --batch-size 2 \
  --microbatch-size 1 \
  --learning-rate 2e-5 \
  --weight-decay 1e-2 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight "$EDGE_WEIGHT" \
  --local-scales 4 16 64 \
  --eval-every 250 \
  --log-every 50 \
  --periodic-train-samples 64 \
  --loss-smoothing-window 100 \
  --boundary-threshold 0.03 \
  --seed 47 \
  --diagnose-loss-gradients

"$ENVIRONMENT/bin/python" -m moge.scripts.evaluate_hypersim_checkpoint_v3 \
  --data "$DATA" \
  --checkpoint "$OUTPUT/latest_checkpoint.pt" \
  --output "$LATEST" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --evaluation-steps 0 1 3 5 \
  --batch-size 2 \
  --boundary-threshold 0.03
