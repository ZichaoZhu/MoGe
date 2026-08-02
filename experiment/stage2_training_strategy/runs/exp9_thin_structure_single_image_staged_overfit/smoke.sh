#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp9_thin_structure_single_image_staged_overfit"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/data"
SELECTION="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/gif_selection.json"
OUTPUT="$EXP_DIR/smoke"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
PHYSICAL_GPU="${MOGE3_SMOKE_GPU:-0}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
case "$(realpath -m "$OUTPUT")/" in
  "$SAFE_ROOT/"*) ;;
  *) echo "冒烟测试目录越过安全根目录" >&2; exit 91 ;;
esac
case "$PHYSICAL_GPU" in
  0|1|2|3) ;;
  *) echo "GPU必须是0至3中的一个" >&2; exit 92 ;;
esac
if [[ -e "$OUTPUT/report.json" ]]; then
  echo "冒烟测试已完成，拒绝覆盖：$OUTPUT/report.json" >&2
  exit 5
fi

install -d "$OUTPUT" "$CACHE/huggingface" "$CACHE/torch" "$CACHE/pip" \
  "$CACHE/xdg" "$TMP"
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$PHYSICAL_GPU"

resume_args=()
if [[ -f "$OUTPUT/checkpoints/resume.pt" ]]; then
  resume_args=(--resume "$OUTPUT/checkpoints/resume.pt")
fi

"$ENVIRONMENT/bin/python" -m moge.scripts.overfit_hypersim_staged_v3 \
  --data "$DATA" \
  --selection "$SELECTION" \
  --sample-id ai_002_003_cam_00_frame.0000 \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  "${resume_args[@]}" \
  --stage1-min-steps 2 \
  --stage1-max-steps 2 \
  --joint-min-steps 2 \
  --joint-max-steps 2 \
  --eval-every 1 \
  --plateau-patience-evals 2 \
  --loss-plateau-window 2 \
  --loss-plateau-chunk 1 \
  --success-patience-evals 3 \
  --backbone-freeze-steps 0 \
  --backbone-warmup-end 1 \
  --log-every 1 \
  --loss-smoothing-window 2 \
  >"$OUTPUT/smoke.log" 2>&1

"$ENVIRONMENT/bin/python" -m pytest -q \
  tests/test_staged_single_overfit_v3.py \
  >"$OUTPUT/test_results.txt"
