#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp29_smooth_bounded_residual_long_joint"
EXP28_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp28_smooth_bounded_residual_joint"
SOURCE_OUTPUT="$EXP28_DIR/artifacts/formal"
SOURCE_RESUME="$SOURCE_OUTPUT/resume_checkpoint.pt"
EXPECTED_SHA=8f82fb9e17cf997645e9c321a1ae3becdfe693ae3e51c71e1f32bc09b0647728
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU="${1:?GPU编号不能为空}"

first_existing() {
  local candidate
  for candidate in "$@"; do
    if [[ -e "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done
  return 1
}

DATA="$(first_existing \
  "$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data" \
  "$REPO/experiment/exp11_hypersim_100_train_staged_joint_overfit/data")" || exit 96
ROIS="$(first_existing \
  "$REPO/experiment/stage2_training_strategy/runs/exp13_fixed_base_ssr_diagnostic/fine_structure_rois.json" \
  "$REPO/experiment/exp13_fixed_base_ssr_diagnostic/fine_structure_rois.json")" || exit 96

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac

OUTPUT="$EXP_DIR/artifacts/formal"
LOGS="$EXP_DIR/artifacts/logs"
STATUS="$EXP_DIR/artifacts/status/formal.json"
test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$EXP28_DIR" "$SOURCE_OUTPUT" "$SOURCE_RESUME" \
  "$DATA" "$ROIS" "$OUTPUT" "$LOGS" "$STATUS" "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
test -f "$SOURCE_RESUME"
ACTUAL_SHA="$(sha256sum "$SOURCE_RESUME" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  exit 95
fi
if [[ -f "$OUTPUT/report.json" ]]; then
  exit 0
fi

install -d "$OUTPUT" "$LOGS" "$(dirname "$STATUS")" \
  "$CACHE/huggingface" "$CACHE/torch" "$CACHE/pip" "$CACHE/xdg" "$TMP"

# The trainer validates that resume histories begin at their original step.
# Seed the new experiment with Exp28 histories and inherited best checkpoint.
if [[ ! -f "$OUTPUT/training_history.csv" ]]; then
  cp "$SOURCE_OUTPUT/training_history.csv" "$OUTPUT/training_history.csv"
fi
if [[ ! -f "$OUTPUT/evaluation_history.csv" ]]; then
  cp "$SOURCE_OUTPUT/evaluation_history.csv" "$OUTPUT/evaluation_history.csv"
fi
if [[ ! -f "$OUTPUT/checkpoint.pt" ]]; then
  cp --reflink=auto "$SOURCE_OUTPUT/checkpoint.pt" "$OUTPUT/checkpoint.pt"
fi

cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU"

CHECKPOINT_ARGS=(--resume "$SOURCE_RESUME")
if [[ -f "$OUTPUT/resume_checkpoint.pt" ]]; then
  CHECKPOINT_ARGS=(--resume "$OUTPUT/resume_checkpoint.pt")
fi

STARTED_AT="$(date --iso-8601=seconds)"
status=0
"$ENVIRONMENT/bin/python" -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$OUTPUT" \
  "${CHECKPOINT_ARGS[@]}" \
  --safe-root "$SAFE_ROOT" \
  --device cuda:0 \
  --freeze-backbone \
  --height 384 \
  --width 512 \
  --num-tokens 2500 \
  --refinement-steps 3 \
  --steps 5000 \
  --batch-size 8 \
  --microbatch-size 1 \
  --refiner-detach-steps 800 \
  --backbone-freeze-steps 0 \
  --backbone-warmup-end 1 \
  --fine-structure-rois "$ROIS" \
  --ssr-learning-rate 2e-6 \
  --head-learning-rate 5e-7 \
  --backbone-learning-rate 0 \
  --weight-decay 1e-2 \
  --gradient-clip-norm 1 \
  --max-preclip-grad-norm 100 \
  --max-skipped-preclip-steps 5 \
  --max-consecutive-skipped-preclip-steps 2 \
  --smooth-log-depth-residual-bound 0.1 \
  --max-abs-log-depth-residual 0.100001 \
  --max-abs-raw-log-depth-residual 2 \
  --max-refined-point-rel 0.3 \
  --max-refined-to-base-ratio 3 \
  --global-weight 1 \
  --local-weight 1 \
  --edge-weight 1 \
  --local-scales 4 16 64 \
  --eval-every 100 \
  --periodic-train-samples 100 \
  --selection-split train \
  --selection-scope structure \
  --eval-batch-size 1 \
  --log-every 25 \
  --loss-smoothing-window 100 \
  --boundary-threshold 0.03 \
  --seed 161 >>"$LOGS/formal.log" 2>&1 || status=$?

"$ENVIRONMENT/bin/python" - "$STATUS" "$status" "$GPU" "$STARTED_AT" <<'PY'
import datetime
import json
import os
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
payload = {
    "status": "complete" if int(sys.argv[2]) == 0 else "failed",
    "exit_code": int(sys.argv[2]),
    "physical_gpu": int(sys.argv[3]),
    "started_at": sys.argv[4],
    "finished_at": datetime.datetime.now().astimezone().isoformat(),
}
temporary = path.with_suffix(path.suffix + ".incomplete")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
exit "$status"
