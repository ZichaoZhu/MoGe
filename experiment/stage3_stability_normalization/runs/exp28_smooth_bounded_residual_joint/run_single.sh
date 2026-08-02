#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp28_smooth_bounded_residual_joint"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU="${1:?GPU编号不能为空}"
MODE="${2:-formal}"

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

EXP15_DIR="$(first_existing \
  "$REPO/experiment/stage2_training_strategy/runs/exp15_detached_head_ssr_coadaptation" \
  "$REPO/experiment/exp15_detached_head_ssr_coadaptation")" || exit 96
DATA="$(first_existing \
  "$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data" \
  "$REPO/experiment/exp11_hypersim_100_train_staged_joint_overfit/data")" || exit 96
ROIS="$(first_existing \
  "$REPO/experiment/stage2_training_strategy/runs/exp13_fixed_base_ssr_diagnostic/fine_structure_rois.json" \
  "$REPO/experiment/exp13_fixed_base_ssr_diagnostic/fine_structure_rois.json")" || exit 96
WARM_START="$EXP15_DIR/artifacts/formal/checkpoint.pt"
EXPECTED_SHA=f602855f9e5628e27393734f121a5ffa2f485e1cbfd1c410a2ed0850ca5200fa

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac
case "$MODE" in
  smoke)
    OUTPUT="$EXP_DIR/artifacts/smoke"
    TOTAL_STEPS=804
    BATCH_SIZE=2
    EVAL_EVERY=4
    PERIODIC_TRAIN=4
    SELECTION_SCOPE=full
    LIMIT_ARGS=(--max-samples-per-split 4)
    ;;
  formal)
    OUTPUT="$EXP_DIR/artifacts/formal"
    TOTAL_STEPS=1000
    BATCH_SIZE=8
    EVAL_EVERY=50
    PERIODIC_TRAIN=100
    SELECTION_SCOPE=structure
    LIMIT_ARGS=()
    ;;
  *)
    exit 94
    ;;
esac

LOGS="$EXP_DIR/artifacts/logs"
STATUS="$EXP_DIR/artifacts/status/${MODE}.json"
test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$EXP15_DIR" "$DATA" "$ROIS" "$WARM_START" "$OUTPUT" \
  "$LOGS" "$STATUS" "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
test -f "$WARM_START"
ACTUAL_SHA="$(sha256sum "$WARM_START" | awk '{print $1}')"
if [[ "$ACTUAL_SHA" != "$EXPECTED_SHA" ]]; then
  exit 95
fi
if [[ -f "$OUTPUT/report.json" ]]; then
  exit 0
fi

install -d "$OUTPUT" "$LOGS" "$(dirname "$STATUS")" \
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
export CUDA_VISIBLE_DEVICES="$GPU"

CHECKPOINT_ARGS=(--warm-start "$WARM_START")
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
  --steps "$TOTAL_STEPS" \
  --batch-size "$BATCH_SIZE" \
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
  --eval-every "$EVAL_EVERY" \
  --periodic-train-samples "$PERIODIC_TRAIN" \
  --selection-split train \
  --selection-scope "$SELECTION_SCOPE" \
  --eval-batch-size 1 \
  --log-every 10 \
  --loss-smoothing-window 50 \
  --boundary-threshold 0.03 \
  --seed 161 \
  "${LIMIT_ARGS[@]}" >>"$LOGS/${MODE}.log" 2>&1 || status=$?

"$ENVIRONMENT/bin/python" - "$STATUS" "$status" "$GPU" "$MODE" \
  "$STARTED_AT" <<'PY'
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
    "mode": sys.argv[4],
    "started_at": sys.argv[5],
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
