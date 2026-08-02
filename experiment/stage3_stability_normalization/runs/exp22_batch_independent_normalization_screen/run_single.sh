#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp22_batch_independent_normalization_screen"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data"
ROIS="$REPO/experiment/stage2_training_strategy/runs/exp13_fixed_base_ssr_diagnostic/fine_structure_rois.json"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU="${1:?GPU编号不能为空}"
MODE="${2:-formal}"

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac
case "$MODE" in
  smoke)
    ROOT="$EXP_DIR/artifacts/smoke"
    TOTAL_STEPS=4
    BATCH_SIZE=2
    EVAL_EVERY=4
    PERIODIC_TRAIN=4
    SELECTION_SCOPE=full
    LIMIT_ARGS=(--max-samples-per-split 4)
    ;;
  formal)
    ROOT="$EXP_DIR/artifacts/formal"
    TOTAL_STEPS=200
    BATCH_SIZE=8
    EVAL_EVERY=100
    PERIODIC_TRAIN=100
    SELECTION_SCOPE=structure
    LIMIT_ARGS=()
    ;;
  *) exit 93 ;;
esac

STATUS="$EXP_DIR/artifacts/status/${MODE}.json"
LOGS="$EXP_DIR/artifacts/logs"
test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$ROIS" "$ROOT" "$STATUS" "$LOGS" "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
if [[ -f "$ROOT/report.json" ]]; then
  exit 0
fi

install -d "$ROOT" "$LOGS" "$(dirname "$STATUS")" \
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

STARTED_AT="$(date --iso-8601=seconds)"
status=0
for NORMALIZATION in layer_norm group_norm
do
  VARIANT_ROOT="$ROOT/$NORMALIZATION"
  TRAINING="$VARIANT_ROOT/training"
  if [[ "$status" -eq 0 && ! -f "$TRAINING/report.json" ]]; then
    "$ENVIRONMENT/bin/python" -m moge.scripts.train_hypersim_joint_v3 \
      --data "$DATA" \
      --output "$TRAINING" \
      --safe-root "$SAFE_ROOT" \
      --device cuda:0 \
      --freeze-backbone \
      --height 384 \
      --width 512 \
      --num-tokens 2500 \
      --ssr-normalization "$NORMALIZATION" \
      --refinement-steps 3 \
      --steps "$TOTAL_STEPS" \
      --batch-size "$BATCH_SIZE" \
      --microbatch-size 1 \
      --refiner-detach-steps "$TOTAL_STEPS" \
      --backbone-freeze-steps 0 \
      --backbone-warmup-end 1 \
      --fine-structure-rois "$ROIS" \
      --ssr-learning-rate 2e-5 \
      --head-learning-rate 1e-6 \
      --backbone-learning-rate 0 \
      --weight-decay 1e-2 \
      --gradient-clip-norm 1 \
      --max-preclip-grad-norm 100 \
      --max-skipped-preclip-steps 5 \
      --max-consecutive-skipped-preclip-steps 2 \
      --max-abs-log-depth-residual 0.5 \
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
      --seed 221 \
      "${LIMIT_ARGS[@]}" >>"$LOGS/${MODE}_${NORMALIZATION}_training.log" 2>&1 \
      || status=$?
  fi

  if [[ "$status" -eq 0 && "$MODE" = formal && ! -f "$VARIANT_ROOT/evaluation/report.json" ]]; then
    "$ENVIRONMENT/bin/python" \
      -m moge.scripts.evaluate_hypersim_checkpoint_rois_v3 \
      --data "$DATA" \
      --checkpoint "$TRAINING/checkpoint.pt" \
      --fine-structure-rois "$ROIS" \
      --output "$VARIANT_ROOT/evaluation" \
      --safe-root "$SAFE_ROOT" \
      --device cuda:0 \
      --height 384 \
      --width 512 \
      --num-tokens 2500 \
      --evaluation-steps 0 1 3 5 \
      --batch-size 1 \
      --boundary-threshold 0.03 \
      >>"$LOGS/${MODE}_${NORMALIZATION}_evaluation.log" 2>&1 || status=$?
  fi

  if [[ "$status" -eq 0 && "$MODE" = formal && ! -f "$VARIANT_ROOT/residual/report.json" ]]; then
    "$ENVIRONMENT/bin/python" tools/moge3/scan_checkpoint_residuals.py \
      --data "$DATA" \
      --checkpoint "$TRAINING/checkpoint.pt" \
      --output "$VARIANT_ROOT/residual" \
      --safe-root "$SAFE_ROOT" \
      --device cuda:0 \
      --height 384 \
      --width 512 \
      --num-tokens 2500 \
      --refinement-steps 3 \
      --modes eval \
      --threshold 0.5 \
      --seed 222 \
      --splits train val test \
      >>"$LOGS/${MODE}_${NORMALIZATION}_residual.log" 2>&1 || status=$?
  fi
done

if [[ "$status" -eq 0 ]]; then
  "$ENVIRONMENT/bin/python" "$EXP_DIR/summarize_results.py" \
    --root "$ROOT" --mode "$MODE" >>"$LOGS/${MODE}_summary.log" 2>&1 || status=$?
fi

"$ENVIRONMENT/bin/python" - "$STATUS" "$status" "$GPU" "$MODE" "$STARTED_AT" <<'PY'
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
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".incomplete")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
exit "$status"
