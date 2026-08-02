#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp24_microbatch2_long_detached"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data"
ROIS="$REPO/experiment/stage2_training_strategy/runs/exp13_fixed_base_ssr_diagnostic/fine_structure_rois.json"
SOURCE_TRAINING="$REPO/experiment/stage3_stability_normalization/runs/exp23_true_microbatch2_batchnorm/artifacts/formal/training"
RESUME="$SOURCE_TRAINING/resume_checkpoint.pt"
SOURCE_BEST="$SOURCE_TRAINING/checkpoint.pt"
BASELINE_METRICS="$REPO/experiment/stage2_training_strategy/runs/exp15_detached_head_ssr_coadaptation/artifacts/posthoc_best/report.json"
BASELINE_RESIDUAL="$REPO/experiment/stage3_stability_normalization/runs/exp17_ssr_residual_outlier_scan/artifacts/scan/report.json"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU="${1:?GPU编号不能为空}"
MODE="${2:-formal}"

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac
case "$MODE" in smoke|formal) ;; *) exit 93 ;; esac
ROOT="$EXP_DIR/artifacts/$MODE"
STATUS="$EXP_DIR/artifacts/status/${MODE}.json"
LOGS="$EXP_DIR/artifacts/logs"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$ROIS" "$SOURCE_TRAINING" "$RESUME" \
  "$SOURCE_BEST" "$BASELINE_METRICS" "$BASELINE_RESIDUAL" "$ROOT" \
  "$STATUS" "$LOGS" "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
for file in "$RESUME" "$SOURCE_BEST" "$BASELINE_METRICS" "$BASELINE_RESIDUAL"
do
  test -f "$file"
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
if [[ "$MODE" = smoke ]]; then
  "$ENVIRONMENT/bin/python" - "$RESUME" "$ROOT/resume_audit.json" <<'PY' \
    >>"$LOGS/smoke.log" 2>&1 || status=$?
import json
import os
import pathlib
import sys
import torch

source = pathlib.Path(sys.argv[1])
destination = pathlib.Path(sys.argv[2])
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
payload = {
    "status": "complete",
    "step": int(checkpoint["step"]),
    "best_step": int(checkpoint["best_step"]),
    "batch_size": int(checkpoint["args"]["batch_size"]),
    "microbatch_size": int(checkpoint["args"]["microbatch_size"]),
    "ssr_normalization": checkpoint["args"]["ssr_normalization"],
    "has_optimizer": "optimizer" in checkpoint,
    "has_cpu_rng": "cpu_generator_state" in checkpoint,
    "has_loss_rng": "loss_generator_state" in checkpoint,
}
temporary = destination.with_suffix(".json.incomplete")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, destination)
PY
  if [[ "$status" -eq 0 ]]; then
    "$ENVIRONMENT/bin/python" "$EXP_DIR/summarize_results.py" \
      --root "$ROOT" --mode smoke >>"$LOGS/smoke.log" 2>&1 || status=$?
  fi
else
  TRAINING="$ROOT/training"
  install -d "$TRAINING"
  if [[ ! -f "$TRAINING/training_history.csv" ]]; then
    cp "$SOURCE_TRAINING/training_history.csv" "$TRAINING/training_history.csv"
  fi
  if [[ ! -f "$TRAINING/evaluation_history.csv" ]]; then
    cp "$SOURCE_TRAINING/evaluation_history.csv" "$TRAINING/evaluation_history.csv"
  fi
  if [[ ! -f "$TRAINING/checkpoint.pt" ]]; then
    ln "$SOURCE_BEST" "$TRAINING/checkpoint.pt"
  fi
  if [[ ! -f "$TRAINING/report.json" ]]; then
    "$ENVIRONMENT/bin/python" -m moge.scripts.train_hypersim_joint_v3 \
      --data "$DATA" \
      --output "$TRAINING" \
      --resume "$RESUME" \
      --safe-root "$SAFE_ROOT" \
      --device cuda:0 \
      --freeze-backbone \
      --height 384 \
      --width 512 \
      --num-tokens 2500 \
      --ssr-normalization batch_norm \
      --refinement-steps 3 \
      --steps 800 \
      --batch-size 8 \
      --microbatch-size 2 \
      --refiner-detach-steps 800 \
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
      --eval-every 100 \
      --periodic-train-samples 100 \
      --selection-split train \
      --selection-scope structure \
      --eval-batch-size 1 \
      --log-every 10 \
      --loss-smoothing-window 50 \
      --boundary-threshold 0.03 \
      --seed 151 >>"$LOGS/formal_training.log" 2>&1 || status=$?
  fi

  FINAL_CHECKPOINT="$TRAINING/latest_checkpoint.pt"
  if [[ "$status" -eq 0 && ! -f "$ROOT/evaluation/report.json" ]]; then
    "$ENVIRONMENT/bin/python" \
      -m moge.scripts.evaluate_hypersim_checkpoint_rois_v3 \
      --data "$DATA" \
      --checkpoint "$FINAL_CHECKPOINT" \
      --fine-structure-rois "$ROIS" \
      --output "$ROOT/evaluation" \
      --safe-root "$SAFE_ROOT" \
      --device cuda:0 \
      --height 384 \
      --width 512 \
      --num-tokens 2500 \
      --evaluation-steps 0 1 3 5 \
      --batch-size 1 \
      --boundary-threshold 0.03 \
      >>"$LOGS/formal_evaluation.log" 2>&1 || status=$?
  fi

  if [[ "$status" -eq 0 && ! -f "$ROOT/residual/report.json" ]]; then
    "$ENVIRONMENT/bin/python" tools/moge3/scan_checkpoint_residuals.py \
      --data "$DATA" \
      --checkpoint "$FINAL_CHECKPOINT" \
      --output "$ROOT/residual" \
      --safe-root "$SAFE_ROOT" \
      --device cuda:0 \
      --height 384 \
      --width 512 \
      --num-tokens 2500 \
      --refinement-steps 3 \
      --modes eval \
      --threshold 0.5 \
      --seed 241 \
      --splits train val test \
      >>"$LOGS/formal_residual.log" 2>&1 || status=$?
  fi

  if [[ "$status" -eq 0 ]]; then
    "$ENVIRONMENT/bin/python" "$EXP_DIR/summarize_results.py" \
      --root "$ROOT" \
      --mode formal \
      --baseline-metrics "$BASELINE_METRICS" \
      --baseline-residual "$BASELINE_RESIDUAL" \
      >>"$LOGS/formal_summary.log" 2>&1 || status=$?
  fi
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
