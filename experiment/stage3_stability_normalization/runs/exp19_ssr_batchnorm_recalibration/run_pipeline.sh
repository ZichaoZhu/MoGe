#!/usr/bin/env bash
set -uo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp19_ssr_batchnorm_recalibration"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data"
ROIS="$REPO/experiment/stage2_training_strategy/runs/exp13_fixed_base_ssr_diagnostic/fine_structure_rois.json"
CHECKPOINT="$REPO/experiment/stage2_training_strategy/runs/exp15_detached_head_ssr_coadaptation/artifacts/formal/checkpoint.pt"
BASELINE_RESIDUAL="$REPO/experiment/stage3_stability_normalization/runs/exp17_ssr_residual_outlier_scan/artifacts/scan/report.json"
BASELINE_METRICS="$REPO/experiment/stage2_training_strategy/runs/exp15_detached_head_ssr_coadaptation/artifacts/posthoc_best/report.json"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU="${1:?GPU编号不能为空}"
MODE="${2:-formal}"

case "$GPU" in 0|1|2|3) ;; *) exit 92 ;; esac
case "$MODE" in smoke|formal) ;; *) exit 93 ;; esac
test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$ROIS" "$CHECKPOINT" "$BASELINE_RESIDUAL" \
  "$BASELINE_METRICS" "$CACHE" "$TMP"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
test -f "$CHECKPOINT"
test -f "$BASELINE_RESIDUAL"
test -f "$BASELINE_METRICS"

if [[ "$MODE" = formal ]]; then
  ROOT="$EXP_DIR/artifacts"
  STATUS="$ROOT/status/scan.json"
  SAMPLE_ARGS=()
else
  ROOT="$EXP_DIR/artifacts/smoke"
  STATUS="$ROOT/status/smoke.json"
  SAMPLE_ARGS=(--max-train-samples 8)
fi
CALIBRATION="$ROOT/calibration"
LOGS="$ROOT/logs"
install -d "$CALIBRATION" "$LOGS" "$(dirname "$STATUS")" \
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
if [[ ! -f "$CALIBRATION/report.json" ]]; then
  "$ENVIRONMENT/bin/python" tools/moge3/calibrate_ssr_batchnorm.py \
    --data "$DATA" \
    --checkpoint "$CHECKPOINT" \
    --output "$CALIBRATION" \
    --safe-root "$SAFE_ROOT" \
    --device cuda:0 \
    --height 384 \
    --width 512 \
    --num-tokens 2500 \
    --refinement-steps 3 \
    --batch-sizes 1 8 \
    --seed 191 \
    "${SAMPLE_ARGS[@]}" >>"$LOGS/calibration.log" 2>&1 || status=$?
fi

if [[ "$status" -eq 0 ]]; then
  for batch_size in 1 8
  do
    checkpoint="$CALIBRATION/bn_recal_b${batch_size}.pt"
    residual="$ROOT/residual_b${batch_size}"
    if [[ ! -f "$residual/report.json" ]]; then
      residual_limit=()
      if [[ "$MODE" = smoke ]]; then
        residual_limit=(--max-samples-per-split 1)
      fi
      "$ENVIRONMENT/bin/python" tools/moge3/scan_checkpoint_residuals.py \
        --data "$DATA" \
        --checkpoint "$checkpoint" \
        --output "$residual" \
        --safe-root "$SAFE_ROOT" \
        --device cuda:0 \
        --height 384 \
        --width 512 \
        --num-tokens 2500 \
        --refinement-steps 3 \
        --modes eval \
        --threshold 0.5 \
        --seed 192 \
        --splits train val test \
        "${residual_limit[@]}" >>"$LOGS/residual_b${batch_size}.log" 2>&1 \
        || status=$?
    fi
    if [[ "$status" -ne 0 ]]; then
      break
    fi
    if [[ "$MODE" = formal ]]; then
      metrics="$ROOT/metrics_b${batch_size}"
      if [[ ! -f "$metrics/report.json" ]]; then
        "$ENVIRONMENT/bin/python" \
          -m moge.scripts.evaluate_hypersim_checkpoint_rois_v3 \
          --data "$DATA" \
          --checkpoint "$checkpoint" \
          --fine-structure-rois "$ROIS" \
          --output "$metrics" \
          --safe-root "$SAFE_ROOT" \
          --device cuda:0 \
          --height 384 \
          --width 512 \
          --num-tokens 2500 \
          --evaluation-steps 0 1 3 5 \
          --batch-size 1 \
          --boundary-threshold 0.03 \
          >>"$LOGS/metrics_b${batch_size}.log" 2>&1 || status=$?
      fi
    fi
    if [[ "$status" -ne 0 ]]; then
      break
    fi
  done
fi

if [[ "$status" -eq 0 && "$MODE" = formal ]]; then
  "$ENVIRONMENT/bin/python" \
    "$EXP_DIR/summarize_results.py" \
    --baseline-residual "$BASELINE_RESIDUAL" \
    --baseline-metrics "$BASELINE_METRICS" \
    --calibration "$CALIBRATION/report.json" \
    --artifacts "$ROOT" \
    --output "$ROOT/scan/report.json" \
    >>"$LOGS/summary.log" 2>&1 || status=$?
fi

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
path.parent.mkdir(parents=True, exist_ok=True)
temporary = path.with_suffix(path.suffix + ".incomplete")
temporary.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
exit "$status"
