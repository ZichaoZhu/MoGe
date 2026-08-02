#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp23_true_microbatch2_batchnorm"
SUPERVISOR="$REPO/experiment/stage2_training_strategy/runs/exp14_head_ssr_short_joint/monitor_training.py"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$SUPERVISOR"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
install -d "$EXP_DIR/artifacts/logs"
cd "$REPO"
export PYTHONPATH="$REPO"
exec "$ENVIRONMENT/bin/python" \
  "$SUPERVISOR" \
  --experiment-dir "$EXP_DIR" \
  --interval-seconds 600 \
  --minimum-free-mib 30000 \
  --maximum-utilization 10
