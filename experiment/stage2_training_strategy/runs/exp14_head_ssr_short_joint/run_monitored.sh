#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp14_head_ssr_short_joint"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
case "$(realpath -m "$EXP_DIR")/" in
  "$SAFE_ROOT/"*) ;;
  *) exit 91 ;;
esac
install -d "$EXP_DIR/artifacts/logs"
cd "$REPO"
export PYTHONPATH="$REPO"
exec "$ENVIRONMENT/bin/python" \
  "$EXP_DIR/monitor_training.py" \
  --experiment-dir "$EXP_DIR" \
  --interval-seconds 600 \
  --minimum-free-mib 30000 \
  --maximum-utilization 10
