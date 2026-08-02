#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp17_ssr_residual_outlier_scan"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
case "$(realpath -m "$EXP_DIR")/" in
  "$SAFE_ROOT/"*) ;;
  *) exit 91 ;;
esac
install -d "$EXP_DIR/artifacts/logs"
cd "$REPO"
nohup "$ENVIRONMENT/bin/python" "$EXP_DIR/monitor_scan.py" \
  --experiment-dir "$EXP_DIR" \
  --interval-seconds 600 \
  --minimum-free-mib 30000 \
  --maximum-utilization 10 \
  >>"$EXP_DIR/artifacts/logs/monitor_launcher.log" 2>&1 &
echo "$!"
