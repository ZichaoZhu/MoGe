#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage3_stability_normalization/runs/exp21_iteration_specific_bn_statistics"
MONITOR="$REPO/experiment/stage3_stability_normalization/runs/exp17_ssr_residual_outlier_scan/monitor_scan.py"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$MONITOR"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done
install -d "$EXP_DIR/artifacts/logs"
cd "$REPO"
nohup "$ENVIRONMENT/bin/python" "$MONITOR" \
  --experiment-dir "$EXP_DIR" \
  --interval-seconds 600 \
  --minimum-free-mib 30000 \
  --maximum-utilization 10 \
  >>"$EXP_DIR/artifacts/logs/monitor_launcher.log" 2>&1 &
echo "$!"
