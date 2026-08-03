#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
REPO="$SAFE_ROOT/2026_TPAMI_InfiniGeometry/third_party/MoGe-3"
EXP_DIR="$REPO/experiment/stage5_scaling_validation/runs/exp30_hypersim_100_long_two_stage_overfit"
case "$(realpath -m "$EXP_DIR")/" in "$SAFE_ROOT/"*) ;; *) exit 91 ;; esac
cd "$REPO"
export PYTHONPATH="$REPO"
"$SAFE_ROOT/2026_TPAMI_InfiniGeometry/envs/moge3/bin/python" \
  tools/moge3/run_exp30.py status \
  --experiment-dir "$EXP_DIR" \
  --safe-root "$SAFE_ROOT"
