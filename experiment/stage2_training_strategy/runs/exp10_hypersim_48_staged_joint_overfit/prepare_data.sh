#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp10_hypersim_48_staged_joint_overfit"
EXP2_DATA="$REPO/experiment/stage1_learning_capability/runs/exp2_hypersim_smallset_overfit/data"
EXP7_DATA="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/data"
OUTPUT="$EXP_DIR/data"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
for path in "$REPO" "$EXP_DIR" "$EXP2_DATA" "$EXP7_DATA" "$OUTPUT"; do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done

cd "$REPO"
export PYTHONPATH="$REPO"
"$ENVIRONMENT/bin/python" tools/moge3/prepare_combined_hypersim.py \
  --exp2-data "$EXP2_DATA" \
  --exp7-data "$EXP7_DATA" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT"
