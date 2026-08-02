#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit"
EXP10_MANIFEST="$REPO/experiment/stage2_training_strategy/runs/exp10_hypersim_48_staged_joint_overfit/data/manifest.json"
CANDIDATES="$EXP_DIR/data_review/candidate_52/candidate_52.json"
APPROVAL="$EXP_DIR/data_review/candidate_approval.json"
OUTPUT="$EXP_DIR/data"
LOGS="$EXP_DIR/artifacts/logs"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$EXP10_MANIFEST" "$CANDIDATES" "$APPROVAL" \
  "$OUTPUT" "$LOGS"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done
test -f "$EXP10_MANIFEST"
test -f "$CANDIDATES"
test -f "$APPROVAL"

install -d "$OUTPUT" "$LOGS"
cd "$REPO"
export PYTHONPATH="$REPO"
"$ENVIRONMENT/bin/python" tools/moge3/prepare_exp11_hypersim.py \
  --exp10-manifest "$EXP10_MANIFEST" \
  --candidate-report "$CANDIDATES" \
  --approval "$APPROVAL" \
  --output "$OUTPUT" \
  --source-root /nas1/datasets/hypersim/raw \
  --safe-root "$SAFE_ROOT" | tee "$LOGS/prepare_data.log"
