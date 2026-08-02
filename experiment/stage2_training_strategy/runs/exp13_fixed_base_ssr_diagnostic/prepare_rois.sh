#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp13_fixed_base_ssr_diagnostic"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
for path in "$REPO" "$EXP_DIR"; do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) exit 91 ;;
  esac
done

cd "$REPO"
export PYTHONPATH="$REPO"
"$ENVIRONMENT/bin/python" tools/moge3/prepare_exp13_fine_structure_rois.py \
  --data-manifest \
  "$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data/manifest.json" \
  --candidates \
  "$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data_review/candidate_52/candidate_52.json" \
  --approval \
  "$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit/data_review/candidate_approval.json" \
  --held-out-selection \
  "$REPO/experiment/stage2_training_strategy/runs/exp12_hypersim_100_immediate_joint_finetuning/results/viewer_selection/fine_sample_selection.json" \
  --output "$EXP_DIR/fine_structure_rois.json" \
  --safe-root "$SAFE_ROOT"
