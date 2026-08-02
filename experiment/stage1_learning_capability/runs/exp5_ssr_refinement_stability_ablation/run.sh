#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage1_learning_capability/runs/exp5_ssr_refinement_stability_ablation"
RUNS="$EXP_DIR/artifacts/runs"
LOGS="$EXP_DIR/artifacts/logs"
SUMMARY="$EXP_DIR/artifacts/summary"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
install -d "$RUNS" "$LOGS" "$SUMMARY" "$CACHE/huggingface" "$CACHE/torch" \
  "$CACHE/pip" "$CACHE/xdg" "$TMP"
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1

run_pair() {
  local arm_2="$1"
  local arm_3="$2"
  "$EXP_DIR/run_arm.sh" "$arm_2" 2 >"$LOGS/$arm_2.log" 2>&1 &
  local pid_2=$!
  "$EXP_DIR/run_arm.sh" "$arm_3" 3 >"$LOGS/$arm_3.log" 2>&1 &
  local pid_3=$!
  wait "$pid_2"
  wait "$pid_3"
}

run_pair arm_a_k1_edge1 arm_b_k1_edge4
run_pair arm_c_k3_edge1 arm_d_k3_edge4

"$ENVIRONMENT/bin/python" tools/moge3/summarize_ssr_stability_ablation.py \
  --config "$EXP_DIR/config.json" \
  --runs "$RUNS" \
  --output "$SUMMARY" \
  --safe-root "$SAFE_ROOT"
