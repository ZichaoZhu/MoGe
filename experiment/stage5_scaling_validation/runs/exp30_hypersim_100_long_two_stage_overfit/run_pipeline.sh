#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage5_scaling_validation/runs/exp30_hypersim_100_long_two_stage_overfit"
LOG_DIR="$EXP_DIR/artifacts/logs"
PID_FILE="$EXP_DIR/artifacts/pipeline.pid"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$PROJECT_ROOT" "$REPO" "$ENVIRONMENT" "$EXP_DIR" "$LOG_DIR" "$PID_FILE"
do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done

install -d "$LOG_DIR"
if [[ -f "$PID_FILE" ]] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
  echo "Exp30 流水线已经运行，拒绝重复启动" >&2
  exit 5
fi

cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$PROJECT_ROOT/tmp/moge3"
export HF_HOME="$PROJECT_ROOT/cache/moge3/huggingface"
export TORCH_HOME="$PROJECT_ROOT/cache/moge3/torch"
export PIP_CACHE_DIR="$PROJECT_ROOT/cache/moge3/pip"
export XDG_CACHE_HOME="$PROJECT_ROOT/cache/moge3/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

install -d "$TMPDIR" "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$XDG_CACHE_HOME"
nohup "$ENVIRONMENT/bin/python" tools/moge3/run_exp30.py run \
  --experiment-dir "$EXP_DIR" \
  --safe-root "$SAFE_ROOT" \
  >"$LOG_DIR/pipeline.log" 2>&1 &
printf '%s\n' "$!" >"$PID_FILE"
echo "Exp30 流水线已启动，PID=$(cat "$PID_FILE")"
