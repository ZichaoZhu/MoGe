#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp11_hypersim_100_train_staged_joint_overfit"
LOGS="$EXP_DIR/artifacts/logs"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$LOGS"; do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done
test -f "$EXP_DIR/data/manifest.json"
install -d "$LOGS"

if pgrep -af "supervise_training.py.*$EXP_DIR" | grep -v "$$" >/dev/null; then
  echo "Exp11监督器已经运行，拒绝重复启动" >&2
  exit 5
fi

cd "$REPO"
export PYTHONPATH="$REPO"
nohup "$ENVIRONMENT/bin/python" "$EXP_DIR/supervise_training.py" \
  --experiment-dir "$EXP_DIR" \
  --interval-seconds 600 \
  --min-free-mib 18000 \
  --max-utilization 30 \
  --max-gpus 4 >"$LOGS/supervisor_launcher.log" 2>&1 &
echo "$!" >"$LOGS/supervisor.pid"
echo "Exp11监督器已启动，PID=$!"
