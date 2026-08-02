#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp9_thin_structure_single_image_staged_overfit"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/data"
SELECTION="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/gif_selection.json"
SMOKE_REPORT="$EXP_DIR/smoke/report.json"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
for path in "$EXP_DIR" "$DATA" "$SELECTION" "$CACHE" "$TMP"; do
  case "$(realpath -m "$path")/" in
    "$SAFE_ROOT/"*) ;;
    *) echo "路径越过安全根目录：$path" >&2; exit 91 ;;
  esac
done
test -f "$SMOKE_REPORT"

install -d "$CACHE/huggingface" "$CACHE/torch" "$CACHE/pip" "$CACHE/xdg" "$TMP"
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

REQUIRED_FREE_MIB="$(
  "$ENVIRONMENT/bin/python" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); peak=max(int(p["peak_memory_bytes"]),int(p.get("observed_nvidia_smi_increment_peak_bytes",0))); print((peak+(1<<20)-1)//(1<<20)+4096)' \
    "$SMOKE_REPORT"
)"

"$ENVIRONMENT/bin/python" tools/moge3/run_exp9_scheduler.py \
  --experiment-root "$EXP_DIR" \
  --data "$DATA" \
  --selection "$SELECTION" \
  --safe-root "$SAFE_ROOT" \
  --required-free-mib "$REQUIRED_FREE_MIB" \
  --poll-seconds 30 \
  --max-parallel 4 \
  --max-utilization-percent 10
