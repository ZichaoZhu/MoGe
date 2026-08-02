#!/usr/bin/env bash
set -euo pipefail

SAFE_ROOT=/mnt/data/home/zhuzichao
PROJECT_ROOT="$SAFE_ROOT/2026_TPAMI_InfiniGeometry"
REPO="$PROJECT_ROOT/third_party/MoGe-3"
ENVIRONMENT="$PROJECT_ROOT/envs/moge3"
EXP_DIR="$REPO/experiment/stage2_training_strategy/runs/exp8_two_gpu_head_ssr_joint_finetuning"
DATA="$REPO/experiment/stage2_training_strategy/runs/exp7_ssr_thin_structure_training/data"
OUTPUT="$EXP_DIR/artifacts/ddp_smoke"
LOGS="$EXP_DIR/artifacts/logs"
CACHE="$PROJECT_ROOT/cache/moge3"
TMP="$PROJECT_ROOT/tmp/moge3"
GPU_PAIR="${MOGE3_GPUS:-0,1}"

test "$(realpath -e "$SAFE_ROOT")" = /mnt/data/home/zhuzichao
test "$(realpath -e "$REPO")" = "$REPO"
case "$(realpath -m "$OUTPUT")/" in
  "$SAFE_ROOT/"*) ;;
  *) echo "DDP烟雾测试目录越过安全根目录" >&2; exit 91 ;;
esac
case "$GPU_PAIR" in
  0,1|0,2|0,3|1,0|1,2|1,3|2,0|2,1|2,3|3,0|3,1|3,2) ;;
  *) echo "GPU组合必须是两张不同的0至3号GPU" >&2; exit 92 ;;
esac
if [[ -e "$OUTPUT/report.json" ]]; then
  echo "DDP烟雾测试已经完成，拒绝覆盖" >&2
  exit 5
fi

install -d "$OUTPUT" "$LOGS" "$CACHE/huggingface" "$CACHE/torch" \
  "$CACHE/pip" "$CACHE/xdg" "$TMP"
cd "$REPO"
export PYTHONPATH="$REPO"
export TMPDIR="$TMP"
export HF_HOME="$CACHE/huggingface"
export TORCH_HOME="$CACHE/torch"
export PIP_CACHE_DIR="$CACHE/pip"
export XDG_CACHE_HOME="$CACHE/xdg"
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="$GPU_PAIR"

"$ENVIRONMENT/bin/torchrun" --standalone --nproc-per-node=2 \
  -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$OUTPUT" \
  --safe-root "$SAFE_ROOT" \
  --ddp-backend gloo \
  --freeze-backbone \
  --height 64 \
  --width 64 \
  --num-tokens 64 \
  --refinement-steps 1 \
  --steps 3 \
  --batch-size 2 \
  --microbatch-size 1 \
  --refiner-detach-steps 1 \
  --backbone-freeze-steps 0 \
  --backbone-warmup-end 1 \
  --ssr-learning-rate 2e-5 \
  --head-learning-rate 1e-6 \
  --backbone-learning-rate 0 \
  --eval-every 1 \
  --periodic-train-samples 2 \
  --eval-batch-size 2 \
  --local-scales 4 16 \
  --log-every 1 \
  --max-samples-per-split 2 \
  --seed 81 >"$LOGS/ddp_smoke_initial.log" 2>&1

"$ENVIRONMENT/bin/torchrun" --standalone --nproc-per-node=2 \
  -m moge.scripts.train_hypersim_joint_v3 \
  --data "$DATA" \
  --output "$OUTPUT" \
  --resume "$OUTPUT/resume_checkpoint.pt" \
  --safe-root "$SAFE_ROOT" \
  --ddp-backend gloo \
  --freeze-backbone \
  --height 64 \
  --width 64 \
  --num-tokens 64 \
  --refinement-steps 1 \
  --steps 4 \
  --batch-size 2 \
  --microbatch-size 1 \
  --refiner-detach-steps 1 \
  --backbone-freeze-steps 0 \
  --backbone-warmup-end 1 \
  --ssr-learning-rate 2e-5 \
  --head-learning-rate 1e-6 \
  --backbone-learning-rate 0 \
  --eval-every 1 \
  --periodic-train-samples 2 \
  --eval-batch-size 2 \
  --local-scales 4 16 \
  --log-every 1 \
  --max-samples-per-split 2 \
  --seed 81 >"$LOGS/ddp_smoke_resume.log" 2>&1

"$ENVIRONMENT/bin/python" - "$OUTPUT/report.json" <<'PY'
import json
import pathlib
import sys

report = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert report["status"] == "complete"
assert report["distributed"]["world_size"] == 2
assert report["distributed"]["global_batch_size"] == 2
assert report["distributed"]["local_batch_size"] == 1
assert report["optimization_steps"] == 4
assert report["resumed_from"]
assert report["backbone_frozen"]
assert report["learning_rates"]["backbone"] == 0
print("双卡DDP及断点恢复烟雾测试通过")
PY
